import asyncio
import base64
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError
import io
import json
import logging
from pydantic import BaseModel, field_validator
from PIL import Image
import soundfile as sf
import sys
from typing import List, Optional, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer, GenerationConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logging.info("[model] Loading model...")

model_path = "/models/Phi-4-multimodal-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    _attn_implementation='eager',
).to("cpu")
model = torch.compile(model)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

logging.info("[model] Model loaded.")

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"[request] Validation error for request {request.url}:\n{exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Phi-4-multimodal-instruct",
                "object": "model"
            }
        ]
    }

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    
    @field_validator('content', mode='before')
    @classmethod
    def ensure_list(cls, v):
        match v:
            case str():
                return [{"type": "text", "text": v}]
            case list():
                return v
            case _:
                raise ValueError("[request] Invalid content type: content must be a string or a list of dictionaries")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

def build_prompt_and_inputs(messages):
    images = []
    audios = []
    new_messages = []
    
    image_idx = 1
    audio_idx = 1
    
    for msg in messages:
        content_str = ""
        for content in msg["content"]:
            match content:
                case {"type": "image", "data": data}:
                    content_str += f"<|image_{image_idx}|>"
                    img_data = base64.b64decode(data)
                    images.append(Image.open(io.BytesIO(img_data)))
                    image_idx += 1
                case {"type": "audio", "data": data}:
                    content_str += f"<|audio_{audio_idx}|>"
                    audio_data = base64.b64decode(data)
                    audio, samplerate = sf.read(io.BytesIO(audio_data))
                    audios.append((audio, samplerate))
                    audio_idx += 1
                case {"type": "text", "text": text}:
                    content_str += text
                case _:
                    pass
        new_messages.append({"role": msg["role"], "content": content_str})

    return new_messages, images, audios

async def normal_completion(generate_kwargs):
    try:
        logging.info("[model] Generating response...")
        outputs = await asyncio.to_thread(
            model.generate,
            **generate_kwargs,
        )
        logging.info("[model] Generation task done.")
    except Exception as e:
        logging.error(f"[model] Error during generation: {e}")
        raise e

    decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)
    return decoded_output[0]

async def stream_completion(generate_kwargs):
    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    async def generate_text():
        try:
            await asyncio.to_thread(
                model.generate,
                **generate_kwargs,
                streamer=streamer,
            )
        except Exception as e:
            logging.error(f"[model] Error during generation: {e}")
            raise e

    generate_task = asyncio.create_task(generate_text())

    async def stream():
        logging.info("[stream] Starting streaming response...")
        last_reply = ""
        last_sent = ""
        for new_text in streamer:
            last_reply += new_text
            logging.debug(f"[stream] last_reply={last_reply}")
            to_send = last_reply[len(last_sent):]
            if to_send:
                data = {
                    "choices": [{
                        "delta": {"content": to_send},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                logging.debug(f"[stream] Sending data: {data}")
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_sent = last_reply
                await asyncio.sleep(0)
        logging.info("[stream] Generation task done, sending final response...")
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        await asyncio.sleep(0)

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    conversation = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]
    
    logging.debug(f"[completion] Received request: {request.json()}")
    
    new_conversation, images, audios = build_prompt_and_inputs(conversation)
    
    prompt = processor.tokenizer.apply_chat_template(
        new_conversation, tokenize=False, add_generation_prompt=True
    )
    
    logging.debug(f"[completion] Prompt: {prompt}")
    
    processor_kwargs = {"text": prompt, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    if audios:
        processor_kwargs["audio"] = audios
    
    inputs = processor(**processor_kwargs).to(model.device)
    inputs = {k: v for k, v in inputs.items()
              if v is not None and not (hasattr(v, 'numel') and v.numel() == 0)}

    generate_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "do_sample": True,
        "generation_config": generation_config,
        # https://huggingface.co/microsoft/Phi-4-multimodal-instruct/discussions/46
        "num_logits_to_keep": 1
    }

    if request.stream:
        return await stream_completion(generate_kwargs)
    else:
        response = await normal_completion(generate_kwargs)
        return JSONResponse(content={
            "id": "phi4-mm-1234",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }]
        })
