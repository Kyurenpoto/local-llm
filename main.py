import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Union
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, TextIteratorStreamer
from qwen_omni_utils import process_mm_info
import base64
import io
import soundfile as sf
import json
import asyncio

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error for request {request.url}:\n{exc.errors()}")
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
                "id": "Qwen2.5-Omni-3B",
                "object": "model"
            }
        ]
    }

print("Loading model...")

model_path = "/models/Qwen2.5-Omni-3B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cpu")
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

print("Model loaded.")

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
                raise ValueError("Invalid content type: content must be a string or a list of dictionaries")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0

def extract_last_assistant_reply(text):
    parts = text.split("assistant\n")
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    conversation = [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]
    
    print(f"Received request: {request.json()}")
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True
    ).to(model.device).to(model.dtype)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

    text_ids_holder = {}

    async def generate_text():
        text_ids, audio = await asyncio.to_thread(
            model.generate,
            **inputs,
            use_audio_in_video=True,
            max_time=60,
            streamer=streamer,
        )
        text_ids_holder["text_ids"] = text_ids
        text_ids_holder["audio"] = audio

    generate_task = asyncio.create_task(generate_text())

    async def stream():
        response_buffer = ""
        last_sent = ""
        while not generate_task.done() or streamer:
            for new_text in streamer:
                response_buffer += new_text
                last_reply = extract_last_assistant_reply(response_buffer)
                to_send = last_reply[len(last_sent):]
                data = {
                    "choices": [{
                        "delta": {"content": to_send},
                        "index": 0,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                last_sent = last_reply
            await asyncio.sleep(0.01)
        #audio = text_ids_holder.get("audio")
        #if audio is not None:
        #    buf = io.BytesIO()
        #    sf.write(buf, audio.cpu().numpy(), samplerate=24000, format="wav")
        #    buf.seek(0)
        #    audio_base64 = base64.b64encode(buf.read()).decode("utf-8")
        #    data = {
        #        "choices": [{
        #            "delta": {"audio": audio_base64},
        #            "index": 0,
        #            "finish_reason": None
        #        }]
        #    }
        #    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
