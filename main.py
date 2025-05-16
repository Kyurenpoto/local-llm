import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Union
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import base64
import io
import soundfile as sf

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

    with torch.no_grad():
        text_ids, audio = model.generate(
            **inputs,
            use_audio_in_video=True,
            max_new_tokens=request.max_tokens,
            max_time=60
        )
        
    output_texts = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    #audio_base64 = None
    #if audio is not None:
    #    buf = io.BytesIO()
    #    sf.write(buf, audio.cpu().numpy(), samplerate=24000, format="wav")
    #    buf.seek(0)
    #    audio_base64 = base64.b64encode(buf.read()).decode("utf-8")

    response = {
        "id": "chatcmpl-1234",
        "object": "chat.completion",
        "choices": [
            {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": extract_last_assistant_reply(msg)
                },
                "finish_reason": "stop"
            }
            for i, msg in enumerate(output_texts)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": len(inputs["input_ids"][0]),
            "completion_tokens": text_ids.shape[-1],
            "total_tokens": len(inputs["input_ids"][0]) + text_ids.shape[-1]
        }
    }
    #if audio_base64:
    #    response["audio"] = audio_base64
    
    print(f"Response: {response}")

    return response
