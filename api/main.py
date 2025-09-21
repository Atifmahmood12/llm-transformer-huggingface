from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.prompt_handler import PromptHandler

app = FastAPI(
    title="LLM Inference & Profiling API",
    description="Generate text and profile resource usage with Hugging Face transformers",
    version="0.1.0"
)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    do_sample: bool = True

handler = PromptHandler()

@app.post("/generate")
async def generate_text(request: PromptRequest):
    result = handler.handle_prompt(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        do_sample=request.do_sample
    )
    return result

@app.get("/")
async def root():
    return {"message": "LLM Inference API is running!"}