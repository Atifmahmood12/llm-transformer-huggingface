import random
import numpy as np
import torch
from fastapi import FastAPI
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
    do_sample: bool = False  # Default is deterministic (greedy decoding)
    seed: int | None = None  # Optional seed for deterministic output
    repetition_penalty: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

# Instantiate a handler for prompt processing and profiling
handler = PromptHandler()

@app.post("/generate")
async def generate_text(request: PromptRequest):
    """
    Generate text using the underlying LLM and profile the system's resource usage.
    - If a seed is provided, set all random seeds for deterministic output (repeatable results).
    - Otherwise, allow natural randomness for creative outputs.
    - Pass all relevant generation parameters to the handler/model
    """
    if request.seed is not None:
        # Setting seeds ensures reproducibility across torch, numpy, and random
        torch.manual_seed(request.seed)
        random.seed(request.seed)
        np.random.seed(request.seed)
    # If seed is None and do_sample=True, outputs will be different each time

    result = handler.handle_prompt(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        do_sample=request.do_sample,
        repetition_penalty=request.repetition_penalty,
        top_k=request.top_k,
        top_p=request.top_p
    )
    return result

@app.get("/")
async def root():
    """
    Health check endpoint for the API.
    """
    return {"message": "LLM Inference API is running!"}