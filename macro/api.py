from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from macro.models import load_and_apply_weights # Import the function

app = FastAPI()

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: int = 50

class ComposeRequest(BaseModel):
    model_name: str
    expert_weights: Dict[str, float]

@app.post("/macro/generate/")
async def generate_text(request: GenerateRequest):
    # This is a placeholder. In a real scenario, this would interact with a model.
    if request.model_name == "llama-2-7b":
        generated_text = f"Generated text for {request.prompt} using {request.model_name}."
    else:
        raise HTTPException(status_code=400, detail="Model not supported.")
    return {"generated_text": generated_text}

@app.post("/macro/compose/")
async def compose_model(request: ComposeRequest):
    result = load_and_apply_weights(request.model_name, request.expert_weights)
    return {"message": f"Model {request.model_name} composed successfully.", "details": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)