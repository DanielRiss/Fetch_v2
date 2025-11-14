#!/usr/bin/env python3
"""
vLLM Policy Server Integration

This wraps PolicyInference as a vLLM-compatible API endpoint.
Allows seamless integration with existing vLLM-based infrastructure.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import torch

from policy_inference import PolicyInference


app = FastAPI()


class CompletionRequest(BaseModel):
    """Request format matching vLLM API"""
    prompt: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    include_stop_str_in_output: bool = True
    skip_special_tokens: bool = False


class CompletionResponse(BaseModel):
    """Response format matching vLLM API"""
    choices: List[dict]


class PolicyInferenceServer:
    """Wrapper for PolicyInference to work with vLLM-like API"""

    def __init__(self, model_path: str):
        """Initialize policy inference engine"""
        print(f"Loading policy model from {model_path}...")
        self.inference = PolicyInference(
            model_path=model_path,
            device="auto",
            torch_dtype=torch.bfloat16,
            verbose=False
        )
        print("✅ Policy model loaded!")

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt"""
        try:
            if "Input:" in prompt and "Possible next steps:" in prompt:
                input_part = prompt.split("Input:")[1].split("\n")[0]
                numbers = [float(x.strip()) for x in input_part.split()]

                # Call policy
                result = self.inference.infer(numbers)

                if result.num_moves > 0:
                    return '\n'.join(sorted(result.moves))
                else:
                    return ""  # Empty for 1-number states
            else:
                return ""
        except Exception as e:
            print(f"[POLICY ERROR]: {str(e)}")
            return ""



# Global inference engine
policy_server = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global policy_server
    import os
    model_path = os.environ.get("POLICY_MODEL_PATH", "ftm_go24_policy_merged")
    policy_server = PolicyInferenceServer(model_path)


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> CompletionResponse:
    """
    vLLM-compatible completions endpoint

    Takes same format as vLLM, returns same format
    """
    if policy_server is None:
        raise HTTPException(status_code=500, detail="Policy server not initialized")

    try:
        # Generate completion
        text = policy_server.complete(request.prompt)

        # Return vLLM-compatible response
        return CompletionResponse(
            choices=[{"text": text}]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def models():
    """List available models"""
    return {
        "data": [
            {
                "id": "ftm_go24_policy_merged",
                "object": "model",
                "created": 0,
                "owned_by": "vllm"
            }
        ]
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import os

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))

    print(f"\nStarting Policy Inference Server on {host}:{port}")
    print("=" * 70)

    uvicorn.run(app, host=host, port=port, workers=1)
