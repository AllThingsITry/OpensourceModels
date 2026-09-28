"""
SageMaker → vLLM adapter.

SageMaker calls POST /invocations with the raw request body.
This script:
  1. Starts vLLM as a background process on port 8000
  2. Exposes SageMaker's required /ping and /invocations on port 8080
  3. Proxies /invocations straight through to vLLM's /v1/chat/completions
"""

import subprocess, os, time, httpx
from fastapi import FastAPI, Request, Response
import uvicorn
import sys

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")
VLLM_PORT = 8000
SM_PORT   = int(os.environ.get("PORT", 8080))

app = FastAPI()
vllm_proc = None

def start_vllm():
    global vllm_proc
    cmd = [
        "vllm", "serve",
        MODEL_DIR,
        "--served-model-name", "qwen3-8b",
        "--port", str(VLLM_PORT),
        "--max-model-len", "8192",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "1",
    ]
    
    vllm_proc = subprocess.Popen(
    cmd,
    stdout=None,
    stderr=None,
)

    # Wait until vLLM is healthy (up to 5 minutes)
    for _ in range(60):
        try:
            r = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=3)
            if r.status_code == 200:
                print("vLLM is ready.")
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("vLLM failed to start within 5 minutes.")

@app.on_event("startup")
async def startup():
    start_vllm()

@app.get("/ping")
def ping():
    """SageMaker health check — returns 200 when vLLM is ready."""
    try:
        r = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
        return Response(status_code=r.status_code)
    except Exception:
        return Response(status_code=503)

@app.post("/invocations")
async def invocations(request: Request):
    """
    Proxy SageMaker inference calls to vLLM.
    Accepts the OpenAI chat completions payload directly.
    """
    body = await request.body()
    headers = {"Content-Type": "application/json"}
    content_type = request.headers.get("content-type", "application/json")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            content=body,
            headers=headers,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )
if __name__ == "__main__":
    print("Starting SageMaker vLLM adapter...")
    print("Arguments:", sys.argv)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SM_PORT
    )