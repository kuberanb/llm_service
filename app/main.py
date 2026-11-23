
from fastapi import FastAPI

app = FastAPI(
    title="LLM Service",
    description="Structured LLM generation with streaming",
    version="1.0.0"
)



@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }