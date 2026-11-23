
from fastapi import FastAPI

from app.routes.llm import llm_router

app = FastAPI(
    title="LLM Service",
    description="Structured LLM generation with streaming",
    version="1.0.0"
)

# Register LLM routers
app.include_router(
    llm_router,
    prefix="/api/v1/llm",
    tags=["LLM"]
)


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }