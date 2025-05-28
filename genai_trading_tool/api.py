"""FastAPI interface for genai_trading_tool."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"message": "genai trading tool API"}
