"""FastAPI server for product validation."""
import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

from config import PROJECT_ID, LOCATION, API_KEY, get_google_credentials
from app.models import VerdictResponse
from app.services.evaluator import evaluate_product, evaluate_product_llm_only

# Initialize logging
DEBUG = os.getenv("DEBUG", "0") in {"1", "true", "True"}
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

#Initialize Vertex AI client (cheap) – defer model load to first request
credentials = get_google_credentials()
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Defaults (can be overridden per request)
DEFAULT_EMBEDDING_DIM = 512
DEFAULT_SIM_LOW = 0.08
DEFAULT_SIM_HIGH = 0.4

_mm_model = None

def get_mm_model() -> MultiModalEmbeddingModel:
    global _mm_model
    if _mm_model is None:
        logging.info("Loading multimodal embedding model 'multimodalembedding@001'...")
        _mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        logging.info("Multimodal embedding model loaded.")
    return _mm_model

app = FastAPI(
    title="Product Validator API",
    description="Validate product images against titles and descriptions using embeddings + LLM.",
    version="1.0.0"
)


def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    """Verify API key from x-api-key header."""
    if not API_KEY:
        # If no API_KEY is set in env, skip validation (dev mode)
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/")
def root():
    return {"message": "Product Validator API", "version": "1.0.0"}


@app.post("/evaluate", response_model=VerdictResponse, dependencies=[Depends(verify_api_key)])
async def evaluate_endpoint(
    image: UploadFile = File(..., description="Product image (JPEG/PNG)"),
    title: str = Form(..., description="Product title"),
    description: str = Form(..., description="Product description"),
    embedding_dim: int = Form(DEFAULT_EMBEDDING_DIM, description="Embedding dimension (128, 256, 512, 1408)"),
    sim_low: float = Form(DEFAULT_SIM_LOW, description="Lower similarity threshold (auto-fail)"),
    sim_high: float = Form(DEFAULT_SIM_HIGH, description="Upper similarity threshold (auto-pass)"),
):
    """
    Evaluate product using embeddings-first approach with gray-zone LLM fallback.

    - Computes image-title, image-description, title-description similarities.
    - Pass if all sims >= sim_high; fail if both image sims <= sim_low; otherwise gray-zone LLM.
    - Thresholds can be tuned per request for different product categories.
    """
    try:
        image_bytes = await image.read()
        verdict = evaluate_product(
            image_bytes=image_bytes,
            title_text=title,
            description_text=description,
            mm_model=get_mm_model(),
            dim=embedding_dim,
            sim_low=sim_low,
            sim_high=sim_high
        )
        return JSONResponse(content=verdict, status_code=200)
    except Exception as e:
        logging.error(f"Error in /evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/llm-only", response_model=VerdictResponse, dependencies=[Depends(verify_api_key)])
async def evaluate_llm_only_endpoint(
    image: UploadFile = File(..., description="Product image (JPEG/PNG)"),
    title: str = Form(..., description="Product title"),
    description: str = Form(..., description="Product description"),
):
    """
    Evaluate product using LLM-only (skip embeddings).

    - Directly calls Gemini to compare image, title, and description.
    - Returns verdict with conflicts and pair disagreements.
    """
    try:
        image_bytes = await image.read()
        verdict = evaluate_product_llm_only(
            image_bytes=image_bytes,
            title_text=title,
            description_text=description
        )
        return JSONResponse(content=verdict, status_code=200)
    except Exception as e:
        logging.error(f"Error in /evaluate/llm-only: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=DEBUG)

