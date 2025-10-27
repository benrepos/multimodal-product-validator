"""CLI demo for product validation."""
import os
import json
import logging
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

from config import PROJECT_ID, LOCATION, get_google_credentials
from app.services.evaluator import evaluate_product

# Initialize logging
DEBUG = os.getenv("DEBUG", "0") in {"1", "true", "True"}
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

# Initialize Vertex AI
credentials = get_google_credentials()
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Test configuration (tune per product category)
EMBEDDING_DIM = 512
SIM_LOW = 0.08
SIM_HIGH = 0.4

mm_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Test data
image_path = "test/screwdriver.jpeg"
title_text = "Phillips head screwdriver"
description_text = (
    "Phillips head screwdriver with a distinctive red and black ergonomic handle, isolated against a plain "
    "white background. Consisting of a metal shaft that ends in a cross-shaped tip, designed specifically "
    "for driving or removing Phillips screws."
)

print(f"Image: {image_path}")
print(f"Title: {title_text}")
print(f"Description: {description_text[:80]}...")
print(f"\nEmbedding dimension: {EMBEDDING_DIM}")
print(f"Thresholds: sim_low={SIM_LOW}, sim_high={SIM_HIGH}\n")

# Read image
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Evaluate
print("=== Verdict (pipeline) ===")
verdict = evaluate_product(
    image_bytes=image_bytes,
    title_text=title_text,
    description_text=description_text,
    mm_model=mm_model,
    dim=EMBEDDING_DIM,
    sim_low=SIM_LOW,
    sim_high=SIM_HIGH
)
print(json.dumps(verdict, indent=2, ensure_ascii=False))
