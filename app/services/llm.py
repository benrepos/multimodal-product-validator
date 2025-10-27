"""LLM helpers for product validation."""
import os
import json
import logging
from typing import Optional
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from pydantic import ValidationError

from config import GEMINI_MODEL_ID
from app.models import LlmVerdictModel

DEBUG = os.getenv("DEBUG", "0") in {"1", "true", "True"}


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return {}
        return {}


def generate_with_fallback(parts_or_prompt, generation_config: Optional[GenerationConfig] = None):
    """Generate content using Gemini model from env."""
    if DEBUG:
        logging.debug(f"[LLM] Using Gemini model: {GEMINI_MODEL_ID}")
    model = GenerativeModel(GEMINI_MODEL_ID)
    if generation_config is not None:
        return model.generate_content(parts_or_prompt, generation_config=generation_config)
    return model.generate_content(parts_or_prompt)


def llm_compare_image_title_description(image_bytes: bytes, title_text: str, description_text: str) -> dict:
    """Use Gemini to compare IMAGE vs TITLE vs DESCRIPTION and return a structured verdict.

    Returns a dict like:
    {
      "verdict": "pass" | "review" | "fail",
      "conflicts": [...],
      "pair_disagreements": [...],
      "support": {...},
      "notes": str
    }
    """
    image_part = Part.from_data(mime_type="image/jpeg", data=image_bytes)
    instructions = (
        "You are a product validator. Compare IMAGE, TITLE, and DESCRIPTION.\n"
        "- If product categories differ (e.g., screwdriver vs masking tape), set verdict=fail.\n"
        "- Use review only for minor uncertainty.\n"
        "- Identify conflicts across brand, product_type, color, material.\n"
        "- Indicate which pair(s) disagree: image_title, image_description, title_description.\n"
        "- Provide concise conflicts with values and a short comment.\n"
        "- Provide minimal supporting attributes for all sources.\n"
        "Return ONLY JSON matching this schema exactly (no extra text):\n"
        "{\n  \"verdict\": \"pass|review|fail\",\n  \"conflicts\": [\n    {\"attribute\": string, \"source_pair\": \"image_title|image_description|title_description\", \"title_value\": string|null, \"image_value\": string|null, \"description_value\": string|null, \"severity\": \"minor|major\", \"comment\": string}\n  ],\n  \"pair_disagreements\": [\"image_title\", \"image_description\", \"title_description\"],\n  \"support\": {\n    \"image_attributes\": {\"brand\": string|null, \"product_type\": string, \"color\": string|null, \"material\": string|null},\n    \"title_attributes\": {\"brand\": string|null, \"product_type\": string, \"color\": string|null, \"material\": string|null},\n    \"description_attributes\": {\"brand\": string|null, \"product_type\": string, \"color\": string|null, \"material\": string|null}\n  },\n  \"notes\": string\n}"
    )
    user = "TITLE: " + title_text + "\nDESCRIPTION: " + description_text
    gen_cfg = GenerationConfig(response_mime_type="application/json", temperature=0.0)
    resp = generate_with_fallback([image_part, instructions, user], generation_config=gen_cfg)
    raw_text = resp.text or ""
    data = _safe_json_loads(raw_text)
    if not isinstance(data, dict) or not data:
        data = {"verdict": "review", "conflicts": [], "support": {}, "notes": "empty_or_unparseable", "pair_disagreements": []}
    try:
        parsed = LlmVerdictModel(**data)
        return json.loads(parsed.json())
    except ValidationError as ve:
        if DEBUG:
            logging.debug(f"[LLM] Validation error: {ve}")
        return {
            "verdict": data.get("verdict", "review"),
            "conflicts": data.get("conflicts", []),
            "pair_disagreements": data.get("pair_disagreements", []),
            "support": data.get("support", {}),
            "notes": data.get("notes", "validation_error"),
        }
