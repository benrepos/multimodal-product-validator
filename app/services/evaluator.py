"""Evaluation pipeline for product validation."""
import os
import logging
import numpy as np
from vertexai.vision_models import Image, MultiModalEmbeddingModel

from app.services.llm import llm_compare_image_title_description

DEBUG = os.getenv("DEBUG", "0") in {"1", "true", "True"}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / den if den != 0.0 else 0.0


def evaluate_product(
    image_bytes: bytes,
    title_text: str,
    description_text: str,
    mm_model: MultiModalEmbeddingModel,
    dim: int,
    sim_low: float = 0.08,
    sim_high: float = 0.25
) -> dict:
    """Run full evaluation and return a structured verdict.

    Returns:
      {
        "image_title_similarity": float,
        "image_description_similarity": float,
        "title_description_similarity": float,
        "llm_verdict": dict | None,
        "flags": [str],
        "decision": "pass" | "review" | "fail",
        "reasons": [str]
      }
    """
    result = {
        "image_title_similarity": None,
        "image_description_similarity": None,
        "title_description_similarity": None,
        "llm_verdict": None,
        "flags": [],
        "decision": "review",
        "reasons": [],
    }

    # Load image from bytes
    image = Image(image_bytes=image_bytes)

    # Embed image once
    e_title = mm_model.get_embeddings(image=image, contextual_text=title_text, dimension=dim)
    img_vec = np.asarray(e_title.image_embedding, dtype=np.float32)
    title_vec = np.asarray(e_title.text_embedding, dtype=np.float32)
    # Compute image-title
    sim_it = cosine_sim(img_vec, title_vec)
    result["image_title_similarity"] = sim_it
    # Compute image-description
    e_desc = mm_model.get_embeddings(image=image, contextual_text=description_text, dimension=dim)
    desc_vec = np.asarray(e_desc.text_embedding, dtype=np.float32)
    sim_id = cosine_sim(img_vec, desc_vec)
    result["image_description_similarity"] = sim_id
    # Compute title-description (text-text)
    td_title = mm_model.get_embeddings(contextual_text=title_text, dimension=dim)
    td_desc = mm_model.get_embeddings(contextual_text=description_text, dimension=dim)
    sim_td = cosine_sim(
        np.asarray(td_title.text_embedding, dtype=np.float32),
        np.asarray(td_desc.text_embedding, dtype=np.float32)
    )
    result["title_description_similarity"] = sim_td

    # Embeddings-first decision gates
    if sim_it >= sim_high and sim_id >= sim_high and sim_td >= sim_high:
        result["decision"] = "pass"
        result["reasons"].append(f"all sims ≥ {sim_high}")
        return result
    if sim_it <= sim_low and sim_id <= sim_low:
        result["flags"].append("LOW_IMAGE_SIMS")
        result["decision"] = "fail"
        result["reasons"].append(f"image-title {sim_it:.3f} and image-description {sim_id:.3f} ≤ {sim_low}")
        return result

    # Gray zone: call LLM comparator only here
    result["flags"].append("GRAY_ZONE_LLM")
    result["reasons"].append(f"gray zone: sims it={sim_it:.3f} id={sim_id:.3f} td={sim_td:.3f}")
    if DEBUG:
        logging.debug(f"[LLM] Gray zone detected. Invoking LLM comparator...")
    llm = llm_compare_image_title_description(image_bytes, title_text, description_text)
    result["llm_verdict"] = llm
    if DEBUG:
        logging.debug(f"[LLM] Comparator verdict: {llm.get('verdict')} | conflicts={len(llm.get('conflicts', []))}")
    if llm.get("verdict") == "fail":
        result["decision"] = "fail"
        result["reasons"].append("llm verdict fail")
    elif llm.get("verdict") == "pass" and not llm.get("conflicts"):
        result["decision"] = "pass"
        result["reasons"].append("llm verdict pass")
    else:
        result["decision"] = "review"
        if llm.get("conflicts"):
            conflict_attrs = ", ".join({c.get("attribute", "?") for c in llm["conflicts"]})
            result["reasons"].append(f"llm conflicts: {conflict_attrs}")

    return result


def evaluate_product_llm_only(
    image_bytes: bytes,
    title_text: str,
    description_text: str
) -> dict:
    """Run LLM-only evaluation (skip embeddings).

    Returns:
      {
        "image_title_similarity": None,
        "image_description_similarity": None,
        "title_description_similarity": None,
        "llm_verdict": dict,
        "flags": ["LLM_ONLY"],
        "decision": "pass" | "review" | "fail",
        "reasons": [str]
      }
    """
    result = {
        "image_title_similarity": None,
        "image_description_similarity": None,
        "title_description_similarity": None,
        "llm_verdict": None,
        "flags": ["LLM_ONLY"],
        "decision": "review",
        "reasons": [],
    }

    llm = llm_compare_image_title_description(image_bytes, title_text, description_text)
    result["llm_verdict"] = llm
    if llm.get("verdict") == "fail":
        result["decision"] = "fail"
        result["reasons"].append("llm verdict fail")
    elif llm.get("verdict") == "pass" and not llm.get("conflicts"):
        result["decision"] = "pass"
        result["reasons"].append("llm verdict pass")
    else:
        result["decision"] = "review"
        if llm.get("conflicts"):
            conflict_attrs = ", ".join({c.get("attribute", "?") for c in llm["conflicts"]})
            result["reasons"].append(f"llm conflicts: {conflict_attrs}")

    return result

