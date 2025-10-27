from typing import List, Optional, Literal
from pydantic import BaseModel


class AttributesModel(BaseModel):
    brand: Optional[str] = None
    product_type: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None


class ConflictModel(BaseModel):
    attribute: str
    source_pair: Optional[Literal['image_title', 'image_description', 'title_description']] = None
    title_value: Optional[str] = None
    image_value: Optional[str] = None
    description_value: Optional[str] = None
    severity: Optional[Literal['minor', 'major']] = None
    comment: Optional[str] = None


class LlmVerdictModel(BaseModel):
    verdict: Literal['pass', 'review', 'fail']
    conflicts: List[ConflictModel] = []
    pair_disagreements: List[Literal['image_title', 'image_description', 'title_description']] = []
    support: Optional[dict] = None
    notes: Optional[str] = None


class VerdictResponse(BaseModel):
    image_title_similarity: float
    image_description_similarity: float
    title_description_similarity: float
    llm_verdict: Optional[LlmVerdictModel] = None
    flags: List[str]
    decision: Literal['pass', 'review', 'fail']
    reasons: List[str]


