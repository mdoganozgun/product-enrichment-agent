from pydantic import BaseModel
from typing import List

class ProductEnrichment(BaseModel):
    category: str                 # Genel ürün kategorisi
    usage_context: str            # Kullanım yeri veya amacı
    price_segment: str            # low / mid / high
    material_type: str            # pamuk, plastik, ahşap, vb.
    target_gender: str            # female / male / unisex
    target_age_group: str         # children / adults / all ages
    tags: List[str]               # Anahtar kelime listesi (5–10)