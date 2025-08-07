from pydantic import BaseModel
from typing import List, Optional

class ProductEnrichment(BaseModel):
    category: Optional[str]                 # Genel ürün kategorisi
    sub_category: Optional[str]             # Daha spesifik alt kategori (örnek: lamp, vase)
    usage_context: Optional[str]            # Kullanım yeri veya amacı
    price_segment: Optional[str]            # low / mid / high
    material_type: Optional[str]            # pamuk, plastik, ahşap, vb.
    target_gender: Optional[str]            # female / male / unisex
    target_age_group: Optional[str]         # children / adults / all ages
    tags: List[str]               # Anahtar kelime listesi (5–10)