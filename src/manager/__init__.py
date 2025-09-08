# Keep manager as a thin orchestration layer
from .embedding_manager import EmbeddingManager

import os
base_dir = os.path.dirname(os.path.dirname(__file__))

# (Opsiyonel) varsa diğer yöneticileri güvenli şekilde re-export et
try:
    from .enrichment_manager import EnrichmentManager  # dosya varsa
except Exception:
    EnrichmentManager = None  # sessizce geç

__all__ = ["EmbeddingManager", "EnrichmentManager"]
