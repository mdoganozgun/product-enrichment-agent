"""
Project package root.

Make sure 'src' is on PYTHONPATH so that:
    from embedding import CustomerEmbedder, ProductEmbedder
    from manager import EmbeddingManager
    from util import build_product_texts, half_life_decay, ...
works from notebooks.
"""
import embedding
import manager
import util
__all__ = ["embedding", "manager", "util"]
__version__ = "0.1.0"
