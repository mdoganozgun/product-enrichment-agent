import logging
from typing import List, Optional
import numpy as np
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils import chunk_iter, total_batches, resolve_openai_api_key

class EmbeddingGenerator:
    """
    Minimal, side-effect free embedding helper.
    - Batch'ler halinde Embedding API çağrısı yapar
    - Retry politikası tenacity ile sarılı
    - np.ndarray döndürür
    """

    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.api_key = resolve_openai_api_key(api_key)
        self.client = openai  # OpenAI SDK v1 kullanıyorsanız burada client=OpenAI() gibi yapın
        openai.api_key = self.api_key  # isterseniz kaldırıp çağrıda 'api_key=self.api_key' ile geçebilirsiniz
        self.logger = logger or logging.getLogger(__name__)

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Tek bir batch için embedding döndürür (retry ile)."""
        resp = self.client.Embedding.create(model=self.model_name, input=texts)
        return [item["embedding"] for item in resp["data"]]

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Tüm metinleri batch'lere bölerek embed eder.
        Dönen şekil: (N, D) np.ndarray
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
w
        all_vecs: List[List[float]] = []
        n_batches = total_batches(len(texts), batch_size)

        for b_idx, batch in enumerate(chunk_iter(texts, batch_size), start=1):
            # isteğe bağlı sade log
            if self.logger:
                self.logger.info(f"Embedding batch {b_idx}/{n_batches} (size={len(batch)})")
            vecs = self._embed_batch(batch)
            all_vecs.extend(vecs)

        return np.asarray(all_vecs, dtype=np.float32)