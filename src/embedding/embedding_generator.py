import openai
from typing import List, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
import logging


class EmbeddingGenerator:
    """
    A modular embedding generator that supports batching and retry logic.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or openai.api_key
        openai.api_key = self.api_key
        self.logger = logging.getLogger(__name__)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Internal method to get embeddings from OpenAI for a batch of texts.
        Automatically retries on transient errors.
        """
        response = openai.Embedding.create(
            input=batch,
            model=self.model_name
        )
        return [item["embedding"] for item in response["data"]]

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Splits input into batches and gets embeddings for each batch.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.info(f"🔢 Embedding batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings