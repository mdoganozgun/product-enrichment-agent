# product_embedder.py
from typing import Iterable, Optional, List, Union
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# util’ten yardımcılar
from utils import build_product_texts, DEFAULT_TEXT_COLS


class ProductEmbedder:
    """
    Ürün metnini üretip (util.build_product_texts) SentenceTransformer ile vektöre çevirir.
    Manager yalnızca bu sınıfın metodlarını çağırır (orchestrator).
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    # -----------------------------
    # embedding metodları
    # -----------------------------
    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_cols: Optional[Iterable[str]] = None,
        tags_col: str = "tags",
        store_text_col: Optional[str] = "embedding_text",
        store_vec_col: str = "embedding",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        DataFrame’deki her satırı tek metne dönüştürüp vektörler.
        Dönen df: orijinal + (opsiyonel) metin kolonu + embedding kolonu (np.ndarray)
        """
        texts = build_product_texts(df, text_cols=text_cols or DEFAULT_TEXT_COLS, tags_col=tags_col)

        if store_text_col:
            df = df.copy()
            df[store_text_col] = texts

        embeddings: np.ndarray = self.model.encode(texts, show_progress_bar=show_progress)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        df[store_vec_col] = list(embeddings)
        return df

    def embed_text(self, text: str) -> np.ndarray:
        """Tek bir metni encode eder, (D,) np.ndarray döndürür."""
        vec = self.model.encode([text], show_progress_bar=False)
        return np.asarray(vec[0], dtype=np.float32)

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Çoklu metni encode eder, (N, D) np.ndarray döndürür."""
        vecs = self.model.encode(texts, show_progress_bar=show_progress)
        return np.asarray(vecs, dtype=np.float32)

    def get_product_embedding(
        self,
        df: pd.DataFrame,
        product_id: Union[str, int],
        id_col: str = "StockCode",
        vec_col: str = "embedding",
    ) -> Optional[np.ndarray]:
        """DataFrame içinden tek ürün embedding’ini döndürür; yoksa None."""
        row = df[df[id_col].astype(str) == str(product_id)]
        if row.empty or vec_col not in row.columns:
            return None
        vec = row.iloc[0][vec_col]
        return np.asarray(vec, dtype=np.float32) if vec is not None else None

    # -----------------------------
    # Manager’ın çağıracağı “yüksek seviye” yardımcı
    # -----------------------------

    def generate_and_cache_embeddings(
            self,
            df: pd.DataFrame,
            out_pkl_path: str,
            *,
            id_col: str = "StockCode",
            text_cols: Optional[Iterable[str]] = None,
            tags_col: str = "tags",
            store_text_col: Optional[str] = "embedding_text",
            store_vec_col: str = "embedding",
            keep_existing: bool = True,
            batch_size: int = 32,
            show_progress: bool = True,
            sort_output: bool = True,
    ) -> str:
        """
        Verilen ürün DataFrame’ini embed eder ve pickle’a kaydeder.
        Dönüş: yazılan pickle yolu.
        """
        import os

        # --- Girdi doğrulama ---
        if id_col not in df.columns:
            raise ValueError(f"'{id_col}' kolonu bulunamadı.")
        df = df.copy()
        df[id_col] = df[id_col].astype(str)

        # (Opsiyonel) text_cols doğrulaması
        if text_cols is not None:
            missing_cols = [c for c in text_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Text columns not found in DataFrame: {missing_cols}")

        # --- Mevcut cache ---
        if keep_existing and os.path.exists(out_pkl_path):
            df_existing = pd.read_pickle(out_pkl_path)
            if id_col not in df_existing.columns:
                raise ValueError(f"Existing cache at {out_pkl_path} is missing required column '{id_col}'.")
            df_existing = df_existing.copy()
            df_existing[id_col] = df_existing[id_col].astype(str)
            existing_ids = set(df_existing[id_col])
        else:
            df_existing = pd.DataFrame()
            existing_ids = set()

        # --- Tekilleştir & yeni kayıtlar ---
        df = df.drop_duplicates(subset=[id_col])
        if existing_ids:
            df = df[~df[id_col].isin(existing_ids)]

        # Boş durumları erken ele al
        if df.empty:
            # Eski cache varsa olduğu gibi koru; yoksa boş bir df yaz
            if df_existing.empty:
                dirpath = os.path.dirname(out_pkl_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                # En azından id ve embedding kolonları olan boş şema yazmak faydalı
                empty_out = pd.DataFrame(columns=[id_col, store_vec_col] + ([store_text_col] if store_text_col else []))
                empty_out.to_pickle(out_pkl_path)
            return out_pkl_path

        # --- Metinleri oluştur & encode et ---
        texts = build_product_texts(df, text_cols=text_cols or DEFAULT_TEXT_COLS, tags_col=tags_col)
        if store_text_col:
            df[store_text_col] = texts

        embeddings = self.model.encode(texts, show_progress_bar=show_progress, batch_size=batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        df[store_vec_col] = list(embeddings)

        # --- Birleştir, tekilleştir, sırala ---
        df_out = pd.concat([df_existing, df], ignore_index=True) if not df_existing.empty else df
        df_out = df_out.drop_duplicates(subset=[id_col])
        if sort_output:
            df_out = df_out.sort_values(by=id_col).reset_index(drop=True)

        # --- Kaydet ---
        dirpath = os.path.dirname(out_pkl_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        df_out.to_pickle(out_pkl_path)
        return out_pkl_path