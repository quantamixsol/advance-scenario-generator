# modules/embeddings.py
"""Embedding utilities and similarity search helpers."""

import numpy as np
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def get_embedder(model_name: str):
    """Return (embed_texts, dim, description) for the chosen model."""
    model_name = model_name.lower()

    if model_name == "baconnier":
        bc = SentenceTransformer("baconnier/Finance2_embedding_small_en-V1.5")
        dim = bc.get_sentence_embedding_dimension()
        desc = "\ud83d\udd52 Light & Fast (30 MB, CPU-friendly)"

        def embed_texts(texts: list[str]) -> np.ndarray:
            embs = bc.encode(texts, convert_to_numpy=True)
            return normalize(embs, axis=1)

        return embed_texts, dim, desc

    if model_name == "trading_hero":
        tokenizer = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
        model = AutoModel.from_pretrained(
            "fuchenru/Trading-Hero-LLM", output_hidden_states=True
        ).eval()
        dim = model.config.hidden_size
        desc = "\ud83e\udde0 Dense & Deep (~700 MB, richer semantics)"

        def embed_texts(texts: list[str]) -> np.ndarray:
            batches = []
            for i in range(0, len(texts), 32):
                chunk = texts[i : i + 32]
                toks = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out = model(**toks, output_hidden_states=True)
                    last = out.hidden_states[-1]
                    mask = toks.attention_mask.unsqueeze(-1)
                    summed = (last * mask).sum(dim=1)
                    cnts = mask.sum(dim=1).clamp(min=1)
                    embs = (summed / cnts).cpu().numpy()
                batches.append(embs)
            embs = np.vstack(batches)
            return normalize(embs, axis=1)

        return embed_texts, dim, desc

    raise ValueError(
        f"Unsupported embedder '{model_name}'. Choose 'baconnier' or 'trading_hero'."
    )


class _NumpyIndex:
    """Simple numpy-based fallback if faiss is unavailable."""

    def __init__(self, dim: int):
        self.embs = np.zeros((0, dim), dtype="float32")

    def add(self, x: np.ndarray):
        self.embs = np.asarray(x, dtype="float32")

    def search(self, q: np.ndarray, k: int):
        sims = self.embs @ q.T
        order = np.argsort(-sims, axis=0)[:k].T
        D = np.take_along_axis(sims.T, order, axis=1)
        return D.astype("float32"), order.astype("int64")


def _new_index(dim: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dim)
    return _NumpyIndex(dim)


def build_index(texts: list[str], *, embedder: str = "baconnier"):
    """Embed texts and build a similarity index."""
    embed_texts, dim, _ = get_embedder(embedder)
    embs = embed_texts(texts).astype("float32")
    idx = _new_index(dim)
    idx.add(embs)
    return idx, embs


def query_index(idx, embs: np.ndarray, query: str, k: int, *, embedder: str = "baconnier"):
    """Return indices and scores of top-k matches for the query."""
    embed_texts, _, _ = get_embedder(embedder)
    q = embed_texts([query]).astype("float32")
    D, I = idx.search(q, k)
    return I[0].tolist(), D[0].tolist()
