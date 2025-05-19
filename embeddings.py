# embeddings.py
# ──────────────────────────────────────────────────────────────────────────────
# To install dependencies, run:
# pip install sentence-transformers transformers torch scikit-learn faiss-cpu

import numpy as np
import torch
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def get_embedder(model_name: str):
    """
    Factory to return (embed_texts, dim, description) for the given model.
    
    model_name: 
      - "baconnier"       → Baconnier Finance_small (30 MB, fast)
      - "trading_hero"    → Trading-Hero-LLM (dense, ~700 MB)
    """
    model_name = model_name.lower()
    
    if model_name == "baconnier":
        bc = SentenceTransformer("baconnier/Finance2_embedding_small_en-V1.5")
        dim = bc.get_sentence_embedding_dimension()
        desc = "🕒 Light & Fast (30 MB, CPU-friendly)"

        def embed_texts(texts: list[str]) -> np.ndarray:
            embs = bc.encode(texts, convert_to_numpy=True)
            return normalize(embs, axis=1)
        
        return embed_texts, dim, desc

    elif model_name == "trading_hero":
        tokenizer = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
        model     = AutoModel.from_pretrained(
                        "fuchenru/Trading-Hero-LLM",
                        output_hidden_states=True
                    ).eval()
        dim = model.config.hidden_size
        desc = "🧠 Dense & Deep (~700 MB, richer semantics)"

        def embed_texts(texts: list[str]) -> np.ndarray:
            batches = []
            for i in range(0, len(texts), 32):
                chunk = texts[i : i+32]
                toks  = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    out   = model(**toks, output_hidden_states=True)
                    last  = out.hidden_states[-1]               # (B, S, H)
                    mask  = toks.attention_mask.unsqueeze(-1)   # (B, S, 1)
                    summed = (last * mask).sum(dim=1)           # (B, H)
                    cnts   = mask.sum(dim=1).clamp(min=1)       # (B, 1)
                    embs   = (summed / cnts).cpu().numpy()      # (B, H)
                batches.append(embs)
            embs = np.vstack(batches)
            return normalize(embs, axis=1)
        
        return embed_texts, dim, desc

    else:
        raise ValueError(f"Unsupported embedder '{model_name}'. Choose 'baconnier' or 'trading_hero'.")


# ─── 4) SELF‐TEST (run `python embeddings.py`) ──────────────────────

if __name__ == "__main__":
    texts = [
        "Credit spreads widen by 50 basis points.",
        "Interest rate curve steepens in the US Treasury market."
    ]

    # Test Baconnier
    print("→ Testing Baconnier embedder…")
    embed_bac, dim_bac, desc_bac = get_embedder("baconnier")
    em1 = embed_bac(texts)
    print(f"  desc: {desc_bac}")
    print("  shape:", em1.shape, "norms:", np.round(np.linalg.norm(em1, axis=1), 3))

    # Test Trading Hero
    print("\n→ Testing Trading Hero embedder…")
    embed_th, dim_th, desc_th = get_embedder("trading_hero")
    em2 = embed_th(texts)
    print(f"  desc: {desc_th}")
    print("  shape:", em2.shape, "norms:", np.round(np.linalg.norm(em2, axis=1), 3))
