import numpy as np
import types
import modules.embeddings as emb


def fake_get_embedder(name: str):
    def embed_texts(texts):
        arr = np.array([[len(t), sum(map(ord, t)) % 7, len(set(t))] for t in texts], dtype=np.float32)
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr
    return embed_texts, 3, "fake"


def setup_module(module):
    # Patch get_embedder with a deterministic fake
    module._orig_get = emb.get_embedder
    emb.get_embedder = fake_get_embedder


def teardown_module(module):
    emb.get_embedder = module._orig_get


def test_build_index_shapes_and_norms():
    texts = ["alpha", "beta", "gamma", "delta"]
    idx, embs = emb.build_index(texts)
    assert embs.shape == (4, 3)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_query_retrieval():
    texts = ["alpha", "beta", "gamma", "delta"]
    idx, embs = emb.build_index(texts)
    ids, scores = emb.query_index(idx, embs, "beta", 2)
    assert ids[0] == 1
    assert scores[0] >= scores[1]
