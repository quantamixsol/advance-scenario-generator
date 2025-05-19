# proxy_match.py
import faiss
from sklearn.preprocessing import normalize
from rapidfuzz import fuzz

def two_stage_match(px_row, universe_df, cfg, alpha, top_k):
    """
    1) apply any business-rule filters
    2) semantic FAISS ANN on full code embedding
    3) compute hybrid_score per candidate
    4) combine α*hybrid + (1-α)*semantic
    5) return best match
    """
    # stub: re-use your existing implementation
    pass

def hybrid_score(px, cd, cfg):
    # stub: copy your code
    pass
