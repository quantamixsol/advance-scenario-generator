#!/usr/bin/env python3
# matching.py

# ─── Standard lib imports ───────────────────────────────────────────
import argparse
import re

# ─── Third-party imports ────────────────────────────────────────────
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from rapidfuzz import fuzz

# ─── Your modules ───────────────────────────────────────────────────
from embeddings import get_embedder
from data_io   import parse_df, ASSET_CONFIG, REGION_GROUP, RATING_SCORE

# ─── Helpers ────────────────────────────────────────────────────────
def tenor_to_months(s: str) -> float:
    """Convert tenor like '5Y', '6M', 'ON' into months."""
    t = (s or "").strip().upper()
    if t == "ON":
        return 1/30
    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", t)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    return {"D":1/30, "W":7/30, "M":1, "Y":12}[unit] * val

def hybrid_score(px: pd.Series, cd: pd.Series, cfg: dict) -> float:
    """Compute hybrid per-field similarity between proxy px and candidate cd."""
    total, wsum = 0.0, 0.0
    for f, w in cfg["weights"].items():
        P, C = px.get(f, ""), cd.get(f, "")
        if f == "rating":
            sc = 1 - abs(RATING_SCORE.get(P,0) - RATING_SCORE.get(C,0)) / 5
        elif f in ("tenor","maturity"):
            a, b = tenor_to_months(P), tenor_to_months(C)
            sc = 1 - abs(a - b) / max(a, b, 1)
        elif f == "region":
            sc = 1 if P and C and (P == C or REGION_GROUP.get(P) == REGION_GROUP.get(C)) else 0
        elif f == "shock":
            sc = 1.0 if P and P == C else 0.0
        else:
            sc = 1.0 if P == C else fuzz.partial_ratio(str(P), str(C)) / 100
        total += w * sc
        wsum  += w
    return (total / wsum) if wsum else 0.0

def two_stage(px: pd.Series, universe: pd.DataFrame, cfg: dict,
              q_embed: np.ndarray, full_embs: np.ndarray, idx: faiss.IndexFlatIP,
              alpha: float, top_k: int) -> tuple[str,float]:
    """
    1) semantic ANN gives top_k candidates
    2) compute hybrid_score for each
    3) combine with semantic score and return best proxy match
    """
    D, I = idx.search(q_embed, top_k)
    valid = [i for i in I[0] if i < len(universe)]
    cand  = universe.iloc[valid].reset_index(drop=True)
    # hybrid
    h_scores = np.array([hybrid_score(px, row, cfg) for _,row in cand.iterrows()])
    # semantic
    s_scores = (full_embs[valid] @ q_embed[0]).flatten()
    combined = alpha * h_scores + (1 - alpha) * s_scores
    best_i   = int(np.nanargmax(combined))
    return cand.loc[best_i, "original"], float(combined[best_i])

def extract_risk_factors(narrative: str, universe: pd.DataFrame,
                         embed_texts, top_k: int, severity: str) -> pd.DataFrame:
    """Use semantic ANN to pull top_k RFs and assign default shocks by severity."""
    # build full‐code index
    full_idx, full_embs = build_fullcode_ann(universe, embed_texts)
    qn = embed_texts([narrative]).astype("float32")
    D, I = full_idx.search(qn, top_k)
    valid = [i for i in I[0] if i < len(universe)]
    rf = universe.iloc[valid].copy().reset_index(drop=True)
    rf["sim"]       = D[0][:len(valid)]
    default_map     = {"Low":1, "Medium":5, "High":10, "Extreme":20}
    rf["shock_pct"] = default_map.get(severity, 5)
    return rf

# ─── FAISS Builders ─────────────────────────────────────────────────
def build_ann_idx(df: pd.DataFrame, field: str, embed_texts) -> tuple[faiss.IndexFlatIP,np.ndarray]:
    embs = embed_texts(df[field].fillna("").astype(str).tolist()).astype("float32")
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, embs

def build_fullcode_ann(df: pd.DataFrame, embed_texts) -> tuple[faiss.IndexFlatIP,np.ndarray]:
    if df.empty:
        idx = faiss.IndexFlatIP(1)
        return idx, np.zeros((0,1), dtype="float32")
    embs = embed_texts(df["original"].fillna("").tolist()).astype("float32")
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx, embs

# ─── Main proxy & RF matching pipeline ─────────────────────────────
def proxy_match(px_df: pd.DataFrame, un_df: pd.DataFrame,
                embed_texts, dim: int, alpha: float, top_k: int,
                narrative: str, rf_k: int, severity: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    1) Extract RFs from narrative
    2) For each RF, find best proxy via two_stage()
    Returns: (df_rf, df_proxy_shocks)
    """
    # 1) limit universe to selected assets
    universe = un_df
    # 2) extract RFs
    df_rf = extract_risk_factors(narrative, universe, embed_texts, rf_k, severity)
    # 3) build proxy matches
    # precompute fullcode index on proxies for speed
    prox_idx, prox_embs = build_fullcode_ann(px_df, embed_texts)
    out = []
    for _, r in df_rf.iterrows():
        # filter proxies matching the same asset class
        px_sub = px_df[px_df["asset"]==r["asset"]].reset_index(drop=True)
        if px_sub.empty:
            continue
        # build ann on this subset
        idx, full = build_fullcode_ann(px_sub, embed_texts)
        q_embed  = embed_texts([r["original"]]).astype("float32")
        bp,score = two_stage(r, px_sub, ASSET_CONFIG[r["asset"]],
                             q_embed, full, idx, alpha, top_k)
        out.append({
            "asset":       r["asset"],
            "risk_factor": r["original"],
            "sim":         r["sim"],
            "shock_pct":   r["shock_pct"],
            "proxy":       bp,
            "match_score": round(score,4)
        })
    df_px = pd.DataFrame(out)
    return df_rf, df_px

# ─── Command-line interface ────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Scenario→RF→Proxy matching")
    p.add_argument("-n","--narrative", default="", help="Scenario narrative")
    p.add_argument("-r","--rf_k",       type=int, default=5, 
                   help="How many risk factors to extract")
    p.add_argument("-p","--proxy",      required=True, 
                   help="Path to proxies.csv")
    p.add_argument("-u","--universe",   required=True, 
                   help="Path to universe.csv")
    p.add_argument("-e","--embedder",   choices=["baconnier","trading_hero"],
                   default="trading_hero",
                   help="Which embedding model to use")
    p.add_argument("-a","--alpha",      type=float, default=0.6,
                   help="Hybrid vs semantic blend weight")
    p.add_argument("-k","--top_k",      type=int, default=10,
                   help="ANN candidate search size")
    p.add_argument("-s","--severity",   choices=["Low","Medium","High","Extreme"],
                   default="Medium", help="Scenario severity level")
    args = p.parse_args()

    print(f"→ Using embedder: {args.embedder}")
    embed_texts, dim, desc = get_embedder(args.embedder)
    print("   model info:", desc)

    # load & parse
    px_df = parse_df(pd.read_csv(args.proxy,  header=None, names=["code"]))
    un_df = parse_df(pd.read_csv(args.universe,header=None, names=["code"]))

    # run matching
    rf_df, px_df2 = proxy_match(
        px_df, un_df,
        embed_texts, dim,
        args.alpha, args.top_k,
        args.narrative, args.rf_k, args.severity
    )

    # output
    rf_out = "rf_shocks.csv"
    px_out = "proxy_shocks.csv"
    rf_df.to_csv(rf_out, index=False)
    px_df2.to_csv(px_out, index=False)
    print(f"→ Extracted RFs saved to {rf_out} ({len(rf_df)} rows)")
    print(f"→ Proxy shocks saved to {px_out} ({len(px_df2)} rows)")

if __name__=="__main__":
    main()
