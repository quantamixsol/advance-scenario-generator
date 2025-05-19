# app.py
# ──────────────────────────────────────────────────────────────────────────────
# pip install streamlit transformers sentence-transformers torch faiss-cpu rapidfuzz scikit-learn plotly openai google-generativeai

import sys, asyncio, os, pathlib, re
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import torch
# silence torch._classes probing
try:
    if hasattr(torch, "_classes") and hasattr(torch._classes, "__path__"):
        del torch._classes.__path__
except:
    pass

import streamlit as st
st.set_page_config(page_title="🔍 Scenario Generator v27", layout="wide")

import pandas as pd
import numpy as np
import faiss
from rapidfuzz import fuzz
from sklearn.preprocessing import normalize
import plotly.express as pex

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from generative import generate_narrative, explain_factors

# ─── MODEL LOADERS ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_trading_hero():
    tok = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
    mdl = AutoModel.from_pretrained("fuchenru/Trading-Hero-LLM",
                                    output_hidden_states=True).eval()
    return tok, mdl

@st.cache_resource(show_spinner=False)
def load_baconnier():
    return SentenceTransformer("baconnier/Finance2_embedding_small_en-V1.5")

# ─── EMBEDDING ENGINE SELECTION ────────────────────────────────────────────
st.sidebar.header("1️⃣ Embedding Engine")
embed_choice = st.sidebar.radio(
    "Choose embeddings:",
    ["🧠 Dense & Deep (Trading Hero)", "🕒 Light & Fast (Baconnier)"],
    index=0
)

if embed_choice.startswith("🧠"):
    tokenizer, finbert = load_trading_hero()
    EMBED_DIM = finbert.config.hidden_size
    st.sidebar.caption("~700 MB, deep semantics")
    def embed_texts(texts: list[str]) -> np.ndarray:
        out, bs = [], 32
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            toks  = tokenizer(chunk, padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
            with torch.no_grad():
                res = finbert(**toks, output_hidden_states=True)
            last = res.hidden_states[-1]
            mask = toks.attention_mask.unsqueeze(-1)
            summed = (last*mask).sum(1)
            cnts = mask.sum(1).clamp(min=1)
            out.append((summed/cnts).cpu().numpy())
        return normalize(np.vstack(out), axis=1)
else:
    bc = load_baconnier()
    EMBED_DIM = bc.get_sentence_embedding_dimension()
    st.sidebar.caption("~30 MB, CPU-friendly")
    def embed_texts(texts: list[str]) -> np.ndarray:
        return normalize(bc.encode(texts, convert_to_numpy=True), axis=1)

# ─── GLOBAL CONFIG ─────────────────────────────────────────────────────────
REGION_GROUP = {"Northern America":"NA","Latin America":"LATAM",
                "Asia":"APAC","Oceania":"APAC",
                "Northwest Europe":"EMEA","Southern Europe":"EMEA","AroundAfrica":"EMEA"}
RATING_SCORE = {"AAA":5,"AA+":4.5,"AA":4,"AA-":3.5,
                "A+":3,"A":2.5,"A-":2,"BBB+":1.5,"BBB":1,"BB":0.5,"B":0}
from config import ASSET_CONFIG

# ─── HELPERS ─────────────────────────────────────────────────────────────
def tenor_to_months(s: str) -> float:
    t = str(s or "").strip().upper()
    if t == "ON": return 1/30
    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", t)
    if not m: return 0.0
    v, u = float(m.group(1)), m.group(2)
    return {"D":1/30,"W":7/30,"M":1,"Y":12}[u] * v

def hybrid_score(px, cd, cfg):
    tot, ws = 0.0, 0.0
    for f, w in cfg.get("weights", {}).items():
        P, C = px.get(f, ""), cd.get(f, "")
        if f == "rating":
            sc = 1 - abs(RATING_SCORE.get(P, 0) - RATING_SCORE.get(C, 0)) / 5
        elif f in ("tenor", "maturity"):
            a, b = tenor_to_months(P), tenor_to_months(C)
            sc = 1 - abs(a - b) / max(a, b, 1)
        elif f == "region":
            sc = 1 if P and C and (P == C or REGION_GROUP.get(P) == REGION_GROUP.get(C)) else 0
        elif f == "shock":
            sc = 1 if P and P == C else 0
        else:
            sc = 1 if P == C else fuzz.partial_ratio(str(P), str(C)) / 100
        tot += w * sc; ws += w
    return (tot / ws) if ws else 0.0

@st.cache_data
def build_fullcode_ann(df: pd.DataFrame):
    idx = faiss.IndexFlatIP(EMBED_DIM)
    if df.empty:
        return idx, np.zeros((0, EMBED_DIM), dtype="float32")
    embs = embed_texts(df["original"].fillna("").tolist()).astype("float32")
    idx.add(embs)
    return idx, embs

# Enhanced parse_row: handles missing and unknown assets
def parse_row(code) -> dict:
    if not isinstance(code, str):
        return {"asset": "","original": code}
    parts = [p.strip() for p in code.split(":")]
    if parts and parts[0].upper() == "CR":
        parts = parts[1:]
    a0 = parts[0].upper() if parts else ""
    if a0 in ("CDS", "CR"): asset = "CDS"
    elif a0 == "BOND": asset = "BOND"
    elif a0 == "FXSPOT": asset = "FXSPOT"
    elif a0 == "FXVOL": asset = "FXVOL"
    elif a0 == "IR" and len(parts) > 1 and parts[1].upper() == "SWAP": asset = "IRSWAP"
    elif a0 == "IR" and len(parts) > 1 and parts[1].upper() in ("SWVOL", "SWAPVOL"): asset = "IRSWVOL"
    elif a0 == "IR": asset = "IR_LINEAR"
    else: asset = a0
    out = {"asset": asset, "original": code}
    fields = ASSET_CONFIG.get(asset, {}).get("fields", [])
    for i, fld in enumerate(fields, start=1):
        out[fld] = parts[i] if i < len(parts) else ""
    return out

# ─── PAGE STYLING & LOGO ─────────────────────────────────────────────────
ING = "#FF6600"
st.markdown(f"""
<style>
.stButton>button {{ background-color:{ING}; color:white; }}
input[type="range"] {{ accent-color:{ING}; }}
</style>
""", unsafe_allow_html=True)
logo = next((p for p in [
    pathlib.Path(__file__).parent / "Inglogo.jpg",
    pathlib.Path.cwd() / "Inglogo.jpg"
] if p.exists()), None)
if logo:
    st.image(str(logo), width=100)

st.title("🔍 Scenario Generator v27")

# ─── 1️⃣ Scenario Narrative ────────────────────────────────────────────
st.header("1️⃣ Scenario Narrative")
if "sc" not in st.session_state:
    st.session_state.sc = {}
sc = st.session_state.sc
sc["name"]     = st.text_input("Name", sc.get("name", ""))
sc["type"]     = st.selectbox("Type",
    ["Historical","Hypothetical","Ad-hoc"],
    index=["Historical","Hypothetical","Ad-hoc"].index(sc.get("type","Historical"))
)
sc["severity"] = st.selectbox("Severity",
    ["Low","Medium","High","Extreme"],
    index=["Low","Medium","High","Extreme"].index(sc.get("severity","Medium"))
)
sc["assets"]   = st.multiselect("Asset Classes",
    list(ASSET_CONFIG.keys()),
    default=sc.get("assets", list(ASSET_CONFIG.keys()))
)

nar_eng = st.selectbox("Narrative Engine",
    ["t5-small","t5-base","gpt-4","gemini-2.5"], index=0
)
n0 = sc.get("narrative", "")
n1 = st.text_area("Freeform Narrative", n0, height=120)
if st.button("🔄 Generate Narrative"):
    sc["narrative"] = generate_narrative(
        sc["name"], sc["type"], sc["assets"], sc["severity"], n1, nar_eng
    )
    st.success("Narrative generated.")
st.text_area("Scenario Narrative", sc.get("narrative",""), height=200)
st.session_state.sc = sc

# ─── 2️⃣ Upload & Parse ────────────────────────────────────────────────
st.header("2️⃣ Upload Proxy & Universe")
pf = st.file_uploader("Proxy CSV", type="csv")
uf = st.file_uploader("Universe CSV", type="csv")

if pf and uf:
    # ensure 'code' always read as str and disable low memory
    px_codes = pd.read_csv(pf, header=None, names=["code"], dtype={"code": str}, low_memory=False)["code"]
    un_codes = pd.read_csv(uf, header=None, names=["code"], dtype={"code": str}, low_memory=False)["code"]

    px_df = pd.json_normalize(px_codes.apply(parse_row))
    un_df = pd.json_normalize(un_codes.apply(parse_row))
    st.success("Parsed Proxy & Universe")

    # ─── 3️⃣ Extract RFs ─────────────────────────────────────────────
    st.subheader("3️⃣ Extract Risk Factors")
    narrative = sc.get("narrative","").strip()
    pool = un_df[un_df.asset.isin(sc["assets"])].reset_index(drop=True)

    if not narrative:
        st.warning("Generate narrative first.")
    elif pool.empty:
        st.warning("No universe entries for selected assets.")
    else:
        idx_full, emb_full = build_fullcode_ann(pool)
        qn = embed_texts([narrative]).astype("float32")
        k  = st.slider("How many factors?", 1, min(50, len(pool)), 5)
        D, I = idx_full.search(qn, k)
        inds = [i for i in I[0] if 0 <= i < len(pool)]
        rf = pool.iloc[inds].copy()
        rf["sim"]       = D[0][:len(inds)]
        rf["shock_pct"] = {"Low":1,"Medium":5,"High":10,"Extreme":20}[sc["severity"]]

        if hasattr(st, "data_editor"):
            rf = st.data_editor(
                rf[["asset","original","sim","shock_pct"]],
                num_rows="dynamic", use_container_width=True
            )
        else:
            st.dataframe(rf[["asset","original","sim","shock_pct"]])

        st.download_button(
            "📥 Download RF shocks",
            rf.to_csv(index=False).encode("utf-8"),
            "rf_shocks.csv"
        )

        # ─── 4️⃣ Explain Factors ───────────────────────────────────
        st.subheader("4️⃣ Explain Factors")
        fx_eng = st.selectbox("Explanation Engine",
            ["t5-small","t5-base","gpt-4","gemini-2.5"], index=0
        )
        with st.spinner("Explaining…"):
            explanations = explain_factors(
                narrative, rf.to_dict("records"), fx_eng
            )
        rf["explanation"] = explanations
        st.dataframe(rf[["asset","original","shock_pct","explanation"]], height=300)
        st.download_button(
            "📥 Download Explanations",
            rf.to_csv(index=False).encode("utf-8"),
            "rf_explanations.csv"
        )

        # ─── 5️⃣ Propagate to Proxies ───────────────────────────
        st.subheader("5️⃣ Map to Proxies")
        α = st.sidebar.slider("Blend α", 0.0, 1.0, 0.6, 0.05)
        tk = st.sidebar.slider("ANN top_k", 1, 50, 10, 1)

        def two_stage(px, univ, cfg):
            # early exit for empty universe
            if univ.empty:
                return "", 0.0
            idx, fe = build_fullcode_ann(univ)
            q = embed_texts([px.get("original", "")]).astype("float32")
            D, I = idx.search(q, tk)
            valid = [i for i in I[0] if 0 <= i < len(univ)]
            if not valid:
                return "", 0.0
            cand = univ.iloc[valid].reset_index(drop=True)
            h = cand.apply(lambda r: hybrid_score(px, r, cfg), axis=1).to_numpy()
            s = (fe[I[0]] @ q[0]).flatten() if fe.shape[0] > 0 else np.zeros_like(h)
            comb = α * h + (1 - α) * s
            if comb.size == 0:
                return "", 0.0
            best = int(np.nanargmax(comb))
            return cand.loc[best, "original"], float(comb[best])

        out = []
        for _, r in rf.iterrows():
            subset = px_df[px_df.asset == r.asset].reset_index(drop=True)
            best, scv = two_stage(r, subset, ASSET_CONFIG.get(r.asset, {}))
            out.append({
                "asset":      r.asset,
                "factor":     r.original,
                "proxy":      best,
                "score":      round(scv, 4),
                "shock_pct":  r.shock_pct
            })
        dfp = pd.DataFrame(out)
        st.dataframe(dfp, height=300)
        st.download_button(
            "📥 Download Proxy shocks",
            dfp.to_csv(index=False).encode("utf-8"),
            "proxy_shocks.csv"
        )
