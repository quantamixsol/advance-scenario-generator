# app.py
# ──────────────────────────────────────────────────────────────────────────────
# pip install streamlit transformers sentence-transformers torch faiss-cpu rapidfuzz scikit-learn plotly openai google-generativeai

import sys, asyncio, os, pathlib, re
from pathlib import Path
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# silence torch._classes probing
try:
    if hasattr(torch, "_classes") and hasattr(torch._classes, "__path__"):
        del torch._classes.__path__
except:
    pass

import streamlit as st
st.set_page_config(page_title="🔍 Scenario Generator v28", layout="wide")

import pandas as pd
import numpy as np
import faiss
from rapidfuzz import fuzz
from sklearn.preprocessing import normalize
import plotly.express as pex
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from generative import generate_narrative, explain_factors
from generative import refine_shocks_with_llm
from portfolio import PortfolioIngestor, RiskFactorMapper
from exposures import apply_shocks, asset_pnl_breakdown, validate_parallel_shocks
from data_io import parse_row
from matching import hybrid_score, build_fullcode_ann as _build_fullcode_ann

# ─── pull all your global lookups & defaults from config.py ──────────────
from config import (
    REGION_GROUP,
    RATING_SCORE,
    ASSET_CONFIG,
    default_weights,
    SHOCK_UNITS,
    BASELINE_SHOCKS
)

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

# # ─── GLOBAL CONFIG ─────────────────────────────────────────────────────────
# REGION_GROUP = {"Northern America":"NA","Latin America":"LATAM",
#                 "Asia":"APAC","Oceania":"APAC",
#                 "Northwest Europe":"EMEA","Southern Europe":"EMEA","AroundAfrica":"EMEA"}
# RATING_SCORE = {"AAA":5,"AA+":4.5,"AA":4,"AA-":3.5,
#                 "A+":3,"A":2.5,"A-":2,"BBB+":1.5,"BBB":1,"BB":0.5,"B":0}
# ASSET_CONFIG = {
#     "BOND":      {"fields":["rating","tenor","seniority","sector","region","currency","covered","shock"]},
#     "CDS":       {"fields":["instrument","issuer","currency","seniority","liquidity","maturity","shock"]},
#     "CR":        {"fields":["instrument","issuer","currency","seniority","liquidity","maturity","shock"]},
#     "FXSPOT":    {"fields":["spot","pair","shock"]},
#     "FXVOL":     {"fields":["smile","type","pair","strike","option_type","tenor","shock"]},
#     "IRSWAP":    {"fields":["instrument","curve_name","currency","tenor","shock"]},
#     "IRSWVOL":   {"fields":["method","currency","tenor","shock"]},
#     "IR_LINEAR": {"fields":["instrument","curve_name","rate","currency","tenor","shock"]},
# }
# # uniform default weights
# for cfg in ASSET_CONFIG.values():
#     cfg["weights"] = {fld:1.0 for fld in cfg["fields"]}

# ─── inject default weights into each ASSET_CONFIG entry ─────────────────
for asset, weights in default_weights().items():
    ASSET_CONFIG[asset]["weights"] = weights

# cache FAISS ANN indices per DataFrame
@st.cache_data
def build_fullcode_ann(df: pd.DataFrame):
    return _build_fullcode_ann(df, embed_texts)

# ─── PAGE STYLING ─────────────────────────────────────────────────────────
ING = "#FF6600"
st.markdown(f"""
<style>
/* ING brand styling */
.stButton>button {{ background-color:{ING}; color:white; }}
input[type="range"] {{ accent-color:{ING}; }}
/* auto-resize our two textareas */
textarea[aria-label="Freeform Narrative"],
textarea[aria-label="Scenario Narrative"] {{
    min-height: 120px !important;
    height: auto !important;
    overflow-y: hidden !important;
}}
</style>
""", unsafe_allow_html=True)

# logo top-left
logo = next((p for p in [
    pathlib.Path(__file__).parent/"Inglogo.jpg",
    pathlib.Path.cwd()/"Inglogo.jpg"
] if p.exists()), None)
if logo:
    st.image(str(logo), width=100)

st.title("🔍 Scenario Generator v28")

# ─── 0️⃣ Upload & Map Portfolio ────────────────────────────────────────
st.header("0️⃣ Upload & Map Portfolio")
col1, col2 = st.columns(2)
with col1:
    port_file = st.file_uploader("Portfolio (CSV/Excel/JSON)", key="port_up")
with col2:
    uni_file_port = st.file_uploader("Universe CSV", type="csv", key="uni_port")
if st.button("▶️ Map Portfolio → RF"):
    if not port_file or not uni_file_port:
        st.warning("Provide both portfolio and universe files.")
    else:
        ing = PortfolioIngestor(port_file)
        pf = ing.get()
        rawu = pd.read_csv(uni_file_port, header=None, names=["code"])
        un_df_port = pd.json_normalize(rawu["code"].apply(parse_row))
        mapper = RiskFactorMapper(un_df_port)
        mapped = mapper.map(pf)
        st.session_state.mapped_pf = mapped
        st.success(f"Mapped {mapped.rf_code.notna().sum()} of {len(mapped)} positions")

if "mapped_pf" in st.session_state:
    st.subheader("Mapped Portfolio Preview")
    st.dataframe(st.session_state.mapped_pf, use_container_width=True)
else:
    st.info("Upload and map a portfolio to continue.")
    st.stop()

# ─── 1️⃣ Scenario Narrative ────────────────────────────────────────────
st.header("1️⃣ Scenario Narrative")
if "sc" not in st.session_state:
    st.session_state.sc = {}
sc = st.session_state.sc

# metadata
sc["name"]     = st.text_input("Name", sc.get("name",""))
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

# choose LLM engine
nar_eng = st.selectbox("Narrative Engine",
    ["t5-small","t5-base","gpt-4","gemini-2.5"], index=0
)

# ─── Freeform Narrative (user prompt) ──────────────────────────────────
n0 = sc.get("narrative","")
n1 = st.text_area("Freeform Narrative", n0, key="free_txt")
if st.button("🔄 Generate Narrative"):
    sc["narrative"] = generate_narrative(
        sc["name"], sc["type"], sc["assets"],
        sc["severity"], n1, nar_eng
    )
    st.success("Narrative generated.")

# ─── Scenario Narrative (rendered markdown) ───────────────────────────
st.markdown("**Scenario Narrative:**")
if sc.get("narrative"):
    st.markdown(sc["narrative"])   # renders lists, line-breaks, etc.
else:
    st.info("Generated narrative will appear here.")

st.session_state.sc = sc

# ─── 2️⃣ Upload & Load Universe ───────────────────────────────────────
st.header("2️⃣ Upload & Load Universe")
uni_file = st.file_uploader("Universe CSV", type="csv", key="uni_upload")
if st.button("▶️ Load Universe"):
    if uni_file is None:
        st.warning("Please select a Universe CSV first.")
    else:
        raw = pd.read_csv(uni_file, header=None, names=["code"])
        st.session_state["un_df"] = pd.json_normalize(raw["code"].apply(parse_row))
        st.success("✔ Universe loaded and parsed.")

# only proceed once universe is loaded
if "un_df" in st.session_state:
    un_df = st.session_state["un_df"]

    # ─── 3️⃣ Extract & Simulate RFs ────────────────────────────────────
    st.subheader("3️⃣ Extract & Simulate Risk Factors")
    narrative = sc.get("narrative", "").strip()
    pool = un_df[un_df.asset.isin(sc["assets"])].reset_index(drop=True)

    if not narrative:
        st.warning("Generate a scenario narrative first.")
    elif pool.empty:
        st.warning("Selected assets not in Universe.")
    else:
        # FAISS over selected universe
        idx_full, emb_full = build_fullcode_ann(pool)
        qn = embed_texts([narrative]).astype("float32")
        k  = st.slider("How many factors?", 1, min(50, len(pool)), 5)
        D, I = idx_full.search(qn, k)
        inds = [i for i in I[0] if i < len(pool)]
        rf = pool.iloc[inds].copy()
        rf["sim"] = D[0][: len(inds)]

        # choose shock engine
        shock_engine = st.sidebar.selectbox(
            "🔧 Shock Engine",
            ["t5-small", "gpt-4", "gemini-2.5"],
            index=1
        )

        # run LLM-based shock simulation
        rf["shock_pct"] = refine_shocks_with_llm(
            narrative,
            rf.to_dict("records"),
            sc["severity"],
            engine=shock_engine
        )
        # enforce curve-wise parallel shifts
        try:
            validate_parallel_shocks(rf)
        except ValueError as e:
            st.warning(str(e))

        # allow inline edits
        if hasattr(st, "data_editor"):
            rf = st.data_editor(
                rf[["asset", "original", "sim", "shock_pct"]],
                num_rows="dynamic", use_container_width=True
            )
        else:
            st.dataframe(rf[["asset", "original", "sim", "shock_pct"]])

        st.download_button(
            "📥 Download RF shocks",
            rf.to_csv(index=False).encode("utf-8"),
            "rf_shocks.csv"
        )

        # save for next step
        st.session_state["rf"] = rf

        # ─── 4️⃣ Explain Factors ───────────────────────────────────
        st.subheader("4️⃣ Explain Factors")
        exp_engine = st.sidebar.selectbox(
            "Explanation Engine",
            ["t5-small","t5-base","gpt-4","gemini-2.5"],
            index=0,
            key="exp_engine"
        )
        with st.spinner("Generating explanations…"):
            explanations = explain_factors(
                narrative,
                rf.to_dict("records"),
                exp_engine
            )
        rf["explanation"] = explanations
        st.dataframe(
            rf[["asset","original","shock_pct","explanation"]],
            height=300,
            use_container_width=True
        )
        st.download_button(
            "📥 Download Explanations",
            rf.to_csv(index=False).encode("utf-8"),
            "rf_explanations.csv"
        )

        # now save to session for next step
        st.session_state["rf"] = rf

        if "mapped_pf" in st.session_state:
            pnl_df = apply_shocks(st.session_state["mapped_pf"], rf)
            breakdown = asset_pnl_breakdown(pnl_df)
            st.session_state["exposures"] = pnl_df
            st.subheader("📈 Scenario PnL")
            st.metric("Total PnL", f"${pnl_df['pnl'].sum():,.2f}")
            st.dataframe(
                pnl_df[["ticker", "quantity", "price", "rf_code", "shock_pct", "pnl"]],
                use_container_width=True,
            )
            st.bar_chart(breakdown.set_index("asset"))
            st.download_button(
                "📥 Download PnL",
                pnl_df.to_csv(index=False).encode("utf-8"),
                "scenario_pnl.csv",
            )

    # ─── 5️⃣ Upload & Load Proxy ──────────────────────────────────────
    st.header("5️⃣ Upload & Load Proxy")
    prox_file = st.file_uploader("Proxy CSV", type="csv", key="prox_upload")
    if st.button("▶️ Load Proxy"):
        if prox_file is None:
            st.warning("Please select a Proxy CSV first.")
        else:
            raw_px = pd.read_csv(prox_file, header=None, names=["code"])
            st.session_state["px_df"] = pd.json_normalize(raw_px["code"].apply(parse_row))
            st.success("✔ Proxy loaded and parsed.")

    # only proceed to mapping once both RFs & proxies are ready
    if "rf" in st.session_state and "px_df" in st.session_state:
        rf    = st.session_state["rf"]
        px_df = st.session_state["px_df"]

        # ─── 6️⃣ Map Shocks to Proxies ─────────────────────────────────
        st.subheader("6️⃣ Map Shocks to Proxies")
        α  = st.sidebar.slider("Blend α",   0.0, 1.0, 0.6, 0.05)
        tk = st.sidebar.slider("ANN top_k", 1, 50, 10, 1)

        def two_stage(px, univ, cfg):
            idx, fe = build_fullcode_ann(univ)
            q       = embed_texts([px["original"]]).astype("float32")
            D, I    = idx.search(q, tk)
            valid   = [i for i in I[0] if i < len(univ)]
            cand    = univ.iloc[valid].reset_index(drop=True)
            h       = cand.apply(lambda r: hybrid_score(px, r, cfg), axis=1).to_numpy()
            s       = (fe[I[0]] @ q[0]).flatten() if fe.shape[0] > 0 else np.zeros_like(h)
            comb    = α * h + (1-α) * s
            best    = int(np.nanargmax(comb))
            return cand.loc[best, "original"], float(comb[best])

        results = []
        for _, r in rf.iterrows():
            subset = px_df[px_df.asset == r.asset].reset_index(drop=True)
            best, scv = two_stage(pd.Series(parse_row(r.original)), subset, ASSET_CONFIG[r.asset])
            results.append({
                "asset":      r.asset,
                "factor":     r.original,
                "proxy":      best,
                "score":      round(scv,4),
                "shock_pct":  r.shock_pct
            })
        dfp = pd.DataFrame(results)
        st.dataframe(dfp, height=300)
        st.download_button(
            "📥 Download Proxy shocks",
            dfp.to_csv(index=False).encode("utf-8"),
            "proxy_shocks.csv"
        )
