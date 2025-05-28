import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

from portfolio import PortfolioIngestor, RiskFactorMapper
from data_io import parse_df
from embeddings import get_embedder
from matching import extract_risk_factors, build_fullcode_ann
from generative import generate_narrative, refine_shocks_with_llm
from exposures import apply_shocks, asset_pnl_breakdown

st.set_page_config(page_title="Scenario Wizard", layout="wide")

# ------------------------------------------------------------------
# Sidebar controls
st.sidebar.header("⚙️ Settings")

embed_model = st.sidebar.selectbox(
    "Embedding Engine",
    ["baconnier", "trading_hero"],
    index=0,
)
embed_texts, EMBED_DIM, embed_desc = get_embedder(embed_model)
st.sidebar.caption(embed_desc)

narrative_llm = st.sidebar.selectbox(
    "Narrative Engine",
    ["t5-small", "t5-base", "gpt-4", "gemini-2.5"],
    index=0,
)

shock_llm = st.sidebar.selectbox(
    "Shock Engine",
    ["t5-small", "gpt-4", "gemini-2.5"],
    index=0,
)

alpha = st.sidebar.slider("α Blend", 0.0, 1.0, 0.6, 0.05)
rf_top_k = st.sidebar.slider("top-k Factors", 1, 50, 5, 1)
severity = st.sidebar.selectbox(
    "Scenario Severity", ["Low", "Medium", "High", "Extreme"], index=1
)

# ------------------------------------------------------------------
# Stage flags
if "pf_loaded" not in st.session_state:
    st.session_state.pf_loaded = False
if "mapped" not in st.session_state:
    st.session_state.mapped = False
if "narrative_done" not in st.session_state:
    st.session_state.narrative_done = False
if "shocks_done" not in st.session_state:
    st.session_state.shocks_done = False
if "pnl_done" not in st.session_state:
    st.session_state.pnl_done = False

# ------------------------------------------------------------------
# 1) Upload or type portfolio
st.header("1️⃣ Upload or Type Portfolio")
port_file = st.file_uploader("Portfolio file")
port_text = st.text_area("Or paste CSV", "")
if st.button("Load Portfolio"):
    try:
        if port_file is not None:
            ing = PortfolioIngestor(port_file)
        elif port_text.strip():
            ing = PortfolioIngestor(StringIO(port_text))
        else:
            st.warning("Provide a file or CSV text.")
            ing = None
        if ing:
            st.session_state.portfolio = ing.get()
            st.session_state.pf_loaded = True
            st.success(f"Loaded {len(st.session_state.portfolio)} rows")
    except Exception as e:
        st.error(str(e))
        st.session_state.pf_loaded = False

if not st.session_state.pf_loaded:
    st.stop()

pf = st.session_state.portfolio
st.dataframe(pf, use_container_width=True)

# ------------------------------------------------------------------
# 2) Map to risk factors
st.header("2️⃣ Map to Risk Factors")
univ_file = st.file_uploader("Universe CSV", key="univ")
if st.button("Map Portfolio"):
    if univ_file is None:
        st.warning("Upload a universe file first")
    else:
        raw = pd.read_csv(univ_file, header=None, names=["code"])
        univ = parse_df(raw)
        mapper = RiskFactorMapper(univ)
        mapped = mapper.map(pf)
        st.session_state.universe = univ
        st.session_state.mapped_pf = mapped
        st.session_state.mapped = True
        st.success("Portfolio mapped")

if not st.session_state.mapped:
    st.stop()

mapped = st.session_state.mapped_pf
if hasattr(st, "data_editor"):
    mapped = st.data_editor(mapped, num_rows="dynamic", use_container_width=True)
else:
    st.dataframe(mapped)

st.session_state.mapped_pf = mapped

# Prepare RF dataframe for later
univ = st.session_state.universe
rf_codes = mapped["rf_code"].dropna().unique()
rf_df = univ[univ["original"].isin(rf_codes)].reset_index(drop=True)

# ------------------------------------------------------------------
# 3) Generate scenario narrative
st.header("3️⃣ Scenario Narrative")
sc_name = st.text_input("Scenario Name")
user_prompt = st.text_area("Freeform Narrative")
if st.button("Generate Narrative"):
    nar = generate_narrative(sc_name, "Hypothetical", [], severity, user_prompt, narrative_llm)
    st.session_state.narrative = nar
    st.session_state.narrative_done = True
    st.success("Narrative generated")

if not st.session_state.narrative_done:
    st.stop()

st.markdown(st.session_state.narrative)

# ------------------------------------------------------------------
# 4) Extract & simulate shocks
st.header("4️⃣ Extract & Simulate Shocks")
if st.button("Run Shock Simulation"):
    narrative = st.session_state.narrative
    rf = extract_risk_factors(narrative, rf_df, embed_texts, rf_top_k, severity)
    factors = rf.to_dict("records")
    shocks = refine_shocks_with_llm(narrative, factors, severity, shock_llm)
    rf["shock_pct"] = shocks
    st.session_state.rf = rf
    st.session_state.shocks_done = True
    st.success("Shocks generated")

if not st.session_state.shocks_done:
    st.stop()

rf = st.session_state.rf
if hasattr(st, "data_editor"):
    rf = st.data_editor(rf, num_rows="dynamic", use_container_width=True)
else:
    st.dataframe(rf)

st.session_state.rf = rf
st.download_button("Download RF shocks", rf.to_csv(index=False), "rf_shocks.csv")

# ------------------------------------------------------------------
# 5) Compute P&L impact
st.header("5️⃣ Compute P&L Impact")
if st.button("Apply Shocks"):
    pnl_df = apply_shocks(mapped, rf)
    breakdown = asset_pnl_breakdown(pnl_df)
    st.session_state.pnl_df = pnl_df
    st.session_state.breakdown = breakdown
    st.session_state.pnl_done = True
    st.success("PnL computed")

if not st.session_state.pnl_done:
    st.stop()

pnl_df = st.session_state.pnl_df
breakdown = st.session_state.breakdown
st.dataframe(pnl_df, use_container_width=True)
fig = px.bar(breakdown, x="asset", y="asset_pnl")
st.plotly_chart(fig, use_container_width=True)
st.download_button("Download PnL", pnl_df.to_csv(index=False), "pnl.csv")

# ------------------------------------------------------------------
# 6) Backtest (future)
st.header("6️⃣ Backtest (future)")
st.info("Backtesting functionality not yet implemented.")

