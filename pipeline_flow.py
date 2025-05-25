import argparse
import pandas as pd
from portfolio import PortfolioIngestor, RiskFactorMapper
from data_io import parse_df
from exposures import (
    apply_shocks,
    asset_pnl_breakdown,
    validate_parallel_shocks,
    total_portfolio_pnl,
)
from config import SHOCK_UNITS, BASELINE_SHOCKS
from embeddings import get_embedder
from matching import proxy_match, build_fullcode_ann
import faiss
import json
from generative import generate_narrative, refine_shocks_with_llm


def _baseline_shocks(rf_df: pd.DataFrame, severity: str, overrides: dict | None = None) -> pd.DataFrame:
    """Assign baseline shocks based on asset class and severity with optional overrides."""
    values = []
    for _, row in rf_df.iterrows():
        asset = row["asset"]
        unit = SHOCK_UNITS.get(asset, "pct")
        shock = BASELINE_SHOCKS[unit][severity]
        if overrides and asset in overrides:
            shock = overrides[asset]
        values.append(shock)
    rf_df = rf_df.copy()
    rf_df["shock_pct"] = values
    return rf_df


def portfolio_assets(mapped: pd.DataFrame, univ: pd.DataFrame) -> list[str]:
    """Return list of asset classes present in the mapped portfolio."""
    return (
        mapped.merge(
            univ[["original", "asset"]],
            left_on="rf_code",
            right_on="original",
            how="left",
        )["asset"].dropna().unique().tolist()
    )


def map_with_proxies(pf: pd.DataFrame, univ: pd.DataFrame, px_df: pd.DataFrame, embed_texts) -> pd.DataFrame:
    """Map portfolio rows to risk factors using universe then fall back to proxies."""
    mapper = RiskFactorMapper(univ)
    mapped = mapper.map(pf)
    missing = mapped["rf_code"].isna()
    if missing.any() and not px_df.empty:
        idx, embs = build_fullcode_ann(px_df, embed_texts)
        q = embed_texts(mapped.loc[missing, "ticker_norm"].astype(str).tolist()).astype("float32")
        D, I = idx.search(q, 1)
        matches = [px_df.iloc[i]["original"] if i < len(px_df) else None for i in I[:, 0]]
        mapped.loc[missing, "rf_code"] = matches
    return mapped


def run_pipeline(portfolio: str, universe: str, severity: str = "Medium", baseline_config: str | None = None) -> pd.DataFrame:
    """Full portfolio → RF → scenario PnL flow."""
    print("Loading portfolio...")
    ing = PortfolioIngestor(portfolio)
    pf = ing.get()

    print("Parsing universe...")
    raw_univ = pd.read_csv(universe, header=None, names=["code"])
    univ = parse_df(raw_univ)

    print("Mapping portfolio tickers to risk factors...")
    mapper = RiskFactorMapper(univ)
    mapped = mapper.map(pf)

    # determine which asset classes actually appear in the portfolio
    pf_assets = portfolio_assets(mapped, univ)
    # keep only risk factors for those assets
    univ_sub = univ[univ["asset"].isin(pf_assets)].reset_index(drop=True)

    rf_codes = mapped["rf_code"].dropna().unique()
    rf_df = univ_sub[univ_sub["original"].isin(rf_codes)].copy()
    overrides = None
    if baseline_config:
        with open(baseline_config) as fh:
            overrides = json.load(fh)
    rf_df = _baseline_shocks(rf_df, severity, overrides)
    validate_parallel_shocks(rf_df)

    print("Applying shocks...")
    pnl_df = apply_shocks(mapped, rf_df)
    pnl_breakdown = asset_pnl_breakdown(pnl_df)
    print(f"Total portfolio PnL: {total_portfolio_pnl(pnl_df):,.2f}")
    if not pnl_breakdown.empty:
        print("PnL by asset class:")
        print(pnl_breakdown.to_string(index=False))
    return pnl_df, pnl_breakdown


def run_scenario_pipeline(
    portfolio: str,
    universe: str,
    proxies: str,
    narrative: str,
    severity: str = "Medium",
    embedder: str = "baconnier",
    rf_k: int = 20,
    alpha: float = 0.6,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end flow ensuring portfolio RFs are shocked."""
    print("Loading portfolio...")
    ing = PortfolioIngestor(portfolio)
    pf = ing.get()

    print("Parsing universe and proxies…")
    raw_univ = pd.read_csv(universe, header=None, names=["code"])
    univ = parse_df(raw_univ)
    raw_px = pd.read_csv(proxies, header=None, names=["code"])
    px_df = parse_df(raw_px)

    print("Embedding texts for mapping…")
    embed_texts, dim, _ = get_embedder(embedder)
    mapped = map_with_proxies(pf, univ, px_df, embed_texts)

    pf_assets = portfolio_assets(mapped, pd.concat([univ, px_df], ignore_index=True))
    combined = pd.concat([univ, px_df], ignore_index=True)
    rf_codes = mapped["rf_code"].dropna().unique()
    rf_df = combined[combined["original"].isin(rf_codes)].drop_duplicates("original").copy()
    rf_df = _baseline_shocks(rf_df, severity)
    validate_parallel_shocks(rf_df)
    shocks = rf_df[["original", "asset", "shock_pct"]]

    print("Applying shocks…")
    pnl_df = apply_shocks(mapped, shocks)
    pnl_breakdown = asset_pnl_breakdown(pnl_df)
    print(f"Total portfolio PnL: {total_portfolio_pnl(pnl_df):,.2f}")
    if not pnl_breakdown.empty:
        print("PnL by asset class:")
        print(pnl_breakdown.to_string(index=False))

    return pnl_df, pnl_breakdown, rf_df, shocks


def run_generated_scenario(
    portfolio: str,
    universe: str,
    proxies: str,
    name: str,
    scenario_type: str,
    user_input: str,
    severity: str = "Medium",
    embedder: str = "baconnier",
    llm_engine: str = "t5-small",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Complete portfolio → PnL pipeline with LLM-generated shocks."""
    # 1) load portfolio
    ing = PortfolioIngestor(portfolio)
    pf = ing.get()

    # 2) parse risk factor universe and proxy set
    raw_univ = pd.read_csv(universe, header=None, names=["code"])
    univ = parse_df(raw_univ)
    raw_px = pd.read_csv(proxies, header=None, names=["code"])
    px_df = parse_df(raw_px)

    # 3) map tickers to risk factors, falling back to proxies
    embed_texts, _, _ = get_embedder(embedder)
    mapped = map_with_proxies(pf, univ, px_df, embed_texts)

    combined = pd.concat([univ, px_df], ignore_index=True)
    pf_assets = portfolio_assets(mapped, combined)
    rf_codes = mapped["rf_code"].dropna().unique()
    rf_df = combined[combined["original"].isin(rf_codes)].drop_duplicates("original").copy()

    # 4) generate scenario narrative then derive shocks via LLM
    narrative = generate_narrative(name, scenario_type, pf_assets, severity, user_input, engine=llm_engine)
    factors = rf_df.to_dict("records")
    shocks = refine_shocks_with_llm(narrative, factors, severity, engine=llm_engine)
    rf_df["shock_pct"] = shocks

    validate_parallel_shocks(rf_df)

    # 5) apply shocks and compute PnL
    pnl_df = apply_shocks(mapped, rf_df)
    pnl_breakdown = asset_pnl_breakdown(pnl_df)
    print(f"Total portfolio PnL: {total_portfolio_pnl(pnl_df):,.2f}")
    if not pnl_breakdown.empty:
        print("PnL by asset class:")
        print(pnl_breakdown.to_string(index=False))

    return pnl_df, pnl_breakdown, rf_df, narrative


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end scenario pipeline")
    p.add_argument("--portfolio", required=True, help="Path to portfolio file")
    p.add_argument("--universe", required=True, help="Path to universe CSV")
    p.add_argument("--proxies", help="Path to proxies CSV")
    p.add_argument("--narrative", help="Scenario narrative")
    p.add_argument("--scenario_name", help="Name of scenario for LLM generation")
    p.add_argument("--scenario_type", default="adverse", help="Type of scenario")
    p.add_argument("--user_input", default="", help="Additional user input")
    p.add_argument("--llm_engine", default="t5-small", help="LLM engine")
    p.add_argument(
        "--severity", choices=["Low", "Medium", "High", "Extreme"], default="Medium"
    )
    p.add_argument("--output", default="scenario_pnl.csv", help="Output CSV path")
    p.add_argument("--baseline_config", help="JSON file of custom baseline shocks", default=None)
    p.add_argument("--embedder", choices=["baconnier", "trading_hero"], default="baconnier")
    p.add_argument("--rf_k", type=int, default=20, help="Number of RFs to extract")
    p.add_argument("--alpha", type=float, default=0.6, help="Hybrid weight")
    p.add_argument("--top_k", type=int, default=10, help="ANN candidate size")
    args = p.parse_args()

    if args.scenario_name and args.proxies:
        pnl, breakdown, rf_df, _ = run_generated_scenario(
            args.portfolio,
            args.universe,
            args.proxies,
            args.scenario_name,
            args.scenario_type,
            args.user_input,
            args.severity,
            args.embedder,
            args.llm_engine,
        )
    elif args.narrative and args.proxies:
        pnl, breakdown, rf_df, px_df = run_scenario_pipeline(
            args.portfolio,
            args.universe,
            args.proxies,
            args.narrative,
            args.severity,
            args.embedder,
            args.rf_k,
            args.alpha,
            args.top_k,
        )
    else:
        pnl, breakdown = run_pipeline(
            args.portfolio,
            args.universe,
            args.severity,
            args.baseline_config,
        )

    pnl.to_csv(args.output, index=False)
    breakdown.to_csv("asset_pnl.csv", index=False)
    print(f"Saved scenario PnL to {args.output} ({len(pnl)} rows)")
    print("Asset-class breakdown saved to asset_pnl.csv")


if __name__ == "__main__":
    main()
