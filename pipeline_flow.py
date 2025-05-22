import argparse
import pandas as pd
from portfolio import PortfolioIngestor, RiskFactorMapper
from data_io import parse_df
from exposures import apply_shocks, asset_pnl_breakdown, validate_parallel_shocks
from config import SHOCK_UNITS, BASELINE_SHOCKS
from embeddings import get_embedder
from matching import proxy_match
import json


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

    rf_codes = mapped["rf_code"].dropna().unique()
    rf_df = univ[univ["original"].isin(rf_codes)].copy()
    overrides = None
    if baseline_config:
        with open(baseline_config) as fh:
            overrides = json.load(fh)
    rf_df = _baseline_shocks(rf_df, severity, overrides)
    validate_parallel_shocks(rf_df)

    print("Applying shocks...")
    pnl_df = apply_shocks(mapped, rf_df)
    pnl_breakdown = asset_pnl_breakdown(pnl_df)
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
    """End-to-end flow using scenario narrative and proxy matching."""
    print("Loading portfolio...")
    ing = PortfolioIngestor(portfolio)
    pf = ing.get()

    print("Parsing universe and proxies…")
    raw_univ = pd.read_csv(universe, header=None, names=["code"])
    univ = parse_df(raw_univ)
    raw_px = pd.read_csv(proxies, header=None, names=["code"])
    px_df = parse_df(raw_px)

    print("Mapping portfolio tickers to risk factors…")
    mapper = RiskFactorMapper(univ)
    mapped = mapper.map(pf)

    # determine portfolio asset classes
    pf_assets = (
        mapped.merge(
            univ[["original", "asset"]],
            left_on="rf_code",
            right_on="original",
            how="left",
        )["asset"].dropna().unique()
    )
    univ_sub = univ[univ["asset"].isin(pf_assets)].reset_index(drop=True)
    px_sub = px_df[px_df["asset"].isin(pf_assets)].reset_index(drop=True)

    print("Embedding texts and extracting risk factors…")
    embed_texts, dim, _ = get_embedder(embedder)
    rf_df, px_match = proxy_match(
        px_sub,
        univ_sub,
        embed_texts,
        dim,
        alpha,
        top_k,
        narrative,
        rf_k,
        severity,
    )

    # convert proxy matches to shock table for apply_shocks
    shocks = px_match.rename(columns={"proxy": "original"})[["original", "asset", "shock_pct"]]

    print("Applying shocks…")
    pnl_df = apply_shocks(mapped, shocks)
    pnl_breakdown = asset_pnl_breakdown(pnl_df)

    return pnl_df, pnl_breakdown, rf_df, px_match


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end scenario pipeline")
    p.add_argument("--portfolio", required=True, help="Path to portfolio file")
    p.add_argument("--universe", required=True, help="Path to universe CSV")
    p.add_argument("--proxies", help="Path to proxies CSV")
    p.add_argument("--narrative", help="Scenario narrative")
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

    if args.narrative and args.proxies:
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
