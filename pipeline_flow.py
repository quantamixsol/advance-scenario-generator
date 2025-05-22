import argparse
import pandas as pd
from portfolio import PortfolioIngestor, RiskFactorMapper
from data_io import parse_df
from exposures import apply_shocks, asset_pnl_breakdown, validate_parallel_shocks
from config import SHOCK_UNITS, BASELINE_SHOCKS
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


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end scenario pipeline")
    p.add_argument("--portfolio", required=True, help="Path to portfolio file")
    p.add_argument("--universe", required=True, help="Path to universe CSV")
    p.add_argument(
        "--severity", choices=["Low", "Medium", "High", "Extreme"], default="Medium"
    )
    p.add_argument("--output", default="scenario_pnl.csv", help="Output CSV path")
    p.add_argument("--baseline_config", help="JSON file of custom baseline shocks", default=None)
    args = p.parse_args()

    pnl, breakdown = run_pipeline(args.portfolio, args.universe, args.severity, args.baseline_config)
    pnl.to_csv(args.output, index=False)
    breakdown.to_csv("asset_pnl.csv", index=False)
    print(f"Saved scenario PnL to {args.output} ({len(pnl)} rows)")
    print("Asset-class breakdown saved to asset_pnl.csv")


if __name__ == "__main__":
    main()
