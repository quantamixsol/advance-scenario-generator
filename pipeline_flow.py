import argparse
import pandas as pd
from portfolio import PortfolioIngestor, RiskFactorMapper
from data_io import parse_df
from exposures import apply_shocks, asset_pnl_breakdown
from config import SHOCK_UNITS, BASELINE_SHOCKS


def _baseline_shocks(rf_df: pd.DataFrame, severity: str) -> pd.DataFrame:
    """Assign baseline shocks based on asset class and severity."""
    values = []
    for _, row in rf_df.iterrows():
        unit = SHOCK_UNITS.get(row["asset"], "pct")
        values.append(BASELINE_SHOCKS[unit][severity])
    rf_df = rf_df.copy()
    rf_df["shock_pct"] = values
    return rf_df


def run_pipeline(portfolio: str, universe: str, severity: str = "Medium") -> pd.DataFrame:
    """Full portfolio → RF → scenario PnL flow."""
    ing = PortfolioIngestor(portfolio)
    pf = ing.get()

    raw_univ = pd.read_csv(universe, header=None, names=["code"])
    univ = parse_df(raw_univ)

    mapper = RiskFactorMapper(univ)
    mapped = mapper.map(pf)

    rf_codes = mapped["rf_code"].dropna().unique()
    rf_df = univ[univ["original"].isin(rf_codes)].copy()
    rf_df = _baseline_shocks(rf_df, severity)

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
    args = p.parse_args()

    pnl, breakdown = run_pipeline(args.portfolio, args.universe, args.severity)
    pnl.to_csv(args.output, index=False)
    breakdown.to_csv("asset_pnl.csv", index=False)
    print(f"Saved scenario PnL to {args.output} ({len(pnl)} rows)")
    print("Asset-class breakdown saved to asset_pnl.csv")


if __name__ == "__main__":
    main()
