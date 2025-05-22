import pandas as pd
from config import SHOCK_UNITS
from pricing import get_pricing_function

def apply_shocks(mapped_portfolio: pd.DataFrame, rf_df: pd.DataFrame) -> pd.DataFrame:
    """Merge portfolio with RF shocks and compute PnL for each position."""
    pf = mapped_portfolio.merge(
        rf_df[["original", "asset", "shock_pct"]].rename(columns={"original": "rf_code"}),
        on="rf_code",
        how="left",
    )

    def _pnl(row):
        if pd.isna(row.get("shock_pct")):
            return 0.0
        fn = get_pricing_function(row.get("asset"))
        if fn:
            return fn(row)
        unit = SHOCK_UNITS.get(row.get("asset"), "pct")
        pct = row["shock_pct"] / 10000 if unit == "bps" else row["shock_pct"] / 100
        return -row["quantity"] * row.get("price", 1.0) * pct

    pf["pnl"] = pf.apply(_pnl, axis=1)
    return pf

def asset_pnl_breakdown(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Return total PnL per asset class."""
    return pnl_df.groupby("asset")["pnl"].sum().reset_index().rename(columns={"pnl": "asset_pnl"})

