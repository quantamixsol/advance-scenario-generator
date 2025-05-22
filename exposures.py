import pandas as pd
from config import SHOCK_UNITS

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
        unit = SHOCK_UNITS.get(row.get("asset"), "pct")
        pct = row["shock_pct"] / 10000 if unit == "bps" else row["shock_pct"] / 100
        return -row["quantity"] * row["price"] * pct

    pf["pnl"] = pf.apply(_pnl, axis=1)
    return pf

