import pandas as pd
from config import SHOCK_UNITS
from pricing import get_pricing_function

def apply_shocks(mapped_portfolio: pd.DataFrame, rf_df: pd.DataFrame, base_currency: str = "USD") -> pd.DataFrame:
    """Merge portfolio with RF shocks and compute base-currency PnL for each position."""
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
            pnl = fn(row)
        else:
            unit = SHOCK_UNITS.get(row.get("asset"), "pct")
            pct = row["shock_pct"] / 10000 if unit == "bps" else row["shock_pct"] / 100
            pnl = -row["quantity"] * row.get("price", 1.0) * pct
        fx = row.get("fx_rate", 1.0)
        return pnl * fx
    pf["pnl"] = pf.apply(_pnl, axis=1)
    return pf

def asset_pnl_breakdown(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Return total PnL per asset class.

    Positions without an associated risk factor will have a missing ``asset``
    value.  We still want to include these in the output so users can clearly
    see the impact of unmapped positions.
    """
    df = pnl_df.copy()
    df["asset"] = df["asset"].fillna("UNKNOWN")
    return (
        df.groupby("asset")["pnl"].sum()
        .reset_index()
        .rename(columns={"pnl": "asset_pnl"})
    )

def total_portfolio_pnl(pnl_df: pd.DataFrame) -> float:
    """Return total portfolio PnL."""
    return float(pnl_df["pnl"].sum())

def validate_parallel_shocks(rf_df: pd.DataFrame) -> None:
    """Ensure shocks are parallel across interest-rate curve names."""
    if "curve_name" not in rf_df.columns:
        return
    grouped = rf_df.groupby(["curve_name"])["shock_pct"].nunique()
    bad = grouped[grouped > 1]
    if not bad.empty:
        curves = ", ".join(bad.index.tolist())
        raise ValueError(f"Non-parallel shocks detected for curves: {curves}")

