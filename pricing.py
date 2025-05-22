import re
import pandas as pd

__all__ = [
    "get_pricing_function",
    "aggregate_asset_pnl",
]

def _tenor_to_years(s: str) -> float:
    """Convert tenor strings like '5Y', '3M', '10D' to years."""
    if not s:
        return 0.0
    t = str(s).strip().upper()
    if t == "ON":
        return 1 / 12
    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", t)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    months = {"D": 1 / 30, "W": 7 / 30, "M": 1, "Y": 12}[unit] * val
    return months / 12

def _equity_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    delta = row.get("delta", 1.0)
    return -row["quantity"] * row["price"] * delta * pct

def _fx_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    delta = row.get("delta", 1.0)
    return -row["quantity"] * row.get("price", 1.0) * delta * pct

def _commodity_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    delta = row.get("delta", 1.0)
    return -row["quantity"] * row["price"] * delta * pct

def _bond_pnl(row: pd.Series) -> float:
    bps = row["shock_pct"]
    if pd.notna(row.get("dv01")):
        return -row["quantity"] * row["dv01"] * bps
    yrs = row.get("duration") or _tenor_to_years(row.get("tenor")) or 5.0
    price = row.get("price", 1.0)
    return -row["quantity"] * price * yrs * (bps / 10000)

def _cds_pnl(row: pd.Series) -> float:
    bps = row["shock_pct"]
    if pd.notna(row.get("spread_dv01")):
        return -row["quantity"] * row["spread_dv01"] * bps
    return _bond_pnl(row)

def _ir_pnl(row: pd.Series) -> float:
    bps = row["shock_pct"]
    if pd.notna(row.get("dv01")):
        return -row["quantity"] * row["dv01"] * bps
    yrs = row.get("duration") or _tenor_to_years(row.get("tenor")) or 1.0
    price = row.get("price", 1.0)
    return -row["quantity"] * price * yrs * (bps / 10000)

def _option_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    delta = row.get("delta", 1.0)
    return -row["quantity"] * row.get("price", 1.0) * delta * pct

def _vega_pnl(row: pd.Series) -> float:
    bps = row["shock_pct"]
    vega = row.get("vega")
    if pd.isna(vega):
        return _ir_pnl(row)
    return -row["quantity"] * vega * bps

def _future_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    return -row["quantity"] * row.get("price", 1.0) * pct

_PRICERS = {
    "EQ": _equity_pnl,
    "FXSPOT": _fx_pnl,
    "CMD": _commodity_pnl,
    "BOND": _bond_pnl,
    "CDS": _cds_pnl,
    "CR": _cds_pnl,
    "IRSWAP": _ir_pnl,
    "IRSWVOL": _vega_pnl,
    "IR_LINEAR": _ir_pnl,
    "EQ_OPT": _option_pnl,
    "FUT": _future_pnl,
    "FXVOL": _vega_pnl,
}

def get_pricing_function(asset: str):
    return _PRICERS.get(asset)

def aggregate_asset_pnl(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("asset")["pnl"].sum().reset_index().rename(columns={"pnl":"asset_pnl"})
