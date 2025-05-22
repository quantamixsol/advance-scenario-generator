import re
import pandas as pd

__all__ = [
    "get_pricing_function",
    "aggregate_asset_pnl",
]

def _tenor_to_years(s: str) -> float:
    t = str(s or "").strip().upper()
    if t == "ON":
        return 1/12
    m = re.match(r"^(\d+(?:\.\d+)?)([DWMY])$", t)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    months = {"D":1/30, "W":7/30, "M":1, "Y":12}[unit] * val
    return months / 12

def _equity_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    return -row["quantity"] * row["price"] * pct

def _fx_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    return -row["quantity"] * row.get("price", 1.0) * pct

def _commodity_pnl(row: pd.Series) -> float:
    pct = row["shock_pct"] / 100
    return -row["quantity"] * row["price"] * pct

def _bond_pnl(row: pd.Series) -> float:
    bps = row["shock_pct"]
    yrs = _tenor_to_years(row.get("tenor")) or 5.0
    return -row["quantity"] * row["price"] * yrs * (bps / 10000)

def _cds_pnl(row: pd.Series) -> float:
    return _bond_pnl(row)

def _ir_pnl(row: pd.Series) -> float:
    bps = row["shock_pct"]
    yrs = _tenor_to_years(row.get("tenor")) or 1.0
    return -row["quantity"] * row["price"] * yrs * (bps / 10000)

_PRICERS = {
    "EQ": _equity_pnl,
    "FXSPOT": _fx_pnl,
    "CMD": _commodity_pnl,
    "BOND": _bond_pnl,
    "CDS": _cds_pnl,
    "CR": _cds_pnl,
    "IRSWAP": _ir_pnl,
    "IRSWVOL": _ir_pnl,
    "IR_LINEAR": _ir_pnl,
}

def get_pricing_function(asset: str):
    return _PRICERS.get(asset)

def aggregate_asset_pnl(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("asset")["pnl"].sum().reset_index().rename(columns={"pnl":"asset_pnl"})
