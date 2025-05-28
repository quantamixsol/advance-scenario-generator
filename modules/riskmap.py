"""Risk factor generation and mapping utilities."""

from __future__ import annotations

import pandas as pd
from typing import List

from data_io import parse_df
from config import ASSET_CONFIG


def _build_code(row: pd.Series) -> str:
    """Construct a risk-factor code from a portfolio row."""
    asset = str(row.get("asset", "EQ")).upper().strip()
    if asset not in ASSET_CONFIG:
        asset = "EQ"
    fields = ASSET_CONFIG[asset]["fields"]
    parts: List[str] = [asset]
    for f in fields:
        if f == "shock":
            val = row.get(f, "Base") or "Base"
        else:
            val = row.get(f, "")
        parts.append(str(val) if pd.notna(val) else "")
    return ":".join(parts)


def map_to_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Convert portfolio rows to normalized risk factors.

    Parameters
    ----------
    df : DataFrame
        Portfolio positions with optional columns matching fields in
        ``config.ASSET_CONFIG``.

    Returns
    -------
    DataFrame
        Deduplicated risk-factor table compatible with the parsed universe.
    """
    codes = [_build_code(row) for _, row in df.iterrows()]
    rf_df = parse_df(pd.DataFrame({"code": codes}))
    rf_df = rf_df.drop_duplicates("original").reset_index(drop=True)
    return rf_df
