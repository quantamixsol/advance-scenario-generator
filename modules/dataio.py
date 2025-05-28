"""Portfolio ingestion utilities."""
import pandas as pd
from typing import Any

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    yf = None  # pragma: no cover


def _read_file(path: str) -> pd.DataFrame:
    """Read CSV or Excel into DataFrame with normalized columns."""
    ext = str(path).split(".")[-1].lower()
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _lookup_metadata(ticker: str) -> dict[str, Any]:
    """Fetch ticker metadata via yfinance if available."""
    if yf is None:
        return {}
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def load_portfolio(path: str) -> pd.DataFrame:
    """Load and enrich a portfolio file.

    Parameters
    ----------
    path:
        Path to CSV or Excel file with required columns ``ticker``, ``quantity``,
        ``price``, ``asset_class``, ``region`` and ``sector``.

    Returns
    -------
    pandas.DataFrame
        Portfolio DataFrame with a ``notional`` column and any metadata returned
        from the ticker lookup.
    """
    df = _read_file(path)

    rename = {
        "assetclass": "asset_class",
        "asset class": "asset_class",
    }
    df = df.rename(columns=rename)

    required = ["ticker", "quantity", "price", "asset_class", "region", "sector"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df[required].copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["notional"] = df["quantity"] * df["price"]

    # lookup metadata and validate tickers
    metas: dict[str, dict[str, Any]] = {}
    for t in df["ticker"].astype(str).str.upper().unique():
        metas[t] = _lookup_metadata(t)

    # fill missing metadata
    meta_cols = set()
    for info in metas.values():
        meta_cols.update(info.keys())
    for col in meta_cols:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        info = metas.get(str(row["ticker"]).upper(), {})
        for col, val in info.items():
            if col not in df.columns:
                continue
            if pd.isna(row.get(col)) or row.get(col) in ("", None):
                df.at[idx, col] = val

    return df
