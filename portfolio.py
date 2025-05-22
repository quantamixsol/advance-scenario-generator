# portfolio.py
"""Utilities for portfolio ingestion and risk-factor mapping."""
import pandas as pd
import re
from rapidfuzz import process
import numpy as np

# try optional embedding engine from embeddings.py
try:
    from embeddings import embed_texts, EMBED_DIM
    import faiss
    from sklearn.preprocessing import normalize
    _SEMANTIC = True
except Exception:
    _SEMANTIC = False

class PortfolioIngestor:
    """Load and clean user portfolio files."""

    REQUIRED = {"ticker", "quantity", "price"}
    OPTIONAL = {"duration", "dv01", "spread_dv01", "recovery_rate", "delta", "vega", "fx_rate", "currency"}

    def __init__(self, path_or_buf):
        self.raw = self._load_file(path_or_buf)
        self.df = self._clean(self.raw)

    def _load_file(self, path_or_buf) -> pd.DataFrame:
        """Load CSV/Excel/JSON into DataFrame with lowercase columns."""
        if hasattr(path_or_buf, "read"):
            # assume file-like object
            name = getattr(path_or_buf, "name", "").lower()
            ext = name.split(".")[-1]
            data = path_or_buf
        else:
            name = str(path_or_buf).lower()
            ext = name.split(".")[-1]
            data = path_or_buf
        if ext in ("xls", "xlsx"):
            df = pd.read_excel(data)
        elif ext == "json":
            df = pd.read_json(data)
        else:
            df = pd.read_csv(data)
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns, enforce numeric types, compute weights."""
        cols = set(df.columns)
        mapping = {}
        for req in self.REQUIRED:
            match, score, _ = process.extractOne(req, list(cols), score_cutoff=60)
            if not match:
                raise ValueError(f"Could not detect '{req}' column in {cols}")
            mapping[match] = req
        df = df.rename(columns=mapping)
        optional = {c:c for c in df.columns if c in self.OPTIONAL}
        df = df[list(self.REQUIRED) + list(optional.keys())].copy()
        df["quantity"] = pd.to_numeric(df.quantity, errors="coerce")
        df["price"] = pd.to_numeric(df.price, errors="coerce")
        for col in self.OPTIONAL:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ticker_norm"] = df.ticker.astype(str).str.upper().str.split(r"\s|\.|/").str[0]
        df["mkt_value"] = df.quantity * df.price
        tot = df["mkt_value"].sum()
        df["weight"] = df["mkt_value"] / (tot if tot > 0 else 1)
        return df

    def get(self) -> pd.DataFrame:
        return self.df

class RiskFactorMapper:
    """Map portfolio rows to risk-factor codes."""

    def __init__(self, universe: pd.DataFrame):
        self.univ = universe.copy()
        self._build_exact_index()
        if _SEMANTIC:
            self._build_semantic_index()

    def _build_exact_index(self):
        rows = []
        for _, r in self.univ.iterrows():
            key = None
            for col in ("symbol", "issuer", "instrument", "pair", "commodity"):
                if col in r and pd.notna(r[col]) and r[col]:
                    key = str(r[col]).upper().strip()
                    break
            if key:
                rows.append((key, r.original))
        self.exact = pd.DataFrame(rows, columns=["key", "rf"])

    def _build_semantic_index(self):
        texts = self.exact["key"].tolist()
        embs = embed_texts(texts).astype("float32")
        self.embs = normalize(embs, axis=1)
        self.idx = faiss.IndexFlatIP(self.embs.shape[1])
        self.idx.add(self.embs)
        self._keys = texts

    def map(self, port_df: pd.DataFrame) -> pd.DataFrame:
        df = port_df.copy()
        df["rf_exact"] = df["ticker_norm"].map(dict(zip(self.exact.key, self.exact.rf)))
        mask = df["rf_exact"].isna()
        choices = self.exact["key"].tolist()
        def _fuzzy(t):
            m, sc, idx = process.extractOne(t, choices)
            return self.exact.rf.iloc[idx] if sc > 75 else None
        df.loc[mask, "rf_fuzzy"] = df.loc[mask, "ticker_norm"].map(_fuzzy)
        if _SEMANTIC:
            miss = df["rf_exact"].fillna(df["rf_fuzzy"]).isna()
            if miss.any():
                q = embed_texts(df.loc[miss, "ticker_norm"].tolist()).astype("float32")
                q = normalize(q, axis=1)
                D, I = self.idx.search(q, 1)
                df.loc[miss, "rf_sem"] = [self.exact.rf.iloc[i] if d > 0.4 else None for d, i in zip(D[:,0], I[:,0])]
        # combine mappings without calling fillna(None) when 'rf_sem' is absent
        df["rf_code"] = df["rf_exact"].fillna(df["rf_fuzzy"])
        if "rf_sem" in df:
            df["rf_code"] = df["rf_code"].fillna(df["rf_sem"])
        return df
