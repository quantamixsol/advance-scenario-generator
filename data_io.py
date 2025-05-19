# data_io.py
import re
import pandas as pd

# ─── GLOBAL LOOKUPS ──────────────────────────────────────────────────────
REGION_GROUP = {
    "Northern America":"NA","Latin America":"LATAM",
    "Asia":"APAC","Oceania":"APAC",
    "Northwest Europe":"EMEA","Southern Europe":"EMEA","AroundAfrica":"EMEA"
}
RATING_SCORE = {
    "AAA":5,"AA+":4.5,"AA":4,"AA-":3.5,
    "A+":3,"A":2.5,"A-":2,
    "BBB+":1.5,"BBB":1,"BB":0.5,"B":0
}

# ─── ASSET SCHEMA & DEFAULT WEIGHTS ─────────────────────────────────────
ASSET_CONFIG = {
    "BOND":      {"fields":["rating","tenor","seniority","sector","region","currency","covered","shock"]},
    "CDS":       {"fields":["instrument","issuer","currency","seniority","liquidity","maturity","shock"]},
    "CR":        {"fields":["instrument","issuer","currency","seniority","liquidity","maturity","shock"]},
    "FXSPOT":    {"fields":["spot","pair","shock"]},
    "FXVOL":     {"fields":["smile","type","pair","strike","option_type","tenor","shock"]},
    "IRSWAP":    {"fields":["instrument","curve_name","currency","tenor","shock"]},
    "IRSWVOL":   {"fields":["method","currency","tenor","shock"]},
    "IR_LINEAR": {"fields":["instrument","curve_name","rate","currency","tenor","shock"]},
}
# fill in default weight=1.0 for every field
for cfg in ASSET_CONFIG.values():
    cfg["weights"] = {f:1.0 for f in cfg["fields"]}

# ─── PARSING LOGIC ──────────────────────────────────────────────────────
def parse_row(code: str) -> dict:
    parts = [p.strip() for p in code.split(":")]
    # drop leading 'CR'
    if parts and parts[0].upper()=="CR":
        parts = parts[1:]
    a0 = parts[0].upper()
    if a0 in ("CDS","CR"):
        asset = "CDS"
    elif a0=="BOND":
        asset = "BOND"
    elif a0=="FXSPOT":
        asset = "FXSPOT"
    elif a0=="FXVOL":
        asset = "FXVOL"
    elif a0=="IR" and len(parts)>1 and parts[1].upper()=="SWAP" and len(parts)==5:
        asset = "IRSWAP"
    elif a0=="IR" and len(parts)>1 and parts[1].upper() in ("SWVOL","SWAPVOL"):
        asset = "IRSWVOL"
    elif a0=="IR":
        asset = "IR_LINEAR"
    else:
        asset = a0
    out = {"asset": asset, "original": code}
    for i, fld in enumerate(ASSET_CONFIG.get(asset, {})["fields"], start=1):
        out[fld] = parts[i] if i < len(parts) else ""
    return out

def parse_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df["code"] column. Returns one-row-per-code DataFrame
    with columns: asset, original, plus each field from ASSET_CONFIG.
    """
    return pd.json_normalize(df["code"].astype(str).apply(parse_row))
