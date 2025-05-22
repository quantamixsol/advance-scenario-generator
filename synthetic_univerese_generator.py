# synthetic_universe_generator.py
# ──────────────────────────────────────────────────────────────────────────────
# Dependencies: pandas, numpy
# Run with: python synthetic_universe_generator.py
import csv
import random
import itertools
import pandas as pd

# --- Asset classes and field templates (must match data_io.ASSET_CONFIG) ---
ASSETS = {
    "BOND": {
        "ratings":      ["AAA","AA+","AA","A+","A","BBB","BB"],
        "tenors":       ["6M","1Y","2Y","5Y","10Y","20Y"],
        "seniorities":  ["Senior","Subordinated"],
        "sectors":      ["Corp","Financial","Utilities","Industrial"],
        "regions":      ["Northern America","Latin America","Asia","Oceania","Northwest Europe","Southern Europe","AroundAfrica"],
        "currencies":   ["USD","EUR","GBP","JPY"],
        "covered":      ["Yes","No"],
    },
    "CDS": {
        "instruments":  ["CDX","iTraxx","Custom"],
        "issuers":      [f"Issuer{c}" for c in list("ABCDEFGHIJ")],
        "currencies":   ["USD","EUR","GBP","JPY"],
        "seniorities":  ["Senior","Subordinated"],
        "liquidities":  ["High","Medium","Low"],
        "maturities":   ["1Y","2Y","5Y","10Y"],
    },
    "CR": {
        "instruments":  ["CRX","CRY","Custom"],
        "issuers":      [f"Issuer{c}" for c in list("KLMNOPQRST")],
        "currencies":   ["USD","EUR","GBP","JPY"],
        "seniorities":  ["Senior","Subordinated"],
        "liquidities":  ["High","Medium","Low"],
        "maturities":   ["1Y","2Y","5Y","10Y"],
    },
    "FXSPOT": {
        "spots":        ["1.1000","0.8500","130.50","0.0068","0.7500"],
        "pairs":        ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD"],
    },
    "FXVOL": {
        "smiles":       ["ATM","RiskRev","Butterfly"],
        "types":        ["IV","Vega"],
        "pairs":        ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD"],
        "strikes":      ["0.5","1.0","1.5","2.0"],
        "opt_types":    ["Call","Put"],
        "tenors":       ["1M","3M","6M","1Y"],
    },
    "IRSWAP": {
        "curve_names":  ["USD-LIBOR","EUR-EURIBOR","GBP-LIBOR"],
        "currencies":   ["USD","EUR","GBP"],
        "tenors":       ["6M","1Y","2Y","5Y","10Y","20Y"],
    },
    "IRSWVOL": {
        "methods":      ["Lognormal","Normal","SABR"],
        "currencies":   ["USD","EUR","GBP"],
        "tenors":       ["1M","3M","6M","1Y"],
    },
    "IR_LINEAR": {
        "curve_names":  ["US-LINEAR","EU-LINEAR","UK-LINEAR"],
        "rates":        ["0.005","0.01","0.015","0.02","0.025"],
        "currencies":   ["USD","EUR","GBP"],
        "tenors":       ["6M","1Y","2Y","5Y","10Y","20Y"],
    },
    "EQ": {
        "symbols":      ["AAPL","MSFT","GOOGL","AMZN","TSLA","JPM","BAC","V","JNJ","WMT"],
        "sectors":      ["Technology","Financials","Healthcare","Consumer","Industrial"],
        "regions":      ["Northern America","Europe","Asia"],
    },
    "CMD": {
        "commodities":  ["Gold","Silver","Oil","NaturalGas","Copper","Soybeans"],
        "regions":      ["Global","Northern America","Asia","Europe"],
        "prices":       ["50.5","75.0","100.0","125.25","150.75"],
    },
}

# baseline “shock” placeholder:
SHOCK_PLACEHOLDER = "Base"

# total rows and per‐asset allocation
TOTAL = 1000
per_asset = TOTAL // len(ASSETS)

codes = []

for asset, tpl in ASSETS.items():
    for _ in range(per_asset):
        if asset == "BOND":
            codes.append(
                ":".join([
                    asset,
                    random.choice(tpl["ratings"]),
                    random.choice(tpl["tenors"]),
                    random.choice(tpl["seniorities"]),
                    random.choice(tpl["sectors"]),
                    random.choice(tpl["regions"]),
                    random.choice(tpl["currencies"]),
                    random.choice(tpl["covered"]),
                    SHOCK_PLACEHOLDER
                ])
            )
        elif asset in ("CDS","CR"):
            codes.append(
                ":".join([
                    asset,
                    random.choice(tpl["instruments"]),
                    random.choice(tpl["issuers"]),
                    random.choice(tpl["currencies"]),
                    random.choice(tpl["seniorities"]),
                    random.choice(tpl["liquidities"]),
                    random.choice(tpl["maturities"]),
                    SHOCK_PLACEHOLDER
                ])
            )
        elif asset == "FXSPOT":
            codes.append(
                ":".join([
                    asset,
                    random.choice(tpl["spots"]),
                    random.choice(tpl["pairs"]),
                    SHOCK_PLACEHOLDER
                ])
            )
        elif asset == "FXVOL":
            codes.append(
                ":".join([
                    asset,
                    random.choice(tpl["smiles"]),
                    random.choice(tpl["types"]),
                    random.choice(tpl["pairs"]),
                    random.choice(tpl["strikes"]),
                    random.choice(tpl["opt_types"]),
                    random.choice(tpl["tenors"]),
                    SHOCK_PLACEHOLDER
                ])
            )
        elif asset == "IRSWAP":
            codes.append(
                ":".join([
                    "IR","SWAP",
                    random.choice(tpl["curve_names"]),
                    random.choice(tpl["currencies"]),
                    random.choice(tpl["tenors"])
                ])
            )
        elif asset == "IRSWVOL":
            codes.append(
                ":".join([
                    "IR","SWVOL",
                    random.choice(tpl["currencies"]),
                    random.choice(tpl["tenors"])
                ])
            )
        elif asset == "IR_LINEAR":
            codes.append(
                ":".join([
                    "IR",
                    random.choice(tpl["curve_names"]),
                    random.choice(tpl["rates"]),
                    random.choice(tpl["currencies"]),
                    random.choice(tpl["tenors"])
                ])
            )
        elif asset == "EQ":
            codes.append(
                ":".join([
                    asset,
                    random.choice(tpl["symbols"]),
                    random.choice(tpl["sectors"]),
                    random.choice(tpl["regions"]),
                    SHOCK_PLACEHOLDER
                ])
            )
        elif asset == "CMD":
            codes.append(
                ":".join([
                    asset,
                    random.choice(tpl["commodities"]),
                    random.choice(tpl["regions"]),
                    random.choice(tpl["prices"]),
                    SHOCK_PLACEHOLDER
                ])
            )

# If there is any remainder, fill with random picks
while len(codes) < TOTAL:
    codes.append(random.choice(codes))

# Shuffle and write out
random.shuffle(codes)
df = pd.DataFrame({"code": codes})
df.to_csv("synthetic_universe.csv", index=False)
print(f"Written {len(df)} rows to synthetic_universe.csv")
