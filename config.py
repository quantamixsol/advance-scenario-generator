# config.py

# ─── Geographic Groupings ────────────────────────────────────────────
REGION_GROUP = {
    "Northern America": "NA",
    "Latin America":    "LATAM",
    "Asia":             "APAC",
    "Oceania":          "APAC",
    "Northwest Europe": "EMEA",
    "Southern Europe":  "EMEA",
    "AroundAfrica":     "EMEA",
}

# ─── Credit Rating Scores ────────────────────────────────────────────
# (Used for normalized “distance” between two letter ratings)
RATING_SCORE = {
    "AAA":  5.0, "AA+": 4.5, "AA":  4.0, "AA-": 3.5,
    "A+":   3.0, "A":   2.5, "A-":  2.0,
    "BBB+": 1.5, "BBB": 1.0, "BB":  0.5, "B":   0.0
}

# ─── Asset Class Schemas ────────────────────────────────────────────
# Each entry lists the fields parsed from “code” and default weight 1.0
ASSET_CONFIG = {
    "BOND": {
        "fields": ["rating","tenor","seniority","sector","region","currency","covered","shock"],
        "default_weight": 1.0
    },
    "CDS": {
        "fields": ["instrument","issuer","currency","seniority","liquidity","maturity","shock"],
        "default_weight": 1.0
    },
    "CR": {
        "fields": ["instrument","issuer","currency","seniority","liquidity","maturity","shock"],
        "default_weight": 1.0
    },
    "FXSPOT": {
        "fields": ["spot","pair","shock"],
        "default_weight": 1.0
    },
    "FXVOL": {
        "fields": ["smile","type","pair","strike","option_type","tenor","shock"],
        "default_weight": 1.0
    },
    "IRSWAP": {
        "fields": ["instrument","curve_name","currency","tenor","shock"],
        "default_weight": 1.0
    },
    "IRSWVOL": {
        "fields": ["method","currency","tenor","shock"],
        "default_weight": 1.0
    },
    "IR_LINEAR": {
        "fields": ["instrument","curve_name","rate","currency","tenor","shock"],
        "default_weight": 1.0
    },
    "CMD":       {"fields": [
                      "Asset Name",
                      "Currency",
                      "Expiry",
                      "Unit",
                      "Sector",
                      "shock"
                  ],
                  "default_weight": 1.0},

    # ─── NEW! EQUITY ────────────────────────────────────────────────
    "EQ":        {"fields": [
                      "Asset Name",
                      "Country",
                      "Sector",
                      "Liquidity category",
                      "Rating",
                      "shock"
                  ],
                  "default_weight": 1.0},

}

# ─── SEM Shock-Expansion Methods ─────────────────────────────────────
# Categorized per CRISIL SEM methodology
SEM_METHODS = {
    "Pre-Determined": [
        "user_defined",
        "quantile",
        "wss",            # worst simultaneous shock
        "realized_hist"   # realized historical shocks
    ],
    "Model-Driven": [
        "mlr",  # multi-linear regression
        "hlr",  # hierarchical linear reg
        "pr",   # polynomial regression
        "lpca", # linear PCA
        "ecm",  # error-correction
        "varx"  # vector autoregression
    ],
    "Business-Rule": [
        "interpolation",
        "proxy_scalar",
        "triangulation"
    ]
}

# ─── Utility: Build per-asset weight dicts ──────────────────────────
# We’ll use ASSET_CONFIG[asset]["default_weight"] to initialize real weight dicts.
def default_weights():
    return {
        asset: { fld: cfg["default_weight"] for fld in cfg["fields"] }
        for asset, cfg in ASSET_CONFIG.items()
    }
