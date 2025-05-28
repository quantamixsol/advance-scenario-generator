import pandas as pd
from config import SHOCK_UNITS


def compute_pnl(portfolio_df: pd.DataFrame, risk_factors_df: pd.DataFrame, shock_values) -> pd.DataFrame:
    """Compute shocked P&L for a portfolio.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio holdings with ``ticker``, ``quantity``, ``price`` and a
        ``rf_code`` column mapping each row to a risk factor.
    risk_factors_df : pd.DataFrame
        DataFrame describing risk factors. Must contain the risk-factor code
        (``original`` or ``rf_code``) and ``asset`` columns so the shock unit can
        be determined.
    shock_values : pd.DataFrame or dict
        Shock percentages for each risk factor.  If a DataFrame is provided it
        should have a ``shock_pct`` column and either ``original`` or
        ``rf_code`` for the code.  A dictionary may also be supplied mapping
        risk-factor code to shock percent/bps.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ticker``, ``pre_notional``, ``post_notional``
        and ``pnl``.  The function also prints the total portfolio P&L.
    """
    pf = portfolio_df.copy()
    rf = risk_factors_df.copy()

    # unify column names for joins
    if "original" in rf.columns and "rf_code" not in rf.columns:
        rf = rf.rename(columns={"original": "rf_code"})

    if isinstance(shock_values, pd.DataFrame):
        shocks = shock_values.copy()
        if "original" in shocks.columns and "rf_code" not in shocks.columns:
            shocks = shocks.rename(columns={"original": "rf_code"})
        shocks = shocks[["rf_code", "shock_pct"]]
    else:
        # assume mapping dict
        shocks = pd.DataFrame(list(shock_values.items()), columns=["rf_code", "shock_pct"])

    # join portfolio -> rf -> shocks
    pf = pf.merge(rf[["rf_code", "asset"]], on="rf_code", how="left")
    pf = pf.merge(shocks, on="rf_code", how="left")

    pf["pre_notional"] = pf["quantity"] * pf["price"]

    def _post_price(row):
        shock = row.get("shock_pct")
        if pd.isna(shock):
            return row["price"]
        unit = SHOCK_UNITS.get(row.get("asset"), "pct")
        pct = shock / 10000 if unit == "bps" else shock / 100
        return row["price"] * (1 + pct)

    pf["post_price"] = pf.apply(_post_price, axis=1)
    pf["post_notional"] = pf["quantity"] * pf["post_price"]
    pf["pnl"] = pf["post_notional"] - pf["pre_notional"]

    total_pnl = pf["pnl"].sum()
    print(f"Total P&L: {total_pnl:,.2f}")

    return pf[["ticker", "pre_notional", "post_notional", "pnl"]]
