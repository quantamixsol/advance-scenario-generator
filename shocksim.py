# shocksim.py
import numpy as np
from config import SEM_METHODS

def simulate_shocks(rf_df, method: str, params: dict) -> pd.DataFrame:
    """
    Given a DataFrame of risk factors (rf_df) and selected SEM method,
    returns the same DF with an added 'shock' column.
    """
    if method == "user_defined":
        # user already entered rf_df["shock"]
        return rf_df
    elif method == "quantile":
        # stub: compute historical quantile shocks
        pass
    elif method == "wss":
        # worst simultaneous shock over historical series
        pass
    elif method in ("mlr","hlr","pr","lpca","ecm","varx"):
        # stub: fit/regress using time series library (statsmodels or sklearn)
        pass
    elif method in ("interpolation","proxy","triangulation"):
        # stub: simple spline or scalar proxy
        pass
    else:
        raise ValueError(f"Unknown SEM method {method}")
