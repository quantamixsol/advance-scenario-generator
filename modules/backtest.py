"""Backtesting utilities for running scenario-driven simulations."""

from __future__ import annotations

import pandas as pd
from typing import Callable, Tuple


def backtest(
    portfolio_df: pd.DataFrame,
    historical_prices: pd.DataFrame,
    scenario_generator: Callable[[pd.Timestamp, pd.DataFrame], dict],
    shock_simulator: Callable[[pd.DataFrame, dict], float],
    lookback: int | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """Run a simple backtest using generated scenarios and shocks.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Current portfolio positions.
    historical_prices : pd.DataFrame
        Market data indexed by date.
    scenario_generator : callable
        Function called with ``(date, history)`` returning a scenario description.
    shock_simulator : callable
        Function called with ``(portfolio_df, scenario)`` returning a P\&L value.
    lookback : int, optional
        Number of periods from ``historical_prices`` to use.  Defaults to the
        entire history.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame suitable for charting with columns ``['date', 'pnl',
        'cum_pnl']`` and a dictionary of basic performance metrics.
    """

    history = historical_prices.copy()
    if not isinstance(history.index, pd.DatetimeIndex):
        history.index = pd.to_datetime(history.index)

    if lookback is not None:
        history = history.iloc[-lookback:]

    records: list[dict] = []
    for dt in history.index:
        past = history.loc[:dt]
        scenario = scenario_generator(dt, past)
        pnl = shock_simulator(portfolio_df, scenario)
        records.append({"date": dt, "pnl": pnl})

    result = pd.DataFrame(records).set_index("date").sort_index()
    result["cum_pnl"] = result["pnl"].cumsum()

    metrics = {
        "total_pnl": result["pnl"].sum(),
        "mean_pnl": result["pnl"].mean(),
        "max_drawdown": (result["cum_pnl"].cummax() - result["cum_pnl"]).max(),
    }

    return result, metrics
