import pandas as pd
from pipeline_flow import portfolio_assets
from exposures import asset_pnl_breakdown

def test_portfolio_assets():
    mapped = pd.DataFrame({'rf_code':['RF1','RF2']})
    univ = pd.DataFrame({'original':['RF1','RF2','RF3'],
                         'asset':['EQ','BOND','FX']})
    assets = portfolio_assets(mapped, univ)
    assert set(assets) == {'EQ', 'BOND'}

def test_asset_pnl_breakdown():
    df = pd.DataFrame({'asset':['EQ', None], 'pnl':[10, -5]})
    brk = asset_pnl_breakdown(df)
    assert set(brk.asset) == {'EQ', 'UNKNOWN'}
    eq_pnl = brk.loc[brk.asset=='EQ','asset_pnl'].iloc[0]
    unk_pnl = brk.loc[brk.asset=='UNKNOWN','asset_pnl'].iloc[0]
    assert eq_pnl == 10
    assert unk_pnl == -5
