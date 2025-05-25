# advance-scenario-generator

## Vertex AI credentials

The application expects a Google service account JSON for Vertex AI.
Provide the file path via the environment variable `VERTEXAI_CREDENTIALS`.
For example:

```bash
export VERTEXAI_CREDENTIALS=/path/to/your/service_account.json
```

This keeps the credentials file outside the repository.

Alternatively you can use a Streamlit secrets file.  Create
`.streamlit/secrets.toml` at the project root with a line like:

```toml
VERTEXAI_CREDENTIALS = "/path/to/your/service_account.json"
```

When running `streamlit run app.py` the path (or the JSON content itself)
will be read from this file if the environment variable is not set.

## Command-line pipeline

The script `pipeline_flow.py` runs the full process:

1. Load a portfolio CSV/Excel/JSON file.
2. Parse a universe of risk factors.
3. Automatically map portfolio tickers to risk-factor codes.
4. Apply baseline shocks by asset class and severity.
5. Output a scenario PnL CSV.

Example:

```bash
python pipeline_flow.py --portfolio synthetic_portfolio.csv \
                       --universe synthetic_universe.csv \
                       --severity Medium --output pnl.csv
```

This will create `pnl.csv` with the shocked PnL per position.

## Portfolio Fields

Portfolio files must contain `ticker`, `quantity`, and `price` columns.  The
system will also make use of optional risk metrics when present:

- `duration` or `dv01` for fixed income
- `spread_dv01` and `recovery_rate` for CDS
- `delta` or `vega` for options and volatility products
- `fx_rate` to convert non-USD PnL to USD

Including these columns results in more accurate PnL calculations.  If omitted,
default assumptions based on tenor and asset class are used.

## Custom Baseline Shocks

Baseline shocks by asset class can be overridden with a JSON file mapping asset
codes to shock magnitudes.  Pass the file with `--baseline_config`:

```bash
python pipeline_flow.py --portfolio mypf.csv --universe universe.csv \
                       --baseline_config custom_shocks.json
```

## Pricing Methodology

Equities, FX, and commodities apply percentage shocks to market value times
delta.  Fixed income instruments use duration or DV01 if available.  CDS
instruments accept `spread_dv01` inputs.  Volatility products can specify Vega to
convert volatility point shocks into PnL.  See `pricing.py` for details.

## LLM-Generated Scenario Flow

`pipeline_flow.py` also exposes an end-to-end workflow that uses an LLM to
create a scenario narrative and shock the mapped risk factors.  Provide a
proxy file and specify a scenario name:

```bash
python pipeline_flow.py \
    --portfolio mypf.csv \
    --universe universe.csv \
    --proxies proxies.csv \
    --scenario_name "Rates Up" \
    --user_input "aggressive central bank hikes"
```

The script will map portfolio tickers to the universe (falling back to proxies),
generate a narrative, assign shocks via the LLM, apply pricing, and output the
scenario PnL.
