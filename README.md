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
