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
