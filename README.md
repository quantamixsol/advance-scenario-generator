# advance-scenario-generator

## Vertex AI credentials

The application expects a Google service account JSON for Vertex AI.
Provide the file path via the environment variable `VERTEXAI_CREDENTIALS`.
For example:

```bash
export VERTEXAI_CREDENTIALS=/path/to/your/service_account.json
```

This keeps the credentials file outside the repository.
