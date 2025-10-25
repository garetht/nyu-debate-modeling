# Explorer Web UI

This directory contains the FastAPI backend (`explorer/web/explorer_backend/server.py`) and the Vite/TypeScript frontend located in `explorer/web/explorer-frontend`.

## Run the Explorer locally

1. Install dependencies:
   - Python: make sure the virtual environment for this repository is active and has the FastAPI dependencies installed.
   - Frontend: from `explorer/web/explorer-frontend` run `npm install` (only required once).

2. From the repository root, make the helper script executable:

   ```bash
   chmod +x explorer/run.sh
   ```

3. Start both the backend API and the frontend dev server with a single command:

   ```bash
   ./explorer/run.sh
   ```

   - The backend runs on `http://127.0.0.1:8067`.
   - The frontend Vite dev server runs on the host/port reported by `npm run dev` (default `http://localhost:5173`).

Press `Ctrl+C` to stop both processes; the script will clean up the FastAPI server and the Vite dev server before exiting.

## Generate the OpenAPI schema and TypeScript client

The backend exposes an OpenAPI schema that can be exported together with a typed REST client for the frontend via `explorer/explorer_backend/scripts/generate_schema.py`. The script performs two steps:

1. Instantiate the FastAPI application defined in `explorer/web/explorer_backend/server.py` and write its OpenAPI schema to `schemas/explorer_web_server.schema.json` (created automatically if missing).
2. Run `npx --yes openapi-typescript-codegen` against that schema to produce a fetch-based TypeScript client under `explorer/web/explorer-frontend/src/clients/explorer`. The script first probes the generator (`npx … --help`) so you get an immediate error if the CLI is unavailable.

Run the default workflow from the repository root:

```bash
python explorer/explorer_backend/scripts/generate_schema.py
```

The command prints the schema path and client directory after a successful run. You can customise the destinations or the generator invocation:

- `--schema-output PATH` controls where the OpenAPI JSON is written (alias: deprecated `--output`).
- `--client-output-dir DIR` changes the generated client output directory.
- `--generator-command ...` overrides the generator executable (e.g. to call a globally installed binary).
- `--generator-args ...` appends extra arguments supported by `openapi-typescript-codegen` (defaults to `--client fetch`).

For example, to emit the schema next to the backend and generate a node-style client without reinstalling the generator each run:

```bash
python explorer/explorer_backend/scripts/generate_schema.py \
  --schema-output explorer/web/schema.json \
  --client-output-dir explorer/web/explorer-frontend/src/clients/explorer-node \
  --generator-command openapi-typescript-codegen \
  --generator-args --client node
```

If the generator exits with a non-zero status the script reports the failing command and stops, leaving the existing schema/client untouched. The exported schema directory is safe to commit so downstream tooling stays in sync with the API.
