# Explorer Web UI

This directory contains the FastAPI backend (`explorer/web/server.py`) and the Vite/TypeScript frontend located in `explorer/web/explorer-frontend`.

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
