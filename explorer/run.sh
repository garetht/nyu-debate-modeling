#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${REPO_ROOT}/explorer/web/explorer-frontend"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && ps -p "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]] && ps -p "${FRONTEND_PID}" >/dev/null 2>&1; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  wait || true
}

handle_signal() {
  cleanup
  exit 1
}

trap handle_signal INT TERM

(
  cd "${REPO_ROOT}"
  python -m explorer.web.explorer_backend.server
) &
SERVER_PID=$!

(
  cd "${FRONTEND_DIR}"
  npm run dev -- --host
) &
FRONTEND_PID=$!

wait "${SERVER_PID}" "${FRONTEND_PID}"
status=$?
cleanup
exit "${status}"
