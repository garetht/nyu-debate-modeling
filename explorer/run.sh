#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BACKEND_PID=""
FRONTEND_PID=""
BACKEND_WAITER_PID=""
FRONTEND_WAITER_PID=""
CLEANED_UP=0
STATUS_DIR="$(mktemp -d -t explorer-run)"

log() {
  echo "[explorer/run.sh] $*"
}

cleanup() {
  if [[ "${CLEANED_UP}" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  for pid_var in BACKEND_PID FRONTEND_PID; do
    pid="${!pid_var:-}"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      log "Stopping ${pid_var} (PID ${pid})"
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done

  for waiter_var in BACKEND_WAITER_PID FRONTEND_WAITER_PID; do
    waiter_pid="${!waiter_var:-}"
    if [[ -n "${waiter_pid}" ]] && kill -0 "${waiter_pid}" 2>/dev/null; then
      kill "${waiter_pid}" 2>/dev/null || true
      wait "${waiter_pid}" 2>/dev/null || true
    fi
  done

  if [[ -n "${STATUS_DIR:-}" && -d "${STATUS_DIR}" ]]; then
    rm -rf "${STATUS_DIR}"
  fi
}

terminate_and_exit() {
  local code="$1"
  cleanup
  exit "${code}"
}

trap 'terminate_and_exit 130' INT
trap 'terminate_and_exit 143' TERM
trap 'cleanup' EXIT

log "Starting FastAPI backend on http://127.0.0.1:8067"
(
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}"
  python -m uvicorn explorer.explorer_backend.server:app \
    --host 127.0.0.1 --port 8067 --reload
) &
BACKEND_PID=$!

log "Starting Vite frontend dev server (npm run dev)"
(
  cd "${SCRIPT_DIR}/explorer-frontend"
  npm run dev -- --host 127.0.0.1
) &
FRONTEND_PID=$!

(
  set +e
  if wait "${BACKEND_PID}"; then
    status=0
  else
    status=$?
  fi
  if [[ -d "${STATUS_DIR}" ]]; then
    printf '%s\n' "${status}" >"${STATUS_DIR}/backend"
  fi
) &
BACKEND_WAITER_PID=$!

(
  set +e
  if wait "${FRONTEND_PID}"; then
    status=0
  else
    status=$?
  fi
  if [[ -d "${STATUS_DIR}" ]]; then
    printf '%s\n' "${status}" >"${STATUS_DIR}/frontend"
  fi
) &
FRONTEND_WAITER_PID=$!

FIRST_EXIT=""
EXIT_CODE=0
while [[ -z "${FIRST_EXIT}" ]]; do
  if [[ -f "${STATUS_DIR}/backend" ]]; then
    EXIT_CODE="$(<"${STATUS_DIR}/backend")"
    FIRST_EXIT="backend"
  elif [[ -f "${STATUS_DIR}/frontend" ]]; then
    EXIT_CODE="$(<"${STATUS_DIR}/frontend")"
    FIRST_EXIT="frontend"
  else
    sleep 1
  fi
done

log "Detected ${FIRST_EXIT} exit with status ${EXIT_CODE}"

cleanup

exit "${EXIT_CODE}"
