#!/usr/bin/env bash
set -euo pipefail

# run_dev.sh - development runner for Upbit Trader
# Usage: ./scripts/run_dev.sh [all|backend|ui|bot] [--detached] [--venv PATH] [--pid-dir PATH]
#                        [--logs-dir PATH] [--no-reload] [--with-redis] [--docker]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Defaults
MODE="all"
DETACHED=false
VENV_PATH=""
PID_DIR="$ROOT_DIR/runtime"
LOG_DIR="$ROOT_DIR/logs"
UVICORN_RELOAD=true
START_REDIS=false
USE_DOCKER=false

# If first arg looks like a mode (not starting with --), take it
if [ $# -ge 1 ] && [[ ! "$1" =~ ^-- ]]; then
  MODE="$1"
  shift
fi

# Parse remaining flags
while [ $# -gt 0 ]; do
  case "$1" in
    --detached)
      DETACHED=true
      shift
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --pid-dir)
      PID_DIR="$2"
      shift 2
      ;;
    --logs-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --no-reload)
      UVICORN_RELOAD=false
      shift
      ;;
    --with-redis)
      START_REDIS=true
      shift
      ;;
    --docker)
      USE_DOCKER=true
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: run_dev.sh [all|backend|ui|bot] [--detached] [--venv PATH] [--pid-dir PATH]
                   [--logs-dir PATH] [--no-reload] [--with-redis] [--docker]

Options:
  --detached        Run processes detached (background or docker -d)
  --venv PATH       Activate virtualenv at PATH before running
  --pid-dir PATH    Directory to write pid files (default: runtime/)
  --logs-dir PATH   Directory to write logs (default: logs/)
  --no-reload       Disable uvicorn --reload when running backend locally
  --with-redis      Start redis service when using --docker (or start a redis container locally)
  --docker          Use docker compose to start selected services instead of local processes
  --help, -h        Show this help

Examples:
  ./scripts/run_dev.sh all --detached               # start all locally in background
  ./scripts/run_dev.sh ui --docker                  # start UI via docker compose
  ./scripts/run_dev.sh backend --no-reload          # run backend locally without reload
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Activate venv if provided
if [ -n "$VENV_PATH" ]; then
  if [ -f "$VENV_PATH/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
    echo "Activated venv: $VENV_PATH"
  else
    echo "Specified venv not found: $VENV_PATH" >&2
  fi
else
  if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/venv/bin/activate"
    echo "Activated venv: $ROOT_DIR/venv"
  fi
fi

# Ensure python imports server package from project
export PYTHONPATH="$ROOT_DIR":${PYTHONPATH-}

write_pid() {
  local pidfile="$1"
  local pid="$2"
  mkdir -p "$(dirname "$pidfile")"
  echo "$pid" > "$pidfile"
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t || true
  else
    # best-effort fallback with netstat
    if command -v netstat >/dev/null 2>&1; then
      netstat -an | grep "\.$port .*LISTEN" || true
    else
      return 1
    fi
  fi
}

start_backend() {
  if ! command -v uvicorn >/dev/null 2>&1; then
    echo "uvicorn not found in PATH. Please install it." >&2
    return 1
  fi

  if [ -f "$ROOT_DIR/server/api.py" ]; then
    ENTRY="server.api:app"
  elif [ -f "$ROOT_DIR/server/app.py" ]; then
    ENTRY="server.app:app"
  elif [ -f "$ROOT_DIR/server/__main__.py" ]; then
    ENTRY="__module__"
  else
    echo "Could not find backend entrypoint." >&2
    return 1
  fi

  BACKEND_LOG="$LOG_DIR/backend.log"

  # avoid starting if port 8000 is used
  if holder=$(port_in_use 8000); then
    if [ -n "$holder" ]; then
      echo "Port 8000 is already in use:" >&2
      echo "$holder" >&2
      return 1
    fi
  fi

  if [ "$ENTRY" = "__module__" ]; then
    if [ "$DETACHED" = true ]; then
      nohup sh -c "cd '$ROOT_DIR' && python -m server" >"$BACKEND_LOG" 2>&1 &
      pid=$!
    else
      (cd "$ROOT_DIR" && python -m server >"$BACKEND_LOG" 2>&1) &
      pid=$!
    fi
  else
    CMD=(uvicorn "$ENTRY" --app-dir "$ROOT_DIR" --host 127.0.0.1 --port 8000)
    if [ "$UVICORN_RELOAD" = true ]; then
      # Restrict reload to server directory to avoid restarts when editing UI files.
      CMD+=(--reload --reload-dir "$ROOT_DIR/server")
    fi
    if [ "$DETACHED" = true ]; then
      nohup "${CMD[@]}" >"$BACKEND_LOG" 2>&1 &
      pid=$!
    else
      "${CMD[@]}" >"$BACKEND_LOG" 2>&1 &
      pid=$!
    fi
  fi

  sleep 0.6
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Backend process exited immediately. See $BACKEND_LOG" >&2
    return 1
  fi

  write_pid "$PID_DIR/backend.pid" "$pid"
  echo "$pid"
}

start_ui() {
  if ! command -v streamlit >/dev/null 2>&1; then
    echo "streamlit not found in PATH. Please install it." >&2
    return 1
  fi

  if [ -f "$ROOT_DIR/ui/ui_dashboard.py" ]; then
    UI_ENTRY="$ROOT_DIR/ui/ui_dashboard.py"
  elif [ -f "$ROOT_DIR/ui/app.py" ]; then
    UI_ENTRY="$ROOT_DIR/ui/app.py"
  else
    echo "Could not find UI entrypoint." >&2
    return 1
  fi

  UI_LOG="$LOG_DIR/ui.log"
  if [ "$DETACHED" = true ]; then
    nohup streamlit run "$UI_ENTRY" >"$UI_LOG" 2>&1 &
    pid=$!
  else
    streamlit run "$UI_ENTRY" >"$UI_LOG" 2>&1 &
    pid=$!
  fi

  sleep 0.6
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "UI process exited immediately. See $UI_LOG" >&2
    return 1
  fi
  write_pid "$PID_DIR/ui.pid" "$pid"
  echo "$pid"
}

start_bot() {
  BOT_LOG="$LOG_DIR/bot.log"
  if [ "$DETACHED" = true ]; then
    nohup sh -c "cd '$ROOT_DIR' && python -m server.bot" >"$BOT_LOG" 2>&1 &
    pid=$!
  else
    (cd "$ROOT_DIR" && python -m server.bot >"$BOT_LOG" 2>&1) &
    pid=$!
  fi
  write_pid "$PID_DIR/bot.pid" "$pid"
  echo "$pid"
}

cleanup() {
  echo "Stopping processes..."
  [ -n "${UI_PID-}" ] && kill "${UI_PID}" 2>/dev/null || true
  [ -n "${BACKEND_PID-}" ] && kill "${BACKEND_PID}" 2>/dev/null || true
  [ -n "${BOT_PID-}" ] && kill "${BOT_PID}" 2>/dev/null || true
}

trap 'cleanup; exit' INT TERM EXIT

# Docker mode: delegate to docker compose
if [ "$USE_DOCKER" = true ]; then
  SERVICES=()
  case "$MODE" in
    all)
      SERVICES=(backend ui bot)
      if [ "$START_REDIS" = true ]; then
        SERVICES+=(redis)
      fi
      ;;
    backend)
      SERVICES=(backend)
      ;;
    ui)
      SERVICES=(ui)
      ;;
    bot)
      SERVICES=(bot)
      ;;
    *)
      echo "Unknown mode for docker: $MODE" >&2
      exit 1
      ;;
  esac

  if [ "$DETACHED" = true ]; then
    echo "Starting services with docker compose (detached): ${SERVICES[*]}" >&2
    docker compose up --build -d "${SERVICES[@]}" || { echo "docker compose up failed" >&2; exit 1; }
    docker compose ps || true
    exit 0
  else
    echo "Starting services with docker compose (foreground): ${SERVICES[*]}" >&2
    docker compose up --build "${SERVICES[@]}"
    exit $?
  fi
fi

wait_for_backend() {
  # Wait up to specified timeout for backend /health to return 200
  local timeout=${1:-30}
  local api_base=${2:-http://127.0.0.1:8000}
  local start_ts
  start_ts=$(date +%s)
  echo "Waiting for backend at $api_base/health (timeout ${timeout}s)" >&2
  while true; do
    if curl -fsS "$api_base/health" >/dev/null 2>&1; then
      echo "Backend is healthy." >&2
      return 0
    fi
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [ $elapsed -ge $timeout ]; then
      echo "Timed out waiting for backend health after ${timeout}s" >&2
      return 1
    fi
    sleep 1
  done
}

# Local-start behavior
case "$MODE" in
  all)
    BACKEND_PID="$(start_backend)" || { echo "Backend start failed. Check $LOG_DIR/backend.log" >&2; exit 1; }
    sleep 1
    UI_PID="$(start_ui)" || { echo "UI start failed. Check $LOG_DIR/ui.log" >&2; kill "$BACKEND_PID" 2>/dev/null || true; exit 1; }
    echo "Backend PID: $BACKEND_PID" >&2
    echo "UI PID: $UI_PID" >&2
    echo "Logs: $LOG_DIR/backend.log , $LOG_DIR/ui.log" >&2
    if [ "$DETACHED" = true ]; then
      exit 0
    fi
    wait
    ;;
  backend)
    BACKEND_PID="$(start_backend)" || { echo "Backend start failed. Check $LOG_DIR/backend.log" >&2; exit 1; }
    echo "Backend PID: $BACKEND_PID" >&2
    if [ "$DETACHED" = true ]; then
      exit 0
    fi
    wait
    ;;
  ui)
    # If backend may be running (docker or local), wait briefly for its health before starting UI
    # Allow override via STREAMLIT_API_BASE env; default to localhost
    BACKEND_HEALTH=${STREAMLIT_API_BASE:-http://127.0.0.1:8000}
    if ! wait_for_backend 20 "$BACKEND_HEALTH"; then
      echo "Warning: backend did not become healthy; starting UI anyway." >&2
    fi
    UI_PID="$(start_ui)" || { echo "UI start failed. Check $LOG_DIR/ui.log" >&2; exit 1; }
    echo "UI PID: $UI_PID" >&2
    if [ "$DETACHED" = true ]; then
      exit 0
    fi
    wait
    ;;
  bot)
    BOT_PID="$(start_bot)" || { echo "Bot start failed. Check $LOG_DIR/bot.log" >&2; exit 1; }
    echo "Bot PID: $BOT_PID" >&2
    if [ "$DETACHED" = true ]; then
      exit 0
    fi
    wait
    ;;
  *)
    echo "Usage: $0 [all|backend|ui|bot] [--detached] [--venv PATH] [--pid-dir PATH] [--logs-dir PATH] [--no-reload] [--docker]" >&2
    exit 1
    ;;
esac
