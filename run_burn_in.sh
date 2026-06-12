#!/usr/bin/env bash
# Chaos Engineering / Burn-In harness for the Edge node.
# Normal run  : ./run_burn_in.sh
# Dry-run     : DRY_RUN=1 ./run_burn_in.sh
set -uo pipefail

# ── Tunables (override via env) ───────────────────────────────────────────────
EDGE_BIN="${EDGE_BIN:-./edge_node}"
CHAOS_SCRIPT="${CHAOS_SCRIPT:-tests/chaos_monkey_mqtt.py}"
CSV_FILE="${CSV_FILE:-telemetry_ram.csv}"
INTERVAL="${INTERVAL:-60}"        # seconds between RSS samples
DURATION="${DURATION:-28800}"     # 8 h in seconds

EDGE_PID=""
CHAOS_PID=""

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ── Cleanup / trap ────────────────────────────────────────────────────────────
cleanup() {
    # Unregister first to avoid recursive re-entry via EXIT
    trap - SIGINT SIGTERM EXIT

    echo ""
    echo "[$(ts)] INFO: Shutting down — sending SIGTERM to child processes."

    for pid_var in EDGE_PID CHAOS_PID; do
        local pid="${!pid_var}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done

    # Grace period, then force-kill any survivors
    sleep 2

    for pid_var in EDGE_PID CHAOS_PID; do
        local pid="${!pid_var}"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "[$(ts)] WARN: PID $pid still alive — sending SIGKILL."
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    echo "[$(ts)] INFO: Harness exited cleanly. CSV: $CSV_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ── Dry-run substitution ──────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" == "1" ]]; then
    INTERVAL=5
    DURATION=10
    echo "[$(ts)] INFO: *** DRY-RUN MODE *** stubs=sleep, interval=${INTERVAL}s, duration=${DURATION}s"
fi

# ── Preflight checks (skipped in dry-run) ────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -x "$EDGE_BIN" ]]; then
        echo "[$(ts)] ERROR: Edge binary not found or not executable: $EDGE_BIN" >&2
        exit 1
    fi
    if [[ ! -f "$CHAOS_SCRIPT" ]]; then
        echo "[$(ts)] ERROR: Chaos script not found: $CHAOS_SCRIPT" >&2
        exit 1
    fi
fi

# ── Launch Edge node ──────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(ts)] INFO: Launching Edge stub (sleep 300)"
    sleep 300 &
else
    echo "[$(ts)] INFO: Launching Edge node: $EDGE_BIN"
    "$EDGE_BIN" &
fi
EDGE_PID=$!
echo "[$(ts)] INFO: Edge node PID = $EDGE_PID"

# ── Launch Chaos monkey ───────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(ts)] INFO: Launching Chaos stub (sleep 300)"
    sleep 300 &
else
    echo "[$(ts)] INFO: Launching Chaos monkey: python3 $CHAOS_SCRIPT"
    python3 "$CHAOS_SCRIPT" &
fi
CHAOS_PID=$!
echo "[$(ts)] INFO: Chaos monkey PID = $CHAOS_PID"

# ── CSV header ────────────────────────────────────────────────────────────────
echo "timestamp_utc,rss_kb,edge_pid" > "$CSV_FILE"
echo "[$(ts)] INFO: Writing samples to $CSV_FILE"

# ── Monitor loop ──────────────────────────────────────────────────────────────
START_TS=$(date +%s)
END_TS=$(( START_TS + DURATION ))
SAMPLE=0

echo "[$(ts)] INFO: Monitor started — ${DURATION}s total, sampling every ${INTERVAL}s."

while true; do
    NOW=$(date +%s)
    if (( NOW >= END_TS )); then
        echo "[$(ts)] INFO: Duration elapsed — initiating normal shutdown."
        break
    fi

    # Guard: abort if the Edge node died unexpectedly
    if ! kill -0 "$EDGE_PID" 2>/dev/null; then
        echo "[$(ts)] ERROR: Edge node (PID $EDGE_PID) died unexpectedly." >&2
        exit 1
    fi

    # Sample RSS (kB) — empty string if process vanished between the two checks
    RSS=$(ps -p "$EDGE_PID" -o rss= 2>/dev/null | tr -d '[:space:]')
    RSS="${RSS:-N/A}"
    ROW="$(ts),${RSS},${EDGE_PID}"
    echo "$ROW" | tee -a "$CSV_FILE"
    SAMPLE=$(( SAMPLE + 1 ))

    sleep "$INTERVAL"
done

echo "[$(ts)] INFO: Burn-in complete — $SAMPLE sample(s) written to $CSV_FILE."
# EXIT trap fires here → cleanup()
