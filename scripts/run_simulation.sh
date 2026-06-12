#!/usr/bin/env bash
# =============================================================================
# Semáforo Inteligente — Run Simulation
# =============================================================================
# Launches a full simulation environment:
#   1. Starts the Mosquitto MQTT broker (via Docker)
#   2. Starts the Green Wave Coordinator
#   3. Runs the RL training script
#
# Prerequisites:
#   - Docker installed and running
#   - Python venv set up (run setup_dev.sh first)
#   - SUMO installed and SUMO_HOME set
#
# Usage:
#   chmod +x scripts/run_simulation.sh
#   ./scripts/run_simulation.sh
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo -e "${BLUE}🚦 Semáforo Inteligente — Simulation Runner${NC}"
echo "=============================================="
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"

    # Stop coordinator if running
    if [ -n "${COORDINATOR_PID:-}" ]; then
        kill "$COORDINATOR_PID" 2>/dev/null || true
        echo -e "  ${GREEN}✓ Coordinator stopped${NC}"
    fi

    # Stop Mosquitto container
    cd "$PROJECT_ROOT"
    docker compose stop mosquitto 2>/dev/null || true
    echo -e "  ${GREEN}✓ Mosquitto stopped${NC}"

    echo -e "${GREEN}All services stopped.${NC}"
}

trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# 1. Activate virtual environment
# -----------------------------------------------------------------------------
echo -e "${BLUE}[1/3] Activating virtual environment...${NC}"

VENV_DIR="${PROJECT_ROOT}/.venv"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo -e "  ${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "  ${RED}✗ Virtual environment not found at ${VENV_DIR}${NC}"
    echo "  Run './scripts/setup_dev.sh' first."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Launch Mosquitto broker
# -----------------------------------------------------------------------------
echo -e "${BLUE}[2/3] Starting Mosquitto MQTT broker...${NC}"

cd "$PROJECT_ROOT"
docker compose up -d mosquitto

# Wait for broker to be ready
echo -n "  Waiting for Mosquitto to be ready"
for i in $(seq 1 15); do
    if docker compose exec -T mosquitto mosquitto_sub -t '$SYS/#' -C 1 -W 2 &>/dev/null; then
        echo ""
        echo -e "  ${GREEN}✓ Mosquitto is ready (port 1883)${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# -----------------------------------------------------------------------------
# 3. Launch Coordinator
# -----------------------------------------------------------------------------
echo -e "${BLUE}[3/3] Starting Green Wave Coordinator...${NC}"

cd "${PROJECT_ROOT}/network/coordinator"
python -m src.coordinator &
COORDINATOR_PID=$!
echo -e "  ${GREEN}✓ Coordinator started (PID: ${COORDINATOR_PID})${NC}"

# Give coordinator time to connect
sleep 2

# -----------------------------------------------------------------------------
# 4. Run Training (if brain module exists)
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}Starting RL training...${NC}"
echo "=============================================="

if [ -f "${PROJECT_ROOT}/brain/pyproject.toml" ]; then
    cd "${PROJECT_ROOT}/brain"
    python src/training/run_green_wave.py 2>&1 || {
        echo -e "${YELLOW}⚠ Training script failed.${NC}"
        echo "  The coordinator is running. You can publish test messages with:"
        echo "    mosquitto_pub -t 'semaforo/int-001/telemetry' -m '{\"phase\":\"NS_GREEN\",\"vehicle_count\":10}'"
        echo ""
        echo "Press Ctrl+C to stop all services."
        wait "$COORDINATOR_PID"
    }
else
    echo -e "${YELLOW}⚠ Brain module not found. Running coordinator only.${NC}"
    echo ""
    echo "The coordinator is running and listening for telemetry."
    echo "Publish test messages with:"
    echo "  mosquitto_pub -t 'semaforo/int-001/telemetry' -m '{\"phase\":\"NS_GREEN\",\"vehicle_count\":10}'"
    echo ""
    echo "Press Ctrl+C to stop all services."
    wait "$COORDINATOR_PID"
fi
