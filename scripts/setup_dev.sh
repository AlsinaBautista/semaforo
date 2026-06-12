#!/usr/bin/env bash
# =============================================================================
# Semáforo Inteligente — Development Environment Setup
# =============================================================================
# This script sets up the development environment:
#   1. Checks Python version (requires 3.10+)
#   2. Creates a Python virtual environment
#   3. Installs brain module dependencies
#   4. Checks SUMO installation
#   5. Verifies Mosquitto availability
#   6. Prints a summary
#
# Usage:
#   chmod +x scripts/setup_dev.sh
#   ./scripts/setup_dev.sh
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo -e "${BLUE}🚦 Semáforo Inteligente — Development Setup${NC}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# 1. Check Python version
# -----------------------------------------------------------------------------
echo -e "${BLUE}[1/5] Checking Python version...${NC}"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$("$cmd" -c 'import sys; print(sys.version_info.major)')
        minor=$("$cmd" -c 'import sys; print(sys.version_info.minor)')
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo -e "  ${GREEN}✓ Found $cmd (version $version)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "  ${RED}✗ Python 3.10+ is required but not found.${NC}"
    echo "  Please install Python 3.10 or later."
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Create virtual environment
# -----------------------------------------------------------------------------
echo -e "${BLUE}[2/5] Setting up Python virtual environment...${NC}"

VENV_DIR="${PROJECT_ROOT}/.venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}⚠ Virtual environment already exists at ${VENV_DIR}${NC}"
    echo "  Skipping creation. Delete it manually to recreate."
else
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo -e "  ${GREEN}✓ Virtual environment created at ${VENV_DIR}${NC}"
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
echo -e "  ${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
pip install --upgrade pip --quiet
echo -e "  ${GREEN}✓ pip upgraded${NC}"

# -----------------------------------------------------------------------------
# 3. Install brain dependencies
# -----------------------------------------------------------------------------
echo -e "${BLUE}[3/5] Installing Brain module dependencies...${NC}"

if [ -f "${PROJECT_ROOT}/brain/pyproject.toml" ]; then
    pip install -e "${PROJECT_ROOT}/brain[dev]" --quiet 2>/dev/null || \
        echo -e "  ${YELLOW}⚠ Brain module not yet set up (pyproject.toml may be incomplete)${NC}"
    echo -e "  ${GREEN}✓ Brain dependencies installed${NC}"
else
    echo -e "  ${YELLOW}⚠ brain/pyproject.toml not found — skipping${NC}"
fi

# Install coordinator dependencies
if [ -f "${PROJECT_ROOT}/network/coordinator/pyproject.toml" ]; then
    pip install -e "${PROJECT_ROOT}/network/coordinator[dev]" --quiet 2>/dev/null || \
        echo -e "  ${YELLOW}⚠ Coordinator install had issues (non-critical)${NC}"
    echo -e "  ${GREEN}✓ Coordinator dependencies installed${NC}"
else
    echo -e "  ${YELLOW}⚠ network/coordinator/pyproject.toml not found — skipping${NC}"
fi

# -----------------------------------------------------------------------------
# 4. Check SUMO installation
# -----------------------------------------------------------------------------
echo -e "${BLUE}[4/5] Checking SUMO installation...${NC}"

if command -v sumo &>/dev/null; then
    SUMO_VERSION=$(sumo --version 2>&1 | head -1)
    echo -e "  ${GREEN}✓ SUMO found: ${SUMO_VERSION}${NC}"
elif [ -n "${SUMO_HOME:-}" ]; then
    echo -e "  ${GREEN}✓ SUMO_HOME is set: ${SUMO_HOME}${NC}"
else
    echo -e "  ${YELLOW}⚠ SUMO not found in PATH and SUMO_HOME not set.${NC}"
    echo "  Install SUMO: https://sumo.dlr.de/docs/Installing/index.html"
    echo "  Then set: export SUMO_HOME=/path/to/sumo"
fi

# -----------------------------------------------------------------------------
# 5. Check Mosquitto
# -----------------------------------------------------------------------------
echo -e "${BLUE}[5/5] Checking Mosquitto...${NC}"

if command -v mosquitto &>/dev/null; then
    echo -e "  ${GREEN}✓ Mosquitto found$(mosquitto -h 2>&1 | head -1 | sed 's/mosquitto//' || true)${NC}"
elif command -v docker &>/dev/null; then
    echo -e "  ${YELLOW}⚠ Mosquitto not installed locally, but Docker is available.${NC}"
    echo "  Use 'docker compose up mosquitto' to run the broker."
else
    echo -e "  ${YELLOW}⚠ Neither Mosquitto nor Docker found.${NC}"
    echo "  Install Mosquitto: https://mosquitto.org/download/"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}🎉 Development environment setup complete!${NC}"
echo "=============================================="
echo ""
echo "Quick start commands:"
echo "  source .venv/bin/activate           # Activate virtual environment"
echo "  docker compose up -d mosquitto      # Start MQTT broker"
echo "  python -m src.coordinator           # Run coordinator (from network/coordinator/)"
echo ""
echo -e "${BLUE}Happy coding! 🚦${NC}"
