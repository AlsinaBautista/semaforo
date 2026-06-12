#!/usr/bin/env bash
# =============================================================================
# Semáforo Inteligente — Edge Deployment (Jetson)
# =============================================================================
# Placeholder script for deploying the edge detection module to
# NVIDIA Jetson devices.
#
# This script will handle:
#   1. SSH connection to the Jetson device
#   2. Syncing the edge source code
#   3. Building the C++ project on the Jetson
#   4. Converting ONNX models to TensorRT engines
#   5. Installing and starting the systemd service
#
# Prerequisites:
#   - NVIDIA Jetson with JetPack ≥ 6.0
#   - SSH access to the Jetson device
#   - ONNX model files in edge/models/
#
# Usage:
#   chmod +x scripts/deploy_edge.sh
#   ./scripts/deploy_edge.sh <jetson-ip> [--build-only]
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

echo -e "${BLUE}🚦 Semáforo Inteligente — Edge Deployment${NC}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
JETSON_IP="${1:-}"
JETSON_USER="${JETSON_USER:-semaforo}"
JETSON_DEPLOY_DIR="/opt/semaforo/edge"
BUILD_ONLY="${2:-}"

if [ -z "$JETSON_IP" ]; then
    echo -e "${RED}Usage: $0 <jetson-ip> [--build-only]${NC}"
    echo ""
    echo "Environment variables:"
    echo "  JETSON_USER     SSH user (default: semaforo)"
    echo ""
    echo "Examples:"
    echo "  $0 192.168.1.100"
    echo "  $0 192.168.1.100 --build-only"
    echo "  JETSON_USER=nvidia $0 jetson.local"
    exit 1
fi

# -----------------------------------------------------------------------------
# TODO: Implement deployment steps
# -----------------------------------------------------------------------------

echo -e "${YELLOW}⚠ This script is a placeholder and not yet implemented.${NC}"
echo ""
echo "Planned deployment steps:"
echo "  1. Verify SSH connectivity to ${JETSON_USER}@${JETSON_IP}"
echo "  2. Sync edge/ directory to ${JETSON_DEPLOY_DIR}"
echo "  3. Build C++ project on Jetson (cmake + make)"
echo "  4. Convert ONNX models to TensorRT engines"
echo "  5. Install systemd service unit"
echo "  6. Start/restart the edge service"
echo "  7. Verify MQTT connectivity"
echo ""
echo "For now, deploy manually:"
echo "  scp -r edge/ ${JETSON_USER}@${JETSON_IP}:${JETSON_DEPLOY_DIR}"
echo "  ssh ${JETSON_USER}@${JETSON_IP} 'cd ${JETSON_DEPLOY_DIR} && mkdir -p build && cd build && cmake .. && make -j\$(nproc)'"
echo ""
echo -e "${BLUE}See docs/deployment-guide.md for more details.${NC}"
