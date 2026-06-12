#!/bin/bash
# ============================================================================
# generate_network.sh
# Generates the SUMO network (.net.xml) from node and edge definitions.
# Semáforo Inteligente Project
#
# Usage:
#   cd brain/sumo_networks/single_intersection/
#   ./generate_network.sh
#
# Prerequisites:
#   - SUMO must be installed and netconvert must be on PATH
#   - Set SUMO_HOME environment variable if not already set
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " SUMO Network Generator"
echo " Semáforo Inteligente - Single Intersection"
echo "============================================"
echo ""

# Check that netconvert is available
if ! command -v netconvert &> /dev/null; then
    echo "ERROR: netconvert not found. Please install SUMO and ensure it is on your PATH."
    echo "  macOS:   brew install sumo"
    echo "  Ubuntu:  sudo apt-get install sumo sumo-tools"
    echo "  Or set SUMO_HOME and add \$SUMO_HOME/bin to PATH."
    exit 1
fi

echo "Using netconvert: $(command -v netconvert)"
echo "SUMO version: $(netconvert --version 2>&1 | head -1)"
echo ""

# Generate network
echo "Generating intersection.net.xml ..."
netconvert \
    --node-files=intersection.nod.xml \
    --edge-files=intersection.edg.xml \
    --output-file=intersection.net.xml \
    --tls.guess true

echo ""
echo "✅ Network generated successfully: intersection.net.xml"
echo ""
echo "To run the simulation:"
echo "  sumo -c intersection.sumocfg"
echo "  sumo-gui -c intersection.sumocfg   (with GUI)"
