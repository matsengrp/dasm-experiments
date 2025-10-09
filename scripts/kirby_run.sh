#!/bin/bash
# Kirby Analysis - Primary run script
# This is the authoritative script for reproducing the complete Kirby analysis pipeline

set -e  # Exit on any error

echo "Starting Kirby analysis pipeline..."

# Activate the netam environment
source ~/re/netam/.venv/bin/activate

# Run UCA baseline partitioning with optimized parameters
echo "Creating UCA baseline partitions..."
python scripts/kirby_uca_baseline.py --uca-only --similarity-threshold 0.90

echo "✅ Kirby analysis pipeline complete!"
echo "Results saved to: DATA_DIR/whitehead/kirby/uca_baseline_partitions/"