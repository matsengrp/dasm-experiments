#!/bin/bash
"""
Setup ProGen2 analysis scripts on ermine server.

This script copies the necessary analysis scripts to the ProGen2 directory
after the environment has been set up with setup_progen_env.sh.

Run this script on ermine AFTER running setup_progen_env.sh:
  bash ~/re/dnsm-experiments-1/scripts/setup_progen_scripts.sh
"""

set -e  # Exit on any error

echo "======================================="
echo "ProGen2 Analysis Scripts Setup"
echo "======================================="

# Configuration
PROGEN_DIR="$HOME/progen2"
DNSM_DIR="$HOME/re/dnsm-experiments-1"

echo "Paths:"
echo "  ProGen2 directory: $PROGEN_DIR"
echo "  DNSM experiments: $DNSM_DIR"
echo ""

# Check if ProGen2 environment exists
if [ ! -d "$PROGEN_DIR" ]; then
    echo "❌ ProGen2 directory not found at $PROGEN_DIR"
    echo "Please run setup_progen_env.sh first"
    exit 1
fi

if [ ! -d "$PROGEN_DIR/.venv" ]; then
    echo "❌ ProGen2 virtual environment not found"
    echo "Please run setup_progen_env.sh first"
    exit 1
fi

# Copy the flab_progen.py script to ProGen2 directory
echo "📄 Copying flab_progen.py to ProGen2 directory..."
cp "$DNSM_DIR/scripts/flab_progen.py" "$PROGEN_DIR/"

# Verify the script works
echo ""
echo "🧪 Testing flab_progen.py script..."
cd "$PROGEN_DIR"
source .venv/bin/activate

python flab_progen.py --help

echo ""
echo "✅ ProGen2 analysis scripts setup complete!"
echo ""
echo "📝 Files in ProGen2 directory:"
ls -la "$PROGEN_DIR/flab_progen.py"
echo ""
echo "🚀 Ready to run ProGen2 analysis via remote wrapper!"