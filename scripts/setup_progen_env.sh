#!/bin/bash
"""
Setup script for ProGen2 environment on ermine server.

This script creates an isolated Python environment for ProGen2 with the correct
dependencies and sets up the ProGen2 repository with the required patches.

Run this script on ermine in the home directory:
  bash ~/re/dnsm-experiments-1/scripts/setup_progen_env.sh

Requirements:
- Git access to clone repositories
- Python 3.8+ available
- CUDA-capable GPU for inference
"""

set -e  # Exit on any error

echo "============================================="
echo "ProGen2 Environment Setup for Ermine Server"
echo "============================================="

# Configuration
PROGEN_DIR="$HOME/progen2"
VENV_DIR="$PROGEN_DIR/.venv"
REPO_URL="https://github.com/matsen/progen2"
BRANCH="patch-1"

echo "Setup paths:"
echo "  ProGen2 directory: $PROGEN_DIR"
echo "  Virtual environment: $VENV_DIR"
echo "  Repository: $REPO_URL (branch: $BRANCH)"
echo ""

# Clean up any existing installation
if [ -d "$PROGEN_DIR" ]; then
    echo "🧹 Removing existing ProGen2 directory..."
    rm -rf "$PROGEN_DIR"
fi

# Clone the patched ProGen2 repository
echo "📦 Cloning ProGen2 repository with patches..."
git clone -b "$BRANCH" "$REPO_URL" "$PROGEN_DIR"
cd "$PROGEN_DIR"

echo "✅ ProGen2 repository cloned successfully"
echo "📍 Current directory: $(pwd)"
echo "📂 Repository contents:"
ls -la

# Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "✅ Virtual environment created and activated"
echo "🐍 Python version: $(python --version)"
echo "📍 Python location: $(which python)"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (compatible with ProGen2)
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required dependencies
echo ""
echo "📦 Installing ProGen2 dependencies..."
# Force binary wheels to avoid Rust compilation
pip install --only-binary=all transformers fire pandas tqdm

# If that fails, try installing specific compatible versions
if [ $? -ne 0 ]; then
    echo "⚠️  Trying fallback installation with specific versions..."
    pip install --only-binary=tokenizers tokenizers==0.13.3
    pip install transformers==4.28.0  # Newer version that should have wheels
    pip install fire pandas tqdm
fi

# Verify CUDA availability
echo ""
echo "🔍 Testing CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available - ProGen2 will run on CPU (very slow)')
"

# Download ProGen2 model checkpoints
echo ""
echo "📥 Downloading ProGen2 model checkpoints..."
mkdir -p checkpoints

# Download small model (default for testing) - using wget with proper error handling
echo "  Downloading progen2-small..."
cd checkpoints

# Try the direct download link from the ProGen2 repo README
echo "  Attempting download from Google Storage..."
wget --no-check-certificate -O progen2-small.tar.gz \
    "https://storage.googleapis.com/sfr-progen-research/checkpoints/progen2-small.tar.gz" || {
    echo "  Direct download failed, checking file content..."
    head -20 progen2-small.tar.gz
    echo ""
    echo "  Trying alternative download method..."
    
    # Alternative: Clone just the model files we need
    echo "  Downloading individual model files..."
    mkdir -p progen2-small
    cd progen2-small
    
    # Download the model files individually based on ProGen2 structure
    wget --no-check-certificate -O pytorch_model.bin \
        "https://storage.googleapis.com/sfr-progen-research/models/progen2-small/pytorch_model.bin"
    wget --no-check-certificate -O config.json \
        "https://storage.googleapis.com/sfr-progen-research/models/progen2-small/config.json"
    
    cd ..
    rm -f progen2-small.tar.gz
}

# Check if we got the files
if [ -f "progen2-small/pytorch_model.bin" ] || [ -f "progen2-small.tar.gz" ]; then
    # If we have a tar.gz, extract it
    if [ -f "progen2-small.tar.gz" ]; then
        echo "  Extracting model files..."
        # The tar file contains just the model files, not a directory
        # So we need to extract into the progen2-small directory
        mkdir -p progen2-small
        tar -xzf progen2-small.tar.gz -C progen2-small
        rm progen2-small.tar.gz
    fi
else
    echo "❌ Failed to download ProGen2 model"
    exit 1
fi

cd ..

# Download tokenizer from the correct URL
echo "  Downloading tokenizer..."
wget --no-check-certificate -O tokenizer.json \
    "https://raw.githubusercontent.com/enijkamp/progen2/main/tokenizer.json" || \
curl -L -o tokenizer.json \
    "https://storage.googleapis.com/sfr-progen-research/checkpoints/tokenizer.json"

echo "📂 Verifying downloads..."
if [ -f "checkpoints/progen2-small/pytorch_model.bin" ]; then
    echo "✅ ProGen2 model files verified"
else
    echo "❌ Model files not found - checking what was downloaded..."
    ls -la checkpoints/progen2-small/
fi

echo "✅ ProGen2 checkpoints downloaded successfully"

# Test the installation
echo ""
echo "🧪 Testing ProGen2 installation..."
cat > test_progen.py << 'EOF'
"""Quick test of ProGen2 installation."""
import torch
import sys
import os

# Add current directory to path for ProGen2 imports
sys.path.insert(0, '.')

try:
    from likelihood import create_model, create_tokenizer_custom
    print("✅ ProGen2 imports successful")
    
    # Test model loading
    print("🔄 Testing model loading...")
    model = create_model(ckpt="checkpoints/progen2-small", fp16=True)
    print("✅ Model loading successful")
    
    # Test tokenizer
    print("🔄 Testing tokenizer...")
    tokenizer = create_tokenizer_custom(file="tokenizer.json")
    print("✅ Tokenizer loading successful")
    
    print("")
    print("🎉 ProGen2 installation test PASSED!")
    print("Ready for antibody sequence evaluation.")
    
except Exception as e:
    print(f"❌ ProGen2 installation test FAILED: {e}")
    sys.exit(1)
EOF

python test_progen.py
rm test_progen.py

# Create a simple test script for the remote interface
echo ""
echo "📝 Creating test script for remote interface..."
cat > test_remote_interface.py << 'EOF'
"""Test script for remote ProGen2 interface."""
import pandas as pd
import sys
sys.path.insert(0, '.')

# Import our flab_progen script
from flab_progen import progen_score_cli

# Create test data
test_data = pd.DataFrame({
    'heavy': ['QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCAR'],
    'light': ['DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK']
})

print("Creating test input file...")
test_data.to_csv('test_input.csv', index=False)

try:
    print("Testing ProGen2 scoring...")
    progen_score_cli(
        input_file='test_input.csv',
        model='progen2-small',
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        dry_run=True,
        output_dir='test_output'
    )
    print("✅ Remote interface test PASSED!")
    
    # Clean up
    import os
    os.remove('test_input.csv')
    if os.path.exists('test_output'):
        import shutil
        shutil.rmtree('test_output')
        
except Exception as e:
    print(f"❌ Remote interface test FAILED: {e}")
    
    # Clean up on failure
    if os.path.exists('test_input.csv'):
        os.remove('test_input.csv')
    if os.path.exists('test_output'):
        import shutil
        shutil.rmtree('test_output')
EOF

python test_remote_interface.py
rm test_remote_interface.py

# Create activation script for easy sourcing
echo ""
echo "📄 Creating activation helper script..."
cat > activate_progen.sh << 'EOF'
#!/bin/bash
# Helper script to activate ProGen2 environment
# Usage: source ~/progen2/activate_progen.sh

echo "🔧 Activating ProGen2 environment..."
source ~/.venv/bin/activate
cd ~/progen2
echo "✅ ProGen2 environment activated"
echo "📍 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
EOF

chmod +x activate_progen.sh

# Final status
echo ""
echo "============================================="
echo "✅ ProGen2 Environment Setup Complete!"
echo "============================================="
echo ""
echo "📋 Setup Summary:"
echo "  • ProGen2 repository: $PROGEN_DIR"
echo "  • Virtual environment: $VENV_DIR" 
echo "  • Model checkpoints: $PROGEN_DIR/checkpoints/"
echo "  • Tokenizer: $PROGEN_DIR/tokenizer.json"
echo ""
echo "🚀 Quick Start:"
echo "  cd $PROGEN_DIR"
echo "  source .venv/bin/activate"
echo "  python flab_progen.py --help"
echo ""
echo "🔗 Integration:"
echo "  The remote ProGen wrapper will automatically use:"
echo "    - Host: ermine"
echo "    - Directory: $PROGEN_DIR"
echo "    - Virtual env: $VENV_DIR/bin/activate"
echo ""
echo "📝 Next Steps:"
echo "  1. Test the setup by running the Kirby analysis script"
echo "  2. The model scoring will automatically use this environment"
echo "  3. Results will be cached for faster subsequent runs"
echo ""
echo "🎉 Ready for antibody sequence evaluation!"