#!/bin/bash
# ============================================================
# CRASH DETECTION - ORBSTACK LINUX SETUP SCRIPT
# ============================================================
# Run this on your OrbStack Linux machine
# Usage: bash setup_orbstack.sh

set -e

echo "========================================"
echo "CRASH DETECTION - ORBSTACK SETUP"
echo "========================================"

# ============================================================
# Check Python version
# ============================================================
echo ""
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Found: $(python3 --version)"

if [[ ! "$PYTHON_VERSION" == "3.10"* ]]; then
    echo "ERROR: Python 3.10 is required but found $PYTHON_VERSION"
    echo "Please install Python 3.10:"
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "  sudo apt update"
    echo "  sudo apt install python3.10 python3.10-venv"
    exit 1
fi

echo "✅ Python 3.10 detected"

# ============================================================
# Create virtual environment
# ============================================================
echo ""
echo "Step 2: Creating virtual environment..."

if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "✅ Created virtual environment"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Activated virtual environment"

# ============================================================
# Install dependencies
# ============================================================
echo ""
echo "Step 3: Installing dependencies..."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# TensorFlow (2.15.x - latest stable)
pip install tensorflow==2.15.0

# NumPy and SciPy
pip install numpy==1.24.3 scipy==1.11.4

# OpenCV
pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78

# Ultralytics (YOLO)
pip install ultralytics==8.0.196

# PyTorch (for YOLO)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Other utilities
pip install pandas matplotlib

echo "✅ All dependencies installed"

# ============================================================
# Verify installation
# ============================================================
echo ""
echo "Step 4: Verifying installation..."

echo -n "TensorFlow: "
python3 -c "import tensorflow as tf; print(tf.__version__)"

echo -n "OpenCV: "
python3 -c "import cv2; print(cv2.__version__)"

echo -n "Ultralytics: "
python3 -c "import ultralytics; print(ultralytics.__version__)"

echo -n "PyTorch: "
python3 -c "import torch; print(torch.__version__)"

echo -n "NumPy: "
python3 -c "import numpy; print(numpy.__version__)"

echo -n "SciPy: "
python3 -c "import scipy; print(scipy.__version__)"

echo ""
echo "✅ All packages verified!"

# ============================================================
# Run self-test
# ============================================================
echo ""
echo "Step 5: Running self-test..."

python3 code/crash_detection_linux.py --test

echo ""
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo ""
echo "To run crash detection:"
echo "  source venv/bin/activate"
echo "  python3 code/crash_detection_linux.py --video safe --no-display"
echo ""