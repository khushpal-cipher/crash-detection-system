# ============================================================
# ORBSTACK LINUX CRASH DETECTION SETUP
# ============================================================
# Exact environment to match Google Colab:
#   Python: 3.10.x
#   TensorFlow: 2.13.x (or 2.15.x latest stable)
#   OpenCV: 4.8.x
#   Ultralytics: 8.0.x
#   PyTorch: 2.0.x (for YOLO)
#   SciPy: 1.11.x

# ============================================================
# STEP 1: SSH INTO ORBSTACK LINUX
# ============================================================
# Get your OrbStack IP from the OrbStack app
# Then connect via SSH:
ssh user@your-orbstack-ip
# (usually like: ssh user@192.168.x.x or ssh user@orb.local)

# ============================================================
# STEP 2: CREATE PROJECT DIRECTORY
# ============================================================

# Create project directory (adjust path as needed)
mkdir -p ~/crash_detection/{code,models,videos}
cd ~/crash_detection

# ============================================================
# STEP 3: INSTALL PYTHON 3.10 (REQUIRED)
# ============================================================

# Ubuntu/Debian:
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment with Python 3.10
python3.10 -m venv venv
source venv/bin/activate

# ============================================================
# STEP 4: INSTALL EXACT DEPENDENCIES (MATCH COLAB)
# ============================================================

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install TensorFlow 2.15.x (latest stable, compatible)
pip install tensorflow==2.15.0

# Install other dependencies
pip install numpy==1.24.3
pip install scipy==1.11.4

# Install OpenCV 4.8.x (for video processing)
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python==4.8.1.78

# Install Ultralytics (YOLOv8)
pip install ultralytics==8.0.196

# Install PyTorch 2.0.x (for YOLO)
pip install torch==2.0.1 torchvision==0.15.2

# Install other utilities
pip install pandas
pip install matplotlib

# ============================================================
# STEP 5: VERIFY INSTALLATION
# ============================================================

python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')"

# ============================================================
# STEP 6: COPY FILES FROM MAC TO LINUX
# ============================================================

# OPTION A: Using scp (from Mac terminal - NOT from within OrbStack)
# Copy code
scp ~/Desktop/crash_detection/code/crash_detection_linux.py user@orb-ip:~/crash_detection/code/
scp ~/Desktop/crash_detection/code/yolov8n.pt user@orb-ip:~/crash_detection/code/

# Copy model
scp ~/Desktop/crash_detection/models/crash_detection_model.keras user@orb-ip:~/crash_detection/models/

# Copy videos
scp ~/Desktop/crash_detection/videos/*.mp4 user@orb-ip:~/crash_detection/videos/
scp ~/Desktop/crash_detection/videos/*.mov user@orb-ip:~/crash_detection/videos/

# ============================================================
# STEP 7: VERIFY FILES ON LINUX
# ============================================================

cd ~/crash_detection
ls -la code/
ls -la models/
ls -la videos/

# ============================================================
# STEP 8: RUN TEST
# ============================================================

# Activate virtual environment
source venv/bin/activate

# Run self-test
python3 code/crash_detection_linux.py --test

# Run on safe video (should show NO CRASH)
python3 code/crash_detection_linux.py --video safe --no-display

# Run on crash video (should show CRASH DETECTED)
python3 code/crash_detection_linux.py --video crash1 --no-display

# ============================================================
# FINAL COMMAND (FROM LINUX)
# ============================================================

# cd to project directory
cd ~/crash_detection

# Activate virtual environment
source venv/bin/activate

# Process any video
python3 code/crash_detection_linux.py --video crash1 --no-display

# Or with display
python3 code/crash_detection_linux.py --video crash1

# ============================================================
# TROUBLESHOOTING
# ============================================================

# If YOLO downloads slow, you can copy yolov8n.pt from Mac
# The file is at ~/Desktop/crash_detection/yolov8n.pt (~6MB)

# If model prediction fails, it's OK - collision detection works without it
# The .keras model should work on Linux better than Mac

# For any issues, check Python version must be exactly 3.10.x
python3 --version