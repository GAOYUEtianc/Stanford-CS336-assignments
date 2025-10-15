#!/bin/bash

# Script to install and setup Nsight Systems on RunPod or similar GPU environments

echo "=========================================="
echo "Nsight Systems Setup Script"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (or use sudo)"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Check if nsys is already installed
if command -v nsys &> /dev/null; then
    echo "✓ nsys is already installed!"
    echo "  Location: $(which nsys)"
    echo "  Version: $(nsys --version | head -n 1)"
    exit 0
fi

# Try to find nsys in common locations
echo "Searching for nsys in common locations..."
FOUND=false
for path in /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems*/bin/nsys; do
    if [ -f "$path" ]; then
        echo "✓ Found nsys at: $path"
        NSYS_DIR=$(dirname "$path")
        
        # Add to PATH for current session
        export PATH="$NSYS_DIR:$PATH"
        
        # Add to .bashrc for future sessions
        if ! grep -q "$NSYS_DIR" ~/.bashrc; then
            echo "export PATH=\"$NSYS_DIR:\$PATH\"" >> ~/.bashrc
            echo "  Added to ~/.bashrc"
        fi
        
        FOUND=true
        break
    fi
done

if [ "$FOUND" = true ]; then
    echo ""
    echo "✓ nsys is now available!"
    echo "  Run: source ~/.bashrc"
    echo "  Or restart your shell"
    exit 0
fi

# If not found, try to install
echo "nsys not found, attempting to install..."
echo ""

# Update package lists
echo "Updating package lists..."
apt-get update -qq

# Try to install nsight-systems
echo "Installing Nsight Systems..."
if apt-get install -y nsight-systems-cli 2>/dev/null; then
    echo "✓ Successfully installed nsight-systems-cli"
elif apt-get install -y cuda-nsight-systems-12-* 2>/dev/null; then
    echo "✓ Successfully installed cuda-nsight-systems"
else
    echo "✗ Failed to install nsight-systems via apt"
    echo ""
    echo "Manual installation required:"
    echo "1. Check CUDA version: nvcc --version"
    echo "2. Download from: https://developer.nvidia.com/nsight-systems"
    echo "3. Or check your container documentation"
    exit 1
fi

# Verify installation
if command -v nsys &> /dev/null; then
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
    echo "nsys location: $(which nsys)"
    echo "nsys version: $(nsys --version | head -n 1)"
    echo ""
    echo "You can now run profiling with:"
    echo "  nsys profile -o result python your_script.py"
else
    echo ""
    echo "✗ Installation completed but nsys still not found"
    echo "You may need to add it to PATH manually"
    exit 1
fi