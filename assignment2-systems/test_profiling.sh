#!/bin/bash

# Quick test script to verify profiling setup works

echo "=========================================="
echo "Profiling Setup Test"
echo "=========================================="
echo ""

# 1. Check GPU
echo "1. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "✗ nvidia-smi not found"
    exit 1
fi
echo ""

# 2. Check nsys
echo "2. Checking nsys installation..."
if command -v nsys &> /dev/null; then
    echo "✓ nsys found at: $(which nsys)"
    nsys --version | head -n 1
else
    echo "✗ nsys not found"
    echo "  Run: ./setup_nsys.sh"
    exit 1
fi
echo ""

# 3. Check Python and PyTorch
echo "3. Checking Python and PyTorch..."
python3 << 'EOF'
import sys
import torch

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("✓ PyTorch CUDA setup OK")
else:
    print("✗ CUDA not available in PyTorch")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "✗ PyTorch setup failed"
    exit 1
fi
echo ""

# 4. Quick profiling test
echo "4. Running quick profiling test..."
nsys profile -o test_profile --trace=cuda,nvtx --force-overwrite=true \
    python3 -c "import torch; x = torch.randn(100, 100, device='cuda'); y = x @ x; torch.cuda.synchronize(); print('Test passed!')"

if [ $? -eq 0 ]; then
    echo "✓ Profiling test successful!"
    echo "  Generated: test_profile.nsys-rep"
    
    # Check file size
    if [ -f "test_profile.nsys-rep" ]; then
        SIZE=$(du -h test_profile.nsys-rep | cut -f1)
        echo "  File size: $SIZE"
    fi
    
    # Cleanup
    rm -f test_profile.* 2>/dev/null
else
    echo "✗ Profiling test failed"
    exit 1
fi
echo ""

# 5. Check if profile script exists
echo "5. Checking for profile_with_nsys.py..."
if [ -f "profile_with_nsys.py" ]; then
    echo "✓ profile_with_nsys.py found"
else
    echo "✗ profile_with_nsys.py not found in current directory"
    echo "  Current directory: $(pwd)"
fi
echo ""

echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  ./run_profiling_experiments.sh"
echo ""
echo "Or run individual profiling:"
echo "  nsys profile -o result python profile_with_nsys.py --forward_only"