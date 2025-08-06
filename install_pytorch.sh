#!/bin/bash

echo "PyTorch Installation Script"
echo "=========================="

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    nvidia-smi
    echo ""
    
    # Install PyTorch with CUDA support (latest stable version)
    echo "Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    
    # Install CPU-only PyTorch
    echo "Installing PyTorch (CPU-only)..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Installation complete!"
echo "You can now run: python3 test_pytorch_gpu.py"