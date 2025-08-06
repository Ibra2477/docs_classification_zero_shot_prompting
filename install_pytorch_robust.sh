#!/bin/bash

echo "Robust PyTorch Installation Script"
echo "=================================="

# Function to check if PyTorch is already installed
check_pytorch_installed() {
    if python3 -c "import torch; print('PyTorch', torch.__version__, 'is already installed')" 2>/dev/null; then
        echo "✅ PyTorch is already installed!"
        python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
        return 0
    else
        return 1
    fi
}

# Function to install with increased timeout and retries
install_with_timeout() {
    local pip_args="$1"
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo "Attempt $((retry + 1)) of $max_retries..."
        
        # Use pip with increased timeout and additional options
        pip3 install --timeout 300 --retries 5 --no-cache-dir $pip_args
        
        if [ $? -eq 0 ]; then
            echo "✅ Installation successful!"
            return 0
        else
            echo "❌ Installation failed on attempt $((retry + 1))"
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "Waiting 10 seconds before retry..."
                sleep 10
            fi
        fi
    done
    
    echo "❌ All installation attempts failed"
    return 1
}

# Check if already installed
if check_pytorch_installed; then
    exit 0
fi

echo ""

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    
    echo "Method 1: Trying CUDA 12.1 version..."
    if install_with_timeout "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"; then
        echo "✅ CUDA 12.1 version installed successfully!"
    else
        echo "Method 2: Trying CUDA 11.8 version (smaller download)..."
        if install_with_timeout "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"; then
            echo "✅ CUDA 11.8 version installed successfully!"
        else
            echo "Method 3: Trying default PyPI version..."
            if install_with_timeout "torch torchvision torchaudio"; then
                echo "⚠️  Default version installed (may not have optimal GPU support)"
            else
                echo "❌ All GPU installation methods failed. Falling back to CPU version..."
                install_with_timeout "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            fi
        fi
    fi
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    
    echo "Method 1: Trying CPU-optimized version..."
    if install_with_timeout "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"; then
        echo "✅ CPU version installed successfully!"
    else
        echo "Method 2: Trying default PyPI version..."
        install_with_timeout "torch torchvision torchaudio"
    fi
fi

echo ""
echo "Installation process completed!"
echo ""

# Verify installation
echo "Verifying installation..."
if check_pytorch_installed; then
    echo ""
    echo "🎉 Success! You can now run:"
    echo "   python3 quick_gpu_test.py     # Quick test"
    echo "   python3 test_pytorch_gpu.py   # Comprehensive test"
else
    echo ""
    echo "❌ Installation verification failed."
    echo ""
    echo "Alternative installation methods:"
    echo "1. Try conda instead: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
    echo "2. Download wheel manually from: https://pytorch.org/get-started/locally/"
    echo "3. Use a different network connection or try again later"
    echo "4. Install CPU-only version: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
fi