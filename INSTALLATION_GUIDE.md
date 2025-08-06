# PyTorch Installation Guide

This guide provides multiple methods to install PyTorch when the automated scripts fail due to network issues or other problems.

## Your System Configuration

Based on your output, you have:
- **GPU**: NVIDIA GeForce RTX 4050 (6GB VRAM)
- **CUDA Version**: 12.8 (driver supports up to this version)
- **Driver Version**: 572.16
- **Platform**: Linux

## Installation Methods (in order of recommendation)

### Method 1: Robust Installation Script (Recommended)

Try the improved installation script that handles timeouts better:

```bash
bash install_pytorch_robust.sh
```

This script:
- Has increased timeouts (300 seconds vs default 15 seconds)
- Retries failed downloads automatically
- Tries multiple CUDA versions if one fails
- Falls back gracefully to CPU version if needed

### Method 2: Manual pip installation with timeout settings

```bash
# For GPU support (CUDA 12.1 - recommended for your RTX 4050)
pip3 install --timeout 600 --retries 10 --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Alternative: CUDA 11.8 (smaller download, still compatible)
pip3 install --timeout 600 --retries 10 --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Fallback: CPU-only version (much smaller download)
pip3 install --timeout 600 --retries 10 --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Method 3: Using conda (if available)

```bash
# Install conda/miniconda first if not available
# Then install PyTorch:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Method 4: Download and install manually

1. Go to https://pytorch.org/get-started/locally/
2. Select:
   - **PyTorch Build**: Stable
   - **Your OS**: Linux
   - **Package**: Pip
   - **Language**: Python
   - **Compute Platform**: CUDA 12.1 (or CUDA 11.8)

3. Download the wheel files manually:
   ```bash
   # Create a downloads directory
   mkdir -p ~/pytorch_downloads
   cd ~/pytorch_downloads
   
   # Download the wheel files (replace URLs with actual ones from PyTorch website)
   wget https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl
   wget https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp310-cp310-linux_x86_64.whl
   wget https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl
   
   # Install from local files
   pip3 install torch-*.whl torchvision-*.whl torchaudio-*.whl
   ```

### Method 5: Alternative PyTorch distributions

If the official PyTorch installation keeps failing, try these alternatives:

```bash
# Install a lighter version first, then upgrade
pip3 install torch-cpu torchvision-cpu
# Then try to install GPU version
pip3 install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting Network Issues

### For slow or unreliable connections:

1. **Use a VPN or different network** if possible
2. **Try during off-peak hours** when PyTorch servers are less busy
3. **Increase pip timeout**:
   ```bash
   pip3 config set global.timeout 600
   ```
4. **Use pip cache** (but clear it first if corrupted):
   ```bash
   pip3 cache purge
   ```

### For corporate networks:

```bash
# If behind a proxy, configure pip
pip3 config set global.proxy http://your-proxy:port
pip3 config set global.trusted-host download.pytorch.org
pip3 config set global.trusted-host files.pythonhosted.org
```

## Verification

After any installation method, verify it worked:

```bash
# Quick test
python3 quick_gpu_test.py

# Comprehensive test
python3 test_pytorch_gpu.py

# Manual verification
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Expected Results for Your RTX 4050

Once properly installed, you should see:
```
✅ PyTorch is installed (version 2.5.1+cu121)
🎉 CUDA is available!
  CUDA Version: 12.1
  Number of GPUs: 1
  GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB)
```

## Common Issues and Solutions

### 1. "CUDA available: False" after installation
- You may have installed the CPU-only version
- Reinstall with: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 2. "RuntimeError: CUDA out of memory"
- Your RTX 4050 has 6GB VRAM, which is sufficient for most tasks
- Reduce batch sizes in your code
- Use `torch.cuda.empty_cache()` to free memory

### 3. "No module named 'torch'"
- Installation failed or incomplete
- Try a different installation method from above

### 4. Download keeps timing out
- Use the robust installation script
- Try Method 4 (manual download)
- Check your internet connection stability

## Alternative: Using the Test Scripts Without GPU

If you can't get GPU support working immediately, you can still use the test scripts with CPU-only PyTorch:

```bash
# Install CPU-only version (small download)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run tests (will show CPU performance)
python3 quick_gpu_test.py
python3 test_pytorch_gpu.py
```

This will let you verify that PyTorch is working, and you can upgrade to GPU support later.