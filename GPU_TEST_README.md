# PyTorch GPU Detection Scripts

This directory contains scripts to test if PyTorch can detect and use your GPU(s).

## Scripts Included

### 1. `quick_gpu_test.py` - Quick Test
A simple script that provides a quick overview of your system's GPU capabilities.

**Usage:**
```bash
python3 quick_gpu_test.py
```

**What it does:**
- Shows system information
- Checks for NVIDIA GPU presence
- Tests PyTorch installation and CUDA support
- Performs a basic GPU computation test
- Provides installation guidance if needed

### 2. `test_pytorch_gpu.py` - Comprehensive Test
A detailed script that performs extensive GPU testing and provides comprehensive information.

**Usage:**
```bash
python3 test_pytorch_gpu.py
```

**What it does:**
- PyTorch installation details
- CUDA availability and version information
- Detailed GPU specifications (memory, compute capability, etc.)
- Basic GPU operations testing
- Tensor transfer tests (CPU ↔ GPU)
- Multiple GPU testing (if available)
- Performance timing tests

### 3. `install_pytorch.sh` - Installation Helper
An automated installation script for PyTorch with appropriate GPU support.

**Usage:**
```bash
bash install_pytorch.sh
```

**What it does:**
- Automatically detects if NVIDIA GPU is available
- Installs PyTorch with CUDA support if GPU is detected
- Installs CPU-only PyTorch if no GPU is found
- Uses the latest stable PyTorch version

## Getting Started

1. **First, run the quick test to see your current status:**
   ```bash
   python3 quick_gpu_test.py
   ```

2. **If PyTorch is not installed, use the installation script:**
   ```bash
   bash install_pytorch.sh
   ```

3. **After installation, run the comprehensive test:**
   ```bash
   python3 test_pytorch_gpu.py
   ```

## Expected Output Examples

### GPU Available and Working
```
🎉 CUDA is available and working!
   PyTorch version: 2.1.0+cu121
   CUDA version: 12.1
   Available GPUs: 1
   GPU 0: NVIDIA GeForce RTX 4090
```

### No GPU Available
```
❌ CUDA is not available.
   PyTorch will use CPU for computations.
```

### GPU Detected but PyTorch Issues
```
⚠️ GPU detected but PyTorch can't use it. 
Try reinstalling PyTorch with CUDA support.
```

## Troubleshooting

### Common Issues and Solutions

1. **PyTorch not installed:**
   - Run: `bash install_pytorch.sh`
   - Or manually: `pip3 install torch torchvision torchaudio`

2. **GPU detected but CUDA not available:**
   - Reinstall PyTorch with CUDA support
   - Check NVIDIA driver installation
   - Verify CUDA toolkit installation

3. **nvidia-smi command not found:**
   - Install NVIDIA drivers
   - On Ubuntu/Debian: `sudo apt install nvidia-utils-XXX`

4. **Out of memory errors:**
   - Your GPU might have insufficient memory
   - Try reducing tensor sizes in the test
   - Close other GPU-intensive applications

## System Requirements

### For GPU Support:
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- NVIDIA drivers (450.80.02 or newer)
- CUDA toolkit (optional, PyTorch includes CUDA libraries)

### For CPU-only:
- Any x86-64 or ARM64 processor
- Python 3.8 or newer

## File Descriptions

- `quick_gpu_test.py` - Quick diagnostic script
- `test_pytorch_gpu.py` - Comprehensive testing script  
- `install_pytorch.sh` - PyTorch installation helper
- `requirements.txt` - Python package requirements
- `GPU_TEST_README.md` - This documentation file

## Additional Information

- The scripts automatically handle both single and multi-GPU setups
- Performance tests help you compare CPU vs GPU computation speeds
- All scripts provide detailed error messages to help diagnose issues
- The installation script automatically selects the appropriate PyTorch version for your system