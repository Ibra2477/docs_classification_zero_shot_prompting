#!/usr/bin/env python3
"""
Quick PyTorch GPU Test Script

A simple script to quickly test GPU detection.
If PyTorch is not installed, it will show system information instead.
"""

import sys
import subprocess
import platform

def run_command(command):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_system_info():
    """Check basic system information"""
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python Version: {sys.version}")
    print()

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("Checking for NVIDIA GPU...")
    
    # Try nvidia-smi command
    returncode, stdout, stderr = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    
    if returncode == 0 and stdout.strip():
        print("✅ NVIDIA GPU(s) detected:")
        for line in stdout.strip().split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
        return True
    else:
        print("❌ No NVIDIA GPU detected or nvidia-smi not available")
        return False

def check_pytorch():
    """Check PyTorch installation and GPU support"""
    try:
        import torch
        print(f"\n✅ PyTorch is installed (version {torch.__version__})")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print("🎉 CUDA is available!")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
                
            # Quick test
            print("\nPerforming quick GPU test...")
            device = torch.device("cuda")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✅ GPU computation successful! Result shape: {z.shape}")
            
        else:
            print("⚠️  PyTorch is installed but CUDA is not available")
            print("  PyTorch will use CPU for computations")
            
    except ImportError:
        print("❌ PyTorch is not installed")
        print("\nTo install PyTorch:")
        print("  For GPU support: bash install_pytorch.sh")
        print("  Or manually: pip3 install torch torchvision torchaudio")
        return False
        
    return True

def main():
    print("Quick PyTorch GPU Detection Test")
    print("=" * 40)
    
    check_system_info()
    has_nvidia = check_nvidia_gpu()
    has_pytorch = check_pytorch()
    
    print("\n" + "=" * 40)
    print("Summary:")
    
    if has_nvidia and has_pytorch:
        try:
            import torch
            if torch.cuda.is_available():
                print("🎉 Everything looks good! GPU acceleration is available.")
            else:
                print("⚠️  GPU detected but PyTorch can't use it. Try reinstalling PyTorch with CUDA support.")
        except ImportError:
            pass
    elif has_nvidia and not has_pytorch:
        print("⚠️  GPU detected but PyTorch not installed. Run: bash install_pytorch.sh")
    elif not has_nvidia and has_pytorch:
        print("ℹ️  PyTorch installed but no GPU detected. CPU-only mode available.")
    else:
        print("ℹ️  No GPU detected. Install PyTorch for CPU-only usage.")

if __name__ == "__main__":
    main()