#!/usr/bin/env python3
"""
PyTorch GPU Detection Test Script

This script tests if PyTorch can detect and use your GPU(s).
It provides comprehensive information about GPU availability, CUDA support,
and basic GPU operations.
"""

import sys
import torch
import platform
from datetime import datetime

def print_separator(title=""):
    """Print a formatted separator with optional title"""
    print("\n" + "="*60)
    if title:
        print(f" {title}")
        print("="*60)
    else:
        print()

def check_pytorch_installation():
    """Check PyTorch installation details"""
    print_separator("PyTorch Installation Info")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
def check_cuda_availability():
    """Check CUDA availability and version"""
    print_separator("CUDA Availability")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        
        # Number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Current GPU
        current_device = torch.cuda.current_device()
        print(f"Current GPU Device ID: {current_device}")
        
        return True
    else:
        print("CUDA is not available. Possible reasons:")
        print("  - No NVIDIA GPU")
        print("  - NVIDIA drivers not installed")
        print("  - CUDA toolkit not installed")
        print("  - PyTorch installed without CUDA support")
        return False

def check_gpu_details():
    """Display detailed GPU information"""
    if not torch.cuda.is_available():
        return
        
    print_separator("GPU Details")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Multi Processor Count: {props.multi_processor_count}")
        print(f"CUDA Compute Capability: {props.major}.{props.minor}")
        
        # Memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")

def test_basic_operations():
    """Test basic GPU operations"""
    if not torch.cuda.is_available():
        print_separator("CPU Operations Test")
        print("Testing basic operations on CPU...")
        device = torch.device("cpu")
    else:
        print_separator("GPU Operations Test")
        print("Testing basic operations on GPU...")
        device = torch.device("cuda")
    
    try:
        # Create tensors
        print(f"Creating tensors on {device}...")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        print("Performing matrix multiplication...")
        start_time = datetime.now()
        c = torch.matmul(a, b)
        end_time = datetime.now()
        
        elapsed = (end_time - start_time).total_seconds()
        print(f"Matrix multiplication completed in {elapsed:.4f} seconds")
        print(f"Result tensor shape: {c.shape}")
        print(f"Result tensor device: {c.device}")
        
        # Memory cleanup
        del a, b, c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("✅ Basic operations test passed!")
        
    except Exception as e:
        print(f"❌ Error during operations test: {e}")

def test_tensor_transfer():
    """Test transferring tensors between CPU and GPU"""
    if not torch.cuda.is_available():
        print_separator("Skipping Tensor Transfer Test (No CUDA)")
        return
        
    print_separator("Tensor Transfer Test")
    
    try:
        # Create tensor on CPU
        print("Creating tensor on CPU...")
        cpu_tensor = torch.randn(100, 100)
        print(f"CPU tensor device: {cpu_tensor.device}")
        
        # Transfer to GPU
        print("Transferring tensor to GPU...")
        gpu_tensor = cpu_tensor.cuda()
        print(f"GPU tensor device: {gpu_tensor.device}")
        
        # Transfer back to CPU
        print("Transferring tensor back to CPU...")
        cpu_tensor_back = gpu_tensor.cpu()
        print(f"Transferred back tensor device: {cpu_tensor_back.device}")
        
        # Verify data integrity
        if torch.equal(cpu_tensor, cpu_tensor_back):
            print("✅ Tensor transfer test passed! Data integrity maintained.")
        else:
            print("❌ Data integrity check failed!")
            
    except Exception as e:
        print(f"❌ Error during tensor transfer test: {e}")

def test_multiple_gpus():
    """Test multiple GPU functionality if available"""
    gpu_count = torch.cuda.device_count()
    
    if gpu_count <= 1:
        print_separator("Multiple GPU Test Skipped")
        print(f"Only {gpu_count} GPU(s) available")
        return
        
    print_separator("Multiple GPU Test")
    print(f"Testing operations across {gpu_count} GPUs...")
    
    try:
        tensors = []
        for i in range(gpu_count):
            print(f"Creating tensor on GPU {i}...")
            tensor = torch.randn(500, 500, device=f"cuda:{i}")
            tensors.append(tensor)
            print(f"Tensor on GPU {i}: {tensor.device}")
        
        print("✅ Multiple GPU test passed!")
        
    except Exception as e:
        print(f"❌ Error during multiple GPU test: {e}")

def main():
    """Main function to run all tests"""
    print("PyTorch GPU Detection Test")
    print(f"Test started at: {datetime.now()}")
    
    # Run all tests
    check_pytorch_installation()
    cuda_available = check_cuda_availability()
    
    if cuda_available:
        check_gpu_details()
    
    test_basic_operations()
    test_tensor_transfer()
    test_multiple_gpus()
    
    print_separator("Test Summary")
    
    if torch.cuda.is_available():
        print("🎉 CUDA is available and working!")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
    else:
        print("❌ CUDA is not available.")
        print("   PyTorch will use CPU for computations.")
    
    print(f"\nTest completed at: {datetime.now()}")

if __name__ == "__main__":
    main()