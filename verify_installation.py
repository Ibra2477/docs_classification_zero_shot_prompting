#!/usr/bin/env python3
"""
Simple PyTorch Installation Verification Script

Run this script on the system where you installed PyTorch with CUDA support.
"""

def main():
    print("🔍 Verifying PyTorch Installation...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully!")
        print(f"   Version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            # Get GPU info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name}")
                print(f"           Memory: {props.total_memory / 1024**3:.1f} GB")
            
            # Quick GPU test
            print("\n🧪 Testing GPU operations...")
            try:
                # Create tensors on GPU
                x = torch.randn(100, 100, device='cuda')
                y = torch.randn(100, 100, device='cuda')
                z = torch.matmul(x, y)
                
                print(f"✅ GPU computation successful!")
                print(f"   Result tensor shape: {z.shape}")
                print(f"   Result tensor device: {z.device}")
                
                # Performance comparison
                import time
                
                # GPU test
                x_gpu = torch.randn(1000, 1000, device='cuda')
                y_gpu = torch.randn(1000, 1000, device='cuda')
                
                start = time.time()
                z_gpu = torch.matmul(x_gpu, y_gpu)
                torch.cuda.synchronize()  # Wait for GPU to finish
                gpu_time = time.time() - start
                
                # CPU test
                x_cpu = torch.randn(1000, 1000, device='cpu')
                y_cpu = torch.randn(1000, 1000, device='cpu')
                
                start = time.time()
                z_cpu = torch.matmul(x_cpu, y_cpu)
                cpu_time = time.time() - start
                
                print(f"\n⚡ Performance Comparison:")
                print(f"   GPU Time: {gpu_time:.4f} seconds")
                print(f"   CPU Time: {cpu_time:.4f} seconds")
                print(f"   Speedup: {cpu_time/gpu_time:.2f}x faster on GPU")
                
            except Exception as e:
                print(f"❌ GPU test failed: {e}")
        else:
            print("⚠️  CUDA not available - using CPU only")
            
        print("\n" + "=" * 50)
        
        if cuda_available:
            print("🎉 SUCCESS! PyTorch with GPU support is working perfectly!")
            print("   Your RTX 4050 is ready for deep learning!")
        else:
            print("ℹ️  PyTorch is working but only with CPU support")
            
    except ImportError as e:
        print(f"❌ PyTorch is not installed or not accessible: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct Python environment")
        print("2. Check if PyTorch was installed with --user flag")
        print("3. Try: python3 -m pip list | grep torch")

if __name__ == "__main__":
    main()