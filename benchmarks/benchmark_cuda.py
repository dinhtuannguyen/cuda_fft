"""
Comprehensive FFT Benchmark Suite

Compares all FFT implementations:
- CPU Iterative
- CPU Vectorized  
- CUDA Naive (V1)
- NumPy (reference)

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
"""

import numpy as np
import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cpu_fft import fft_cpu_iterative, fft_cpu_vectorized
from utils import compute_fft_metrics, save_benchmark_results

# Try to import CUDA implementation
try:
    from numba import cuda
    from fft_v1_naive import fft_cuda_naive
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA not available. Skipping GPU benchmarks.")


def benchmark_implementation(
    name: str,
    fft_func,
    sizes: list,
    num_warmup: int = 3,
    num_runs: int = 10,
    use_cuda_sync: bool = False
) -> dict:
    """
    Benchmark an FFT implementation across multiple sizes.
    """
    results = {
        'name': name,
        'sizes': sizes,
        'times_ms': [],
        'gflops': [],
        'bandwidth_gb_s': []
    }
    
    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Warmup
        for _ in range(num_warmup):
            try:
                fft_func(x)
                if use_cuda_sync:
                    cuda.synchronize()
            except Exception as e:
                print(f"  Warning: {name} failed for N={N}: {e}")
                results['times_ms'].append(None)
                results['gflops'].append(None)
                results['bandwidth_gb_s'].append(None)
                continue
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            fft_func(x)
            if use_cuda_sync:
                cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        median_time_ms = np.median(times) * 1000
        metrics = compute_fft_metrics(N, median_time_ms)
        
        results['times_ms'].append(median_time_ms)
        results['gflops'].append(metrics['gflops'])
        results['bandwidth_gb_s'].append(metrics['bandwidth_gb_s'])
    
    return results


def run_all_benchmarks(sizes: list = None) -> dict:
    """Run benchmarks for all implementations."""
    
    if sizes is None:
        sizes = [2**k for k in range(10, 21)]  # 1K to 1M
    
    print("=" * 80)
    print("FFT Comprehensive Benchmark Suite")
    print("=" * 80)
    print(f"Sizes: {sizes[0]:,} to {sizes[-1]:,}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if CUDA_AVAILABLE:
        try:
            device = cuda.get_current_device()
            print(f"GPU: {device.name}")
        except:
            pass
    
    print("=" * 80)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'sizes': sizes,
        'implementations': {}
    }
    
    # Benchmark NumPy (reference)
    print("\n[1/4] Benchmarking NumPy FFT (reference)...")
    numpy_results = benchmark_implementation("NumPy", np.fft.fft, sizes)
    all_results['implementations']['numpy'] = numpy_results
    
    # Benchmark CPU iterative
    print("[2/4] Benchmarking CPU Iterative...")
    iter_results = benchmark_implementation("CPU Iterative", fft_cpu_iterative, sizes)
    all_results['implementations']['cpu_iterative'] = iter_results
    
    # Benchmark CPU vectorized
    print("[3/4] Benchmarking CPU Vectorized...")
    vec_results = benchmark_implementation("CPU Vectorized", fft_cpu_vectorized, sizes)
    all_results['implementations']['cpu_vectorized'] = vec_results
    
    # Benchmark CUDA naive
    if CUDA_AVAILABLE:
        print("[4/4] Benchmarking CUDA Naive...")
        cuda_results = benchmark_implementation(
            "CUDA Naive", 
            fft_cuda_naive, 
            sizes,
            use_cuda_sync=True
        )
        all_results['implementations']['cuda_naive'] = cuda_results
    else:
        print("[4/4] Skipping CUDA (not available)...")
    
    return all_results


def print_results_table(results: dict):
    """Print benchmark results in formatted tables."""
    
    sizes = results['sizes']
    impls = results['implementations']
    
    # Table 1: Execution Times
    print("\n" + "=" * 100)
    print("EXECUTION TIME (ms)")
    print("=" * 100)
    
    header = f"{'Size':>12}"
    for impl in impls.values():
        header += f" {impl['name']:>14}"
    print(header)
    print("-" * 100)
    
    for i, N in enumerate(sizes):
        row = f"{N:>12,}"
        for impl in impls.values():
            t = impl['times_ms'][i]
            if t is not None:
                row += f" {t:>14.3f}"
            else:
                row += f" {'N/A':>14}"
        print(row)
    
    print("=" * 100)
    
    # Table 2: GFLOPS
    print("\n" + "=" * 100)
    print("PERFORMANCE (GFLOPS)")
    print("=" * 100)
    
    header = f"{'Size':>12}"
    for impl in impls.values():
        header += f" {impl['name']:>14}"
    print(header)
    print("-" * 100)
    
    for i, N in enumerate(sizes):
        row = f"{N:>12,}"
        for impl in impls.values():
            g = impl['gflops'][i]
            if g is not None:
                row += f" {g:>14.2f}"
            else:
                row += f" {'N/A':>14}"
        print(row)
    
    print("=" * 100)
    
    # Table 3: Speedups (if CUDA available)
    if 'cuda_naive' in impls:
        print("\n" + "=" * 80)
        print("SPEEDUP ANALYSIS")
        print("=" * 80)
        print(f"{'Size':>12} {'CUDA vs CPU':>15} {'CUDA vs NumPy':>15} {'NumPy vs CPU':>15}")
        print("-" * 80)
        
        for i, N in enumerate(sizes):
            cuda_time = impls['cuda_naive']['times_ms'][i]
            cpu_time = impls['cpu_iterative']['times_ms'][i]
            numpy_time = impls['numpy']['times_ms'][i]
            
            if cuda_time and cpu_time:
                cuda_vs_cpu = cpu_time / cuda_time
                cuda_vs_cpu_str = f"{cuda_vs_cpu:>14.1f}x"
            else:
                cuda_vs_cpu_str = f"{'N/A':>15}"
            
            if cuda_time and numpy_time:
                cuda_vs_numpy = numpy_time / cuda_time
                if cuda_vs_numpy >= 1:
                    cuda_vs_numpy_str = f"{cuda_vs_numpy:>14.2f}x"
                else:
                    cuda_vs_numpy_str = f"{1/cuda_vs_numpy:>11.2f}x slower"
            else:
                cuda_vs_numpy_str = f"{'N/A':>15}"
            
            if numpy_time and cpu_time:
                numpy_vs_cpu = cpu_time / numpy_time
                numpy_vs_cpu_str = f"{numpy_vs_cpu:>14.1f}x"
            else:
                numpy_vs_cpu_str = f"{'N/A':>15}"
            
            print(f"{N:>12,} {cuda_vs_cpu_str} {cuda_vs_numpy_str} {numpy_vs_cpu_str}")
        
        print("=" * 80)


def analyze_results(results: dict):
    """Analyze and summarize benchmark results."""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    impls = results['implementations']
    sizes = results['sizes']
    
    # Peak GFLOPS for each implementation
    print("\nPeak Performance:")
    print("-" * 40)
    for name, impl in impls.items():
        gflops = [g for g in impl['gflops'] if g is not None]
        if gflops:
            peak = max(gflops)
            peak_idx = impl['gflops'].index(peak)
            peak_size = sizes[peak_idx]
            print(f"  {impl['name']:20} {peak:>8.2f} GFLOPS @ N={peak_size:,}")
    
    # CUDA specific analysis
    if 'cuda_naive' in impls:
        cuda_impl = impls['cuda_naive']
        numpy_impl = impls['numpy']
        cpu_impl = impls['cpu_iterative']
        
        print("\nCUDA Naive vs NumPy:")
        print("-" * 40)
        
        faster_count = 0
        slower_count = 0
        
        for i, N in enumerate(sizes):
            cuda_time = cuda_impl['times_ms'][i]
            numpy_time = numpy_impl['times_ms'][i]
            
            if cuda_time and numpy_time:
                if cuda_time < numpy_time:
                    faster_count += 1
                else:
                    slower_count += 1
        
        print(f"  CUDA faster than NumPy: {faster_count}/{len(sizes)} sizes")
        print(f"  CUDA slower than NumPy: {slower_count}/{len(sizes)} sizes")
        
        # Average speedup vs CPU
        speedups = []
        for i in range(len(sizes)):
            cuda_time = cuda_impl['times_ms'][i]
            cpu_time = cpu_impl['times_ms'][i]
            if cuda_time and cpu_time:
                speedups.append(cpu_time / cuda_time)
        
        if speedups:
            print(f"\n  Average speedup vs CPU: {np.mean(speedups):.1f}x")
            print(f"  Max speedup vs CPU: {max(speedups):.1f}x")
    
    print("\n" + "=" * 80)


def main():
    """Main benchmark runner."""
    
    # Sizes to test
    sizes = [2**k for k in range(10, 21)]  # 1K to 1M
    
    # Run benchmarks
    results = run_all_benchmarks(sizes)
    
    # Print results
    print_results_table(results)
    analyze_results(results)
    
    # Save results
    output_file = os.path.join(
        os.path.dirname(__file__),
        'results',
        'full_benchmark.json'
    )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    metadata = {
        'python_version': sys.version,
        'cuda_available': CUDA_AVAILABLE
    }
    
    if CUDA_AVAILABLE:
        try:
            device = cuda.get_current_device()
            metadata['gpu_name'] = device.name.decode() if isinstance(device.name, bytes) else str(device.name)
        except:
            pass
    
    save_benchmark_results(results, output_file, metadata)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

