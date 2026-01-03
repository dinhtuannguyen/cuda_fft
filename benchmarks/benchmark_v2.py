"""
FFT Benchmark Suite - All Implementations

Compares:
- CPU Iterative
- CPU Vectorized  
- CUDA Naive (V1)
- CUDA Shared Memory (V2)
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

# CUDA implementations
try:
    from numba import cuda
    from fft_v1_naive import fft_cuda_naive
    from fft_v2_shared import fft_cuda_shared
    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f"Warning: CUDA not available: {e}")


def benchmark_single(fft_func, x, num_warmup, num_runs, use_cuda_sync=False):
    """Benchmark a single FFT function."""
    # Warmup
    for _ in range(num_warmup):
        fft_func(x)
        if use_cuda_sync:
            cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fft_func(x)
        if use_cuda_sync:
            cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return np.median(times) * 1000  # Return milliseconds


def run_all_benchmarks(sizes: list = None, num_warmup: int = 5, num_runs: int = 10) -> dict:
    """Run benchmarks for all implementations."""
    
    if sizes is None:
        sizes = [2**k for k in range(10, 21)]  # 1K to 1M
    
    print("=" * 90)
    print("FFT Comprehensive Benchmark Suite - All Implementations")
    print("=" * 90)
    print(f"Sizes: {sizes[0]:,} to {sizes[-1]:,}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if CUDA_AVAILABLE:
        try:
            device = cuda.get_current_device()
            print(f"GPU: {device.name}")
        except:
            pass
    
    print("=" * 90)
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'sizes': sizes,
        'numpy': {'times_ms': [], 'gflops': []},
        'cpu_iter': {'times_ms': [], 'gflops': []},
        'cuda_naive': {'times_ms': [], 'gflops': []},
        'cuda_shared': {'times_ms': [], 'gflops': []}
    }
    
    for N in sizes:
        print(f"\nBenchmarking N = {N:,}...")
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # NumPy
        t = benchmark_single(np.fft.fft, x, num_warmup, num_runs)
        results['numpy']['times_ms'].append(t)
        results['numpy']['gflops'].append(compute_fft_metrics(N, t)['gflops'])
        print(f"  NumPy:        {t:>10.3f} ms")
        
        # CPU Iterative (skip for very large sizes - too slow)
        if N <= 262144:
            t = benchmark_single(fft_cpu_iterative, x, min(num_warmup, 2), min(num_runs, 3))
            results['cpu_iter']['times_ms'].append(t)
            results['cpu_iter']['gflops'].append(compute_fft_metrics(N, t)['gflops'])
            print(f"  CPU Iter:     {t:>10.3f} ms")
        else:
            results['cpu_iter']['times_ms'].append(None)
            results['cpu_iter']['gflops'].append(None)
            print(f"  CPU Iter:     {'skipped':>10}")
        
        # CUDA Naive
        if CUDA_AVAILABLE:
            t = benchmark_single(fft_cuda_naive, x, num_warmup, num_runs, use_cuda_sync=True)
            results['cuda_naive']['times_ms'].append(t)
            results['cuda_naive']['gflops'].append(compute_fft_metrics(N, t)['gflops'])
            print(f"  CUDA Naive:   {t:>10.3f} ms")
            
            # CUDA Shared
            t = benchmark_single(fft_cuda_shared, x, num_warmup, num_runs, use_cuda_sync=True)
            results['cuda_shared']['times_ms'].append(t)
            results['cuda_shared']['gflops'].append(compute_fft_metrics(N, t)['gflops'])
            print(f"  CUDA Shared:  {t:>10.3f} ms")
    
    return results


def print_results_table(results: dict):
    """Print comprehensive results table."""
    sizes = results['sizes']
    
    # Table 1: Execution Times
    print("\n" + "=" * 100)
    print("EXECUTION TIME (ms)")
    print("=" * 100)
    print(f"{'Size':>12} {'NumPy':>12} {'CPU Iter':>12} {'CUDA Naive':>12} {'CUDA Shared':>12}")
    print("-" * 100)
    
    for i, N in enumerate(sizes):
        numpy_t = results['numpy']['times_ms'][i]
        cpu_t = results['cpu_iter']['times_ms'][i]
        naive_t = results['cuda_naive']['times_ms'][i] if CUDA_AVAILABLE else None
        shared_t = results['cuda_shared']['times_ms'][i] if CUDA_AVAILABLE else None
        
        row = f"{N:>12,}"
        row += f" {numpy_t:>12.3f}" if numpy_t else f" {'N/A':>12}"
        row += f" {cpu_t:>12.3f}" if cpu_t else f" {'N/A':>12}"
        row += f" {naive_t:>12.3f}" if naive_t else f" {'N/A':>12}"
        row += f" {shared_t:>12.3f}" if shared_t else f" {'N/A':>12}"
        print(row)
    
    print("=" * 100)
    
    # Table 2: GFLOPS
    print("\n" + "=" * 100)
    print("PERFORMANCE (GFLOPS)")
    print("=" * 100)
    print(f"{'Size':>12} {'NumPy':>12} {'CPU Iter':>12} {'CUDA Naive':>12} {'CUDA Shared':>12}")
    print("-" * 100)
    
    for i, N in enumerate(sizes):
        numpy_g = results['numpy']['gflops'][i]
        cpu_g = results['cpu_iter']['gflops'][i]
        naive_g = results['cuda_naive']['gflops'][i] if CUDA_AVAILABLE else None
        shared_g = results['cuda_shared']['gflops'][i] if CUDA_AVAILABLE else None
        
        row = f"{N:>12,}"
        row += f" {numpy_g:>12.2f}" if numpy_g else f" {'N/A':>12}"
        row += f" {cpu_g:>12.2f}" if cpu_g else f" {'N/A':>12}"
        row += f" {naive_g:>12.2f}" if naive_g else f" {'N/A':>12}"
        row += f" {shared_g:>12.2f}" if shared_g else f" {'N/A':>12}"
        print(row)
    
    print("=" * 100)
    
    # Table 3: Speedup Analysis
    if CUDA_AVAILABLE:
        print("\n" + "=" * 100)
        print("SPEEDUP ANALYSIS")
        print("=" * 100)
        print(f"{'Size':>12} {'Shared/Naive':>14} {'Shared/NumPy':>14} {'Shared/CPU':>14}")
        print("-" * 100)
        
        for i, N in enumerate(sizes):
            naive_t = results['cuda_naive']['times_ms'][i]
            shared_t = results['cuda_shared']['times_ms'][i]
            numpy_t = results['numpy']['times_ms'][i]
            cpu_t = results['cpu_iter']['times_ms'][i]
            
            row = f"{N:>12,}"
            
            # Shared vs Naive
            if naive_t and shared_t:
                speedup = naive_t / shared_t
                row += f" {speedup:>13.2f}x"
            else:
                row += f" {'N/A':>14}"
            
            # Shared vs NumPy
            if numpy_t and shared_t:
                speedup = numpy_t / shared_t
                if speedup >= 1:
                    row += f" {speedup:>13.2f}x"
                else:
                    row += f" {1/speedup:>10.2f}x slower"
            else:
                row += f" {'N/A':>14}"
            
            # Shared vs CPU
            if cpu_t and shared_t:
                speedup = cpu_t / shared_t
                row += f" {speedup:>13.1f}x"
            else:
                row += f" {'N/A':>14}"
            
            print(row)
        
        print("=" * 100)


def print_summary(results: dict):
    """Print performance summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    sizes = results['sizes']
    
    # Peak GFLOPS
    print("\nPeak Performance (GFLOPS):")
    print("-" * 40)
    
    for name, data in [
        ('NumPy', results['numpy']),
        ('CPU Iterative', results['cpu_iter']),
        ('CUDA Naive', results['cuda_naive']),
        ('CUDA Shared', results['cuda_shared'])
    ]:
        gflops = [g for g in data['gflops'] if g is not None]
        if gflops:
            peak = max(gflops)
            peak_idx = data['gflops'].index(peak)
            peak_size = sizes[peak_idx]
            print(f"  {name:20} {peak:>8.2f} GFLOPS @ N={peak_size:,}")
    
    if CUDA_AVAILABLE:
        # V2 vs V1 improvement
        print("\nCUDA Shared (V2) vs CUDA Naive (V1):")
        print("-" * 40)
        
        improvements = []
        for i in range(len(sizes)):
            naive_t = results['cuda_naive']['times_ms'][i]
            shared_t = results['cuda_shared']['times_ms'][i]
            if naive_t and shared_t:
                improvements.append(naive_t / shared_t)
        
        if improvements:
            print(f"  Average speedup: {np.mean(improvements):.2f}x")
            print(f"  Min speedup: {min(improvements):.2f}x")
            print(f"  Max speedup: {max(improvements):.2f}x")
        
        # V2 vs NumPy
        print("\nCUDA Shared (V2) vs NumPy:")
        print("-" * 40)
        
        faster = 0
        slower = 0
        for i in range(len(sizes)):
            numpy_t = results['numpy']['times_ms'][i]
            shared_t = results['cuda_shared']['times_ms'][i]
            if numpy_t and shared_t:
                if shared_t < numpy_t:
                    faster += 1
                else:
                    slower += 1
        
        print(f"  CUDA faster: {faster}/{len(sizes)} sizes")
        print(f"  NumPy faster: {slower}/{len(sizes)} sizes")
    
    print("\n" + "=" * 70)


def main():
    """Main benchmark runner."""
    
    # Test sizes
    sizes = [2**k for k in range(10, 21)]  # 1K to 1M
    
    # Run benchmarks
    results = run_all_benchmarks(sizes, num_warmup=5, num_runs=10)
    
    # Print results
    print_results_table(results)
    print_summary(results)
    
    # Save results
    output_file = os.path.join(
        os.path.dirname(__file__),
        'results',
        'benchmark_v2.json'
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {
        'timestamp': results['timestamp'],
        'sizes': results['sizes'],
        'implementations': {
            'numpy': results['numpy'],
            'cpu_iterative': results['cpu_iter'],
            'cuda_naive': results['cuda_naive'],
            'cuda_shared': results['cuda_shared']
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=lambda x: None if x is None else float(x))
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()