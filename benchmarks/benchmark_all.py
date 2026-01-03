"""
Comprehensive FFT Benchmark Suite

Benchmarks CPU implementations and will be extended to include CUDA versions.
Compares against NumPy (which uses FFTW/Intel MKL internally).

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


def benchmark_implementation(
    name: str,
    fft_func,
    sizes: list,
    num_warmup: int = 3,
    num_runs: int = 10,
    sync_func=None
) -> dict:
    """
    Benchmark an FFT implementation across multiple sizes.
    
    Args:
        name: Implementation name
        fft_func: FFT function to benchmark
        sizes: List of FFT sizes
        num_warmup: Warmup runs
        num_runs: Timed runs
        sync_func: GPU synchronization function (optional)
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'name': name,
        'sizes': sizes,
        'times_ms': [],
        'gflops': [],
        'bandwidth_gb_s': []
    }
    
    for N in sizes:
        # Generate random input
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Warmup
        for _ in range(num_warmup):
            try:
                fft_func(x)
                if sync_func:
                    sync_func()
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
            if sync_func:
                sync_func()
            times.append(time.perf_counter() - start)
        
        median_time_ms = np.median(times) * 1000
        metrics = compute_fft_metrics(N, median_time_ms)
        
        results['times_ms'].append(median_time_ms)
        results['gflops'].append(metrics['gflops'])
        results['bandwidth_gb_s'].append(metrics['bandwidth_gb_s'])
    
    return results


def run_cpu_benchmarks(sizes: list = None) -> dict:
    """Run benchmarks for all CPU implementations."""
    
    if sizes is None:
        # Default: powers of 2 from 2^8 to 2^20
        sizes = [2**k for k in range(8, 21)]
    
    print("=" * 70)
    print("FFT CPU Benchmark Suite")
    print("=" * 70)
    print(f"Sizes: {sizes[0]:,} to {sizes[-1]:,}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'sizes': sizes,
        'implementations': {}
    }
    
    # Benchmark NumPy (reference)
    print("\n[1/3] Benchmarking NumPy FFT (reference)...")
    numpy_results = benchmark_implementation(
        "NumPy",
        np.fft.fft,
        sizes
    )
    all_results['implementations']['numpy'] = numpy_results
    
    # Benchmark iterative CPU
    print("[2/3] Benchmarking CPU Iterative...")
    iter_results = benchmark_implementation(
        "CPU Iterative",
        fft_cpu_iterative,
        sizes
    )
    all_results['implementations']['cpu_iterative'] = iter_results
    
    # Benchmark vectorized CPU
    print("[3/3] Benchmarking CPU Vectorized...")
    vec_results = benchmark_implementation(
        "CPU Vectorized",
        fft_cpu_vectorized,
        sizes
    )
    all_results['implementations']['cpu_vectorized'] = vec_results
    
    return all_results


def print_results_table(results: dict):
    """Print benchmark results in a formatted table."""
    
    sizes = results['sizes']
    impls = results['implementations']
    
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS - Execution Time (ms)")
    print("=" * 90)
    
    # Header
    print(f"{'Size':>12}", end="")
    for name in impls:
        print(f" {impls[name]['name']:>14}", end="")
    print(f" {'Speedup':>10}")
    print("-" * 90)
    
    # Data rows
    for i, N in enumerate(sizes):
        row = f"{N:>12,}"
        
        numpy_time = impls['numpy']['times_ms'][i]
        
        for name in impls:
            t = impls[name]['times_ms'][i]
            if t is not None:
                row += f" {t:>14.3f}"
            else:
                row += f" {'N/A':>14}"
        
        # Speedup (NumPy vs CPU Vectorized)
        cpu_time = impls['cpu_vectorized']['times_ms'][i]
        if numpy_time and cpu_time:
            speedup = cpu_time / numpy_time
            row += f" {speedup:>9.1f}×"
        else:
            row += f" {'N/A':>10}"
        
        print(row)
    
    print("=" * 90)
    
    # Performance summary
    print("\n" + "=" * 90)
    print("PERFORMANCE METRICS - GFLOPS")
    print("=" * 90)
    print(f"{'Size':>12}", end="")
    for name in impls:
        print(f" {impls[name]['name']:>14}", end="")
    print()
    print("-" * 90)
    
    for i, N in enumerate(sizes):
        row = f"{N:>12,}"
        for name in impls:
            gflops = impls[name]['gflops'][i]
            if gflops is not None:
                row += f" {gflops:>14.2f}"
            else:
                row += f" {'N/A':>14}"
        print(row)
    
    print("=" * 90)


def analyze_results(results: dict):
    """Analyze and summarize benchmark results."""
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    impls = results['implementations']
    sizes = results['sizes']
    
    # Find peak performance for each implementation
    for name, impl in impls.items():
        gflops = [g for g in impl['gflops'] if g is not None]
        if gflops:
            peak = max(gflops)
            peak_idx = impl['gflops'].index(peak)
            peak_size = sizes[peak_idx]
            print(f"{impl['name']:20} Peak: {peak:.2f} GFLOPS @ N={peak_size:,}")
    
    # Calculate speedup ratios at different sizes
    print("\n" + "-" * 70)
    print("NumPy Speedup over CPU Implementations:")
    print("-" * 70)
    
    for size_idx in [0, len(sizes)//2, -1]:
        N = sizes[size_idx]
        numpy_time = impls['numpy']['times_ms'][size_idx]
        cpu_time = impls['cpu_vectorized']['times_ms'][size_idx]
        
        if numpy_time and cpu_time:
            speedup = cpu_time / numpy_time
            print(f"  N={N:>12,}: NumPy is {speedup:.1f}× faster than CPU Vectorized")
    
    print("\n" + "=" * 70)
    print("Note: NumPy uses highly optimized FFTW/Intel MKL libraries internally.")
    print("Our goal is to match or exceed NumPy performance with CUDA.")
    print("=" * 70)


def main():
    """Main benchmark runner."""
    
    # Use smaller set of sizes for quick testing
    # Full benchmark: sizes = [2**k for k in range(8, 25)]
    sizes = [2**k for k in range(8, 19)]  # 256 to 262,144
    
    # Run benchmarks
    results = run_cpu_benchmarks(sizes)
    
    # Print results
    print_results_table(results)
    analyze_results(results)
    
    # Save results
    output_file = os.path.join(
        os.path.dirname(__file__),
        'results',
        'cpu_benchmark.json'
    )
    save_benchmark_results(results, output_file, {
        'hardware': 'CPU baseline',
        'python_version': sys.version
    })
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()