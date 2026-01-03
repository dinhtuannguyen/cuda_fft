"""
Naive CUDA FFT Implementation (Version 1)

First GPU implementation of Cooley-Tukey radix-2 DIT FFT.
Uses separate kernels for bit-reversal and each butterfly stage.

This version prioritizes correctness over performance.
Optimization will come in subsequent versions.

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
Hardware: Tesla P100-PCIE-16GB
"""

import numpy as np
from numba import cuda
import math
from typing import Tuple
import time


# =============================================================================
# CUDA Kernels
# =============================================================================

@cuda.jit
def bit_reverse_kernel(
    x_real_in: np.ndarray,
    x_imag_in: np.ndarray,
    x_real_out: np.ndarray,
    x_imag_out: np.ndarray,
    N: int,
    log2_N: int
):
    """
    Bit-reversal permutation kernel.
    
    Each thread handles one element, computing its bit-reversed
    destination index and copying the value there.
    
    Args:
        x_real_in: Input real components
        x_imag_in: Input imaginary components
        x_real_out: Output real components (bit-reversed order)
        x_imag_out: Output imaginary components (bit-reversed order)
        N: FFT size
        log2_N: log2(N) for bit reversal
    """
    idx = cuda.grid(1)

    if idx < N:
        # Compute bit-reversed index
        rev_idx = 0
        temp = idx
        for _ in range(log2_N):
            rev_idx = (rev_idx << 1) | (temp & 1)
            temp >>= 1

        # Copy to bit-recersed index
        x_real_out[rev_idx] = x_real_in[idx]
        x_imag_out[rev_idx] = x_imag_in[idx]


@cuda.jit
def butterfly_kernel(
    X_real: np.ndarray,
    X_imag: np.ndarray,
    N: int,
    stage: int
):
    """
    Single butterfly stage kernel.
    
    Performs all butterfly operations for one stage of the FFT.
    Each thread handles one butterfly operation.
    
    Butterfly operation:
        u = X[k + j]
        t = W * X[k + j + m/2]
        X[k + j] = u + t
        X[k + j + m/2] = u - t
    
    Where W = exp(-2*pi*i*j/m) is the twiddle factor.
    
    Args:
        X_real: Real components (modified in-place)
        X_imag: Imaginary components (modified in-place)
        N: FFT size
        stage: Current stage (0 to log2(N)-1)
    """
    tid = cuda.grid(1)

    # Number of butterflies per stage = N/2
    num_butterflies = N >> 1

    if tid < num_butterflies:
        # m = size of butterfly groups at this stage = 2^(stage+1)
        m = 1 << (stage + 1)

        # m2 = half of group size = 2^stage
        m2 = 1 << stage

        # Determine which group and position within group
        # tid = group_idx * m2 + j
        group_idx = tid // m2
        j = tid % m2

        # k = start of this butterfly group
        k = group_idx * m

        #Indices for butterfly
        idx_top = k + j
        idx_bot = k + j + m2

        # Compute twiddle factor: W = exp(-2*pi*i*j/m)
        angle = -2.0 * math.pi * j / m
        w_real = math.cos(angle)
        w_imag = math.sin(angle)

        # Load value
        u_real = X_real[idx_top]
        u_imag = X_imag[idx_top]
        v_real = X_real[idx_bot]
        v_imag = X_imag[idx_bot]
        
        # Complex multiplication: t = W * v
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        t_real = w_real * v_real - w_imag * v_imag
        t_imag = w_real * v_imag + w_imag * v_real
        
        # Butterfly: top = u + t, bottom = u - t
        X_real[idx_top] = u_real + t_real
        X_imag[idx_top] = u_imag + t_imag
        X_real[idx_bot] = u_real - t_real
        X_imag[idx_bot] = u_imag - t_imag


# =============================================================================
# Host Functions
# =============================================================================

def fft_cuda_naive(x: np.ndarray, return_device: bool = False) -> np.ndarray:
    """
    Naive CUDA FFT implementation.
    
    Performs radix-2 DIT FFT using:
    1. One kernel launch for bit-reversal permutation
    2. One kernel launch per butterfly stage (log2(N) launches)
    
    Args:
        x: Input array (complex, length must be power of 2)
        return_device: If True, return device arrays (for benchmarking)
        
    Returns:
        FFT of input array
    """
    N = len(x)
    log2_N = int(math.log2(N))

    # Validate input
    if 2**log2_N != N:
        raise ValueError(f"Input length {N} must be a power if 2")

    # Separate real and imaginary parts
    # (Numba CUDA has limited complex number support)
    x_real = np.ascontiguousarray(x.real.astype(np.float64))
    x_imag = np.ascontiguousarray(x.imag.astype(np.float64))

    # Allocate devic e memory
    d_real_in = cuda.to_device(x_real)
    d_imag_in = cuda.to_device(x_imag)
    d_real_out = cuda.device_array(N, dtype=np.float64)
    d_imag_out = cuda.device_array(N, dtype=np.float64)

    # Configure kernel launch parameters
    threads_per_block = 256
    blocks_bit_rev = (N + threads_per_block - 1) // threads_per_block
    blocks_butterfly = (N // 2 + threads_per_block - 1) // threads_per_block

    # Step 1: Bit-reversal permutation
    bit_reverse_kernel[blocks_bit_rev, threads_per_block] (
        d_real_in, d_imag_in,
        d_real_out, d_imag_out,
        N, log2_N
    )

    #Step 2:  Butterfly stages
    for stage in range(log2_N):
        butterfly_kernel[blocks_butterfly, threads_per_block] (
            d_real_out, d_imag_out,
            N, stage
        )

    if return_device:
        return d_real_out, d_imag_out

    # Copy results back to host
    result_real = d_real_out.copy_to_host()
    result_imag = d_imag_out.copy_to_host()

    return result_real + 1j * result_imag


def ifft_cuda_naive(X: np.ndarray) -> np.ndarray:
    """
    Inverse FFT using the forward FFT.
    
    IFFT(X) = (1/N) * conj(FFT(conj(X)))
    
    Args:
        X: Input spectrum array
        
    Returns:
        Inverse FFT (time domain signal)
    """
    N = len(X)
    return np.conj(fft_cuda_naive(np.conj(X))) / N


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_cuda_naive(
    sizes: list,
    num_warmup: int = 5,
    num_runs: int = 20
) -> dict:
    """
    Benchmark naive CUDA FFT implementation.
    
    Args:
        sizes: List of FFT sizes to test
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'sizes': sizes,
        'times_ms': [],
        'gflops': [],
        'speedup_vs_numpy': []
    }

    for N in sizes:
        print(f"  Benchmarking N = {N:,}...")

        # Generate random input
        x = np.random.randn(N) + 1j * np.random.randn(N)

        # Warmup
        for _ in range(num_warmup):
            fft_cuda_naive(x)
        cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            fft_cuda_naive(x)
            cuda.synchronize()
            times.append(time.perf_counter() - start)

        median_time_ms = np.median(times) * 1000.0

        # Compute GFLOPS: 5N*log2(N) operations for FFT
        flops = 5 * N * math.log2(N)
        gflops = flops / (median_time_ms *1e6)

        # Compare with NumPy
        numpy_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            np.fft.fft(x)
            numpy_times.append(time.perf_counter() - start)
        numpy_time_ms = np.median(numpy_times) * 1000

        speedup = numpy_time_ms / median_time_ms if median_time_ms > 0 else 0

        results['times_ms'].append(median_time_ms)
        results['gflops'].append(gflops)
        results['speedup_vs_numpy'].append(speedup)

    return results

def validate_cuda_fft(sizes: list = None, tolerance: float = 1e-10) -> bool:
    """
    Validate CUDA FFT against NumPy.
    
    Args:
        sizes: List of sizes to test
        tolerance: Maximum allowed error
        
    Returns:
        True if all tests pass
    """
    if sizes is None:
        sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print("Validating CUDA FFT implementation...")
    print("-" * 50)
    
    all_passed = True

    for N in sizes:
        # Random complex input
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Compute FFT with both implementations
        cuda_result = fft_cuda_naive(x)
        numpy_result = np.fft.fft(x)
        
        # Check error
        max_error = np.max(np.abs(cuda_result - numpy_result))
        passed = max_error < tolerance
        
        status = "PASS" if passed else "FAIL"
        print(f"  N={N:>6}: max_error = {max_error:.2e} [{status}]")
        
        if not passed:
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("All validation tests PASSED")
    else:
        print("Some validation tests FAILED")
    
    return all_passed


def print_benchmark_results(results: dict):
    """Print benchmark results in formatted table."""
    print("\n" + "=" * 70)
    print("CUDA Naive FFT Benchmark Results")
    print("=" * 70)
    print(f"{'Size':>12} {'Time (ms)':>12} {'GFLOPS':>12} {'vs NumPy':>12}")
    print("-" * 70)
    
    for i, N in enumerate(results['sizes']):
        time_ms = results['times_ms'][i]
        gflops = results['gflops'][i]
        speedup = results['speedup_vs_numpy'][i]
        
        speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"
        
        print(f"{N:>12,} {time_ms:>12.3f} {gflops:>12.2f} {speedup_str:>12}")
    
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CUDA FFT Implementation - Version 1 (Naive)")
    print("=" * 70)
    
    # Check CUDA availability
    print("\nCUDA Device Information:")
    print("-" * 70)
    try:
        device = cuda.get_current_device()
        print(f"  Device: {device.name.decode() if isinstance(device.name, bytes) else device.name}")
        print(f"  Compute Capability: {device.compute_capability}")
    
        # Get memory info using the correct API
        mem_info = cuda.current_context().get_memory_info()
        free_mem, total_mem = mem_info
        print(f"  Total Memory: {total_mem / 1e9:.2f} GB")
        print(f"  Free Memory:  {free_mem / 1e9:.2f} GB")
    except Exception as e:
        print(f"  Warning: Could not get device info: {e}")
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    validation_passed = validate_cuda_fft()
    
    if not validation_passed:
        print("\nValidation failed. Stopping.")
        exit(1)
    
    # Benchmark
    print("\n" + "=" * 70)
    print("BENCHMARKING")
    print("=" * 70)
    
    # Test sizes from 1K to 1M
    sizes = [2**k for k in range(10, 21)]  # 1K to 1M
    
    results = benchmark_cuda_naive(sizes, num_warmup=5, num_runs=10)
    print_benchmark_results(results)
    
    # Summary
    print("\nPerformance Summary:")
    print("-" * 70)
    
    # Find best and worst speedups
    speedups = results['speedup_vs_numpy']
    best_idx = np.argmax(speedups)
    worst_idx = np.argmin(speedups)
    
    print(f"  Best speedup:  {speedups[best_idx]:.2f}x at N={results['sizes'][best_idx]:,}")
    print(f"  Worst speedup: {speedups[worst_idx]:.2f}x at N={results['sizes'][worst_idx]:,}")
    print(f"  Peak GFLOPS:   {max(results['gflops']):.2f}")
    
    print("\nNote: This is the naive implementation. Optimizations coming in V2!")    