"""
CPU Reference Implementation of Cooley-Tukey FFT Algorithm

This module provides a pure Python/NumPy implementation of the radix-2 
Decimation-in-Time (DIT) Fast Fourier Transform for validation and 
baseline performance comparison.

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
"""

import numpy as np
from typing import Union
import time

def bit_reverse(n: int, bits: int) -> int:
    """
    Reverse the bits of integer n using 'bits' bit positions.
    
    This is required for the Cooley-Tukey DIT algorithm which requires
    input data to be in bit-reversed order.
    
    Args:
        n: Integer to reverse
        bits: Number of bits to consider
        
    Returns:
        Bit-reversed integer
        
    Example:
        bit_reverse(1, 3) = 4  # 001 -> 100
        bit_reverse(3, 3) = 6  # 011 -> 110
    """
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1 )
        n >>= 1
    return result

def bit_reverse_permutation(x: np.ndarray) -> np.ndarray:
    """
    Apply bit-reversal permutation to input array.
    
    Args:
        x: Input array of length N (must be power of 2)
        
    Returns:
        Bit-reversed array
    """
    N = len(x)
    bits = int(np.log2(N))
    indices = np.array([bit_reverse(i, bits) for i in range(N)])
    return x[indices]

def fft_cpu_recursive(x: np.ndarray) -> np.ndarray:
    """
    Recursive Cooley-Tukey FFT implementation.
    
    This is the classic textbook implementation, elegant but slower
    due to Python recursion overhead. Used for understanding the algorithm.
    
    Args:
        x: Input array (length must be power of 2)
        
    Returns:
        FFT of input array
    """
    N = len(x)

    if N == 1:
        return x.copy()

    # Recursively compute FFT of even and odd elements
    even = fft_cpu_recursive(x[0::2])
    odd = fft_cpu_recursive(x[1::2])

    # Combine results using butterfly operations
    twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N)

    return np.concatenate([
        even + twiddle * odd,
        even - twiddle * odd
    ])
    
def fft_cpu_iterative(x: np.ndarray) -> np.ndarray:    
    """
    Iterative Cooley-Tukey radix-2 DIT FFT.
    
    This is the optimized iterative version that:
    1. Performs bit-reversal permutation once at start
    2. Processes all butterfly stages iteratively
    3. Works in-place on the data
    
    This implementation closely mirrors how the CUDA version will work.
    
    Args:
        x: Input array (length must be power of 2)
        
    Returns:
        FFT of input array
        
    Algorithm:
        For each stage s = 0 to log2(N)-1:
            m = 2^(s+1)           # Butterfly group size
            For each group starting at k = 0, m, 2m, ...
                For each butterfly j = 0 to m/2-1:
                    w = exp(-2πij/m)    # Twiddle factor
                    u = X[k+j]
                    t = w * X[k+j+m/2]
                    X[k+j] = u + t
                    X[k+j+m/2] = u - t
    """
    N = len(x)
    bits = int(np.log2(N))

    # Validate input
    if 2**bits != N:
        raise ValueError(f"Input length {N} must be power of 2")

    # Step 1: Bit-reversal permutation
    X = bit_reverse_permutation(x.astype(np.complex128))

    # Step 2: Butterfly stages
    for stage in range(bits):
        m = 2 ** (stage + 1)  # Butterfly group size
        m2 = m // 2                # Half group size

        # Primitive m-th root of unity
        w_m = np.exp(-2j * np.pi / m)

        # Process each group
        for k in range(0, N, m):
            w = 1.0 + 0j  # Twiddle factor starts at 1
            
            # Process each butterfly in the group
            for j in range(m2):
                # Butterfly operation
                t = w * X[k + j + m2]
                u = X[k + j]
                
                X[k + j] = u + t
                X[k + j + m2] = u - t
                
                # Update twiddle factor
                w *= w_m
    
    return X

def fft_cpu_vectorized(x: np.ndarray) -> np.ndarray:
    """
    Vectorized iterative FFT using NumPy operations.
    
    This version uses NumPy's vectorization for better performance
    while maintaining the same algorithm structure.
    
    Args:
        x: Input array (length must be power of 2)
        
    Returns:
        FFT of input array
    """
    N = len(x)
    bits = int(np.log2(N))

    if 2**bits != N:
        raise ValueError(f"Input length {N} must be a power of 2")

    # Bit-reversal permutation
    X = bit_reverse_permutation(x.astype(np.complex128))

    # Butterfly stages - vectorized
    for stage in range(bits):
        m = 2 ** (stage + 1)
        m2 = m // 2

        # Precompute all twiddle factors for this stage
        j_indices = np.arange(m2)
        twiddles = np.exp(-2j * np.pi * j_indices / m)

        # Process all groups simultaneously
        for k in range(0, N, m):
            # Get indices for this group
            idx_top = k + j_indices
            idx_bot = k + j_indices + m2

            # Butterfly operations (vectorized)
            t = twiddles * X[idx_bot]
            u = X[idx_top].copy()

            X[idx_top] = u + t
            X[idx_bot] = u - t

    return X

def ifft_cpu(X: np.ndarray) -> np.ndarray:
    """
    Inverse FFT using the forward FFT.
    
    IFFT(X) = (1/N) * conj(FFT(conj(X)))
    
    Args:
        X: Input spectrum array
        
    Returns:
        Inverse FFT (time domain signal)
    """
    N = len(X)
    return np.conj(fft_cpu_vectorized(np.conj(X))) / N

# Alias for the recommended implementation
fft_cpu = fft_cpu_vectorized

def benchmark_cpu_fft(sizes: list, num_runs: int = 10) -> dict:
    """
    Benchmark CPU FFT implementations.
    
    Args:
        sizes: List of FFT sizes to test
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'sizes': sizes,
        'iterative_ms': [],
        'vectorized_ms': [],
        'numpy_ms': [],
        'recursive_ms': []
    }

    for N in sizes:
        print(f"Benchmarking N = {N:,}...")
        x = np.random.randn(N) + 1j * np.random.randn(N)

        # Iterative version
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            fft_cpu_iterative(x)
            times.append(time.perf_counter() - start)
        results['iterative_ms'].append(np.median(times) * 1000)

        # Vectorized version
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            fft_cpu_vectorized(x)
            times.append(time.perf_counter() - start)
        results['vectorized_ms'].append(np.median(times) * 1000)

        # NumPy FFT
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            np.fft.fft(x)
            times.append(time.perf_counter() - start)
        results['numpy_ms'].append(np.median(times) * 1000)

        # Recursive version (only for small sizes due to recursion depth)
        if N <= 4096:
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                fft_cpu_recursive(x)
                times.append(time.perf_counter() - start)
            results['recursive_ms'].append(np.median(times) * 1000)
        else:
            results['recursive_ms'].append(None)

    return results

def print_benchmark_results(results: dict) -> None:
    """
    Print benchmark results in a formatted table.
    
    Args:
        results: Dictionary from benchmark_cpu_fft()
    """
    print("\n" + "=" * 80)
    print("CPU FFT Benchmark Results")
    print("=" * 80)
    print(f"{'Size':>10} {'Iterative':>12} {'Vectorized':>12} {'NumPy':>12} {'Recursive':>12}")
    print(f"{'':>10} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12}")
    print("-" * 80)
    
    for i, N in enumerate(results['sizes']):
        iter_ms = results['iterative_ms'][i]
        vec_ms = results['vectorized_ms'][i]
        np_ms = results['numpy_ms'][i]
        rec_ms = results['recursive_ms'][i]
        
        rec_str = f"{rec_ms:.3f}" if rec_ms is not None else "N/A"
        
        print(f"{N:>10,} {iter_ms:>12.3f} {vec_ms:>12.3f} {np_ms:>12.3f} {rec_str:>12}")
    
    print("-" * 80)
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"  Vectorized vs Iterative speedup: {results['iterative_ms'][-1] / results['vectorized_ms'][-1]:.1f}x")
    print(f"  NumPy vs Vectorized speedup: {results['vectorized_ms'][-1] / results['numpy_ms'][-1]:.1f}x")
    print(f"  NumPy vs Iterative speedup: {results['iterative_ms'][-1] / results['numpy_ms'][-1]:.1f}x")

if __name__ == "__main__":
    # Quick validation
    print("FFT CPU Implementation - Quick Test")
    print("-" * 40)
    
    # Test with small array
    N = 8
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.complex128)
    
    result_iter = fft_cpu_iterative(x)
    result_vec = fft_cpu_vectorized(x)
    result_rec = fft_cpu_recursive(x)
    result_np = np.fft.fft(x)
    
    print(f"Input: {x.real}")
    print(f"\nIterative vs NumPy max error: {np.max(np.abs(result_iter - result_np)):.2e}")
    print(f"Vectorized vs NumPy max error: {np.max(np.abs(result_vec - result_np)):.2e}")
    print(f"Recursive vs NumPy max error: {np.max(np.abs(result_rec - result_np)):.2e}")
    
    # Test IFFT
    x_recovered = ifft_cpu(result_vec)
    print(f"IFFT reconstruction error: {np.max(np.abs(x_recovered - x)):.2e}")
    
    # Benchmark
    print("\n" + "=" * 40)
    print("Running benchmarks...")
    sizes = [2**k for k in range(8, 17)]  # 256 to 65536
    results = benchmark_cpu_fft(sizes, num_runs=5)
    print_benchmark_results(results)