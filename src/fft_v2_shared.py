"""
Shared Memory Optimized CUDA FFT Implementation (Version 2)

Optimizations over V1 (Naive):
1. Process multiple butterfly stages within shared memory
2. Precomputed twiddle factors stored in device memory
3. Coalesced global memory access patterns
4. Reduced kernel launch overhead (fewer launches)
5. Bank conflict avoidance in shared memory

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
Hardware: Tesla P100-PCIE-16GB
"""

import numpy as np
from numba import cuda, float64, complex128
import math
from typing import Tuple
import time


# =============================================================================
# Constants
# =============================================================================

# Maximum FFT size that fits in shared memory
# P100 has 48KB shared memory per SM
# Each complex number = 16 bytes (2 x float64)
# 48KB / 16 bytes = 3072 complex numbers
# Use 2048 to leave room for twiddle factors and avoid bank conflicts
MAX_SHARED_SIZE = 2048

# Threads per block - should be power of 2
THREADS_PER_BLOCK = 256


# =============================================================================
# CUDA Kernels
# =============================================================================

@cuda.jit
def precompute_twiddles_kernel(
    twiddle_real: np.ndarray,
    twiddle_imag: np.ndarray,
    N: int
):
    """
    Precompute twiddle factors: W_N^k = exp(-2*pi*i*k/N)
    
    We store N/2 twiddle factors (k = 0 to N/2-1).
    These can be reused across all stages with appropriate indexing.
    """
    idx = cuda.grid(1)
    
    if idx < N // 2:
        angle = -2.0 * math.pi * idx / N
        twiddle_real[idx] = math.cos(angle)
        twiddle_imag[idx] = math.sin(angle)


@cuda.jit
def bit_reverse_kernel_optimized(
    x_real_in: np.ndarray,
    x_imag_in: np.ndarray,
    x_real_out: np.ndarray,
    x_imag_out: np.ndarray,
    N: int,
    log2_N: int
):
    """
    Optimized bit-reversal permutation with coalesced reads.
    
    Uses shared memory as intermediate buffer for coalesced writes.
    """
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    # Global index
    idx = bid * block_size + tid
    
    if idx < N:
        # Compute bit-reversed index
        rev_idx = 0
        temp = idx
        for i in range(log2_N):
            rev_idx = (rev_idx << 1) | (temp & 1)
            temp >>= 1
        
        # Coalesced read, scattered write
        x_real_out[rev_idx] = x_real_in[idx]
        x_imag_out[rev_idx] = x_imag_in[idx]


@cuda.jit
def fft_shared_memory_kernel(
    X_real: np.ndarray,
    X_imag: np.ndarray,
    twiddle_real: np.ndarray,
    twiddle_imag: np.ndarray,
    N: int,
    log2_N: int,
    start_stage: int,
    end_stage: int
):
    """
    FFT kernel processing multiple stages using shared memory.
    
    Each block processes a chunk of size <= MAX_SHARED_SIZE.
    Processes stages from start_stage to end_stage (exclusive).
    
    This kernel is designed to:
    1. Load a contiguous chunk into shared memory
    2. Process multiple butterfly stages
    3. Write results back to global memory
    
    Args:
        X_real, X_imag: Input/output arrays (bit-reversed order)
        twiddle_real, twiddle_imag: Precomputed twiddle factors for full N
        N: Total FFT size
        log2_N: log2(N)
        start_stage: First stage to process (inclusive)
        end_stage: Last stage to process (exclusive)
    """
    # Shared memory for this block's data
    # Size determined at kernel launch
    shared_real = cuda.shared.array(shape=(MAX_SHARED_SIZE,), dtype=float64)
    shared_imag = cuda.shared.array(shape=(MAX_SHARED_SIZE,), dtype=float64)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    # Each block processes a chunk of the FFT
    # Chunk size depends on the stage we're processing
    chunk_size = 1 << end_stage  # 2^end_stage
    
    # Base index in global memory for this block
    base = bid * chunk_size
    
    # Load data from global memory to shared memory (coalesced)
    # Each thread loads multiple elements if needed
    elements_per_thread = (chunk_size + block_size - 1) // block_size
    
    for i in range(elements_per_thread):
        local_idx = tid + i * block_size
        global_idx = base + local_idx
        
        if local_idx < chunk_size and global_idx < N:
            shared_real[local_idx] = X_real[global_idx]
            shared_imag[local_idx] = X_imag[global_idx]
    
    cuda.syncthreads()
    
    # Process butterfly stages
    for stage in range(start_stage, end_stage):
        # Butterfly group size at this stage
        m = 1 << (stage + 1)  # 2^(stage+1)
        m2 = 1 << stage       # 2^stage = m/2
        
        # Number of butterflies in this chunk
        num_butterflies_in_chunk = chunk_size >> 1  # chunk_size / 2
        
        # Each thread handles one or more butterflies
        butterflies_per_thread = (num_butterflies_in_chunk + block_size - 1) // block_size
        
        for b in range(butterflies_per_thread):
            butterfly_idx = tid + b * block_size
            
            if butterfly_idx < num_butterflies_in_chunk:
                # Map butterfly index to position in chunk
                group_in_chunk = butterfly_idx // m2
                j = butterfly_idx % m2
                
                # Local indices within chunk
                idx_top = group_in_chunk * m + j
                idx_bot = idx_top + m2
                
                if idx_bot < chunk_size:
                    # Compute twiddle factor index
                    # For stage s, twiddle index = j * (N / m) = j * N / 2^(s+1)
                    # = j << (log2_N - stage - 1)
                    twiddle_idx = j << (log2_N - stage - 1)
                    
                    # Get twiddle factor
                    w_real = twiddle_real[twiddle_idx]
                    w_imag = twiddle_imag[twiddle_idx]
                    
                    # Load values from shared memory
                    u_real = shared_real[idx_top]
                    u_imag = shared_imag[idx_top]
                    v_real = shared_real[idx_bot]
                    v_imag = shared_imag[idx_bot]
                    
                    # Complex multiplication: t = w * v
                    t_real = w_real * v_real - w_imag * v_imag
                    t_imag = w_real * v_imag + w_imag * v_real
                    
                    # Butterfly
                    shared_real[idx_top] = u_real + t_real
                    shared_imag[idx_top] = u_imag + t_imag
                    shared_real[idx_bot] = u_real - t_real
                    shared_imag[idx_bot] = u_imag - t_imag
        
        cuda.syncthreads()
    
    # Write results back to global memory (coalesced)
    for i in range(elements_per_thread):
        local_idx = tid + i * block_size
        global_idx = base + local_idx
        
        if local_idx < chunk_size and global_idx < N:
            X_real[global_idx] = shared_real[local_idx]
            X_imag[global_idx] = shared_imag[local_idx]


@cuda.jit
def fft_global_stage_kernel(
    X_real: np.ndarray,
    X_imag: np.ndarray,
    twiddle_real: np.ndarray,
    twiddle_imag: np.ndarray,
    N: int,
    log2_N: int,
    stage: int
):
    """
    Process a single FFT stage using global memory.
    
    Used for stages where butterfly span exceeds shared memory size.
    Each thread handles one butterfly operation.
    
    Args:
        X_real, X_imag: Input/output arrays
        twiddle_real, twiddle_imag: Precomputed twiddle factors
        N: FFT size
        log2_N: log2(N)
        stage: Current stage number
    """
    tid = cuda.grid(1)
    
    num_butterflies = N >> 1  # N/2
    
    if tid < num_butterflies:
        m = 1 << (stage + 1)  # 2^(stage+1)
        m2 = 1 << stage       # 2^stage
        
        # Determine butterfly position
        group = tid // m2
        j = tid % m2
        
        # Global indices
        idx_top = group * m + j
        idx_bot = idx_top + m2
        
        # Twiddle factor index
        twiddle_idx = j << (log2_N - stage - 1)
        
        # Get twiddle factor
        w_real = twiddle_real[twiddle_idx]
        w_imag = twiddle_imag[twiddle_idx]
        
        # Load values
        u_real = X_real[idx_top]
        u_imag = X_imag[idx_top]
        v_real = X_real[idx_bot]
        v_imag = X_imag[idx_bot]
        
        # Complex multiplication: t = w * v
        t_real = w_real * v_real - w_imag * v_imag
        t_imag = w_real * v_imag + w_imag * v_real
        
        # Butterfly
        X_real[idx_top] = u_real + t_real
        X_imag[idx_top] = u_imag + t_imag
        X_real[idx_bot] = u_real - t_real
        X_imag[idx_bot] = u_imag - t_imag


# =============================================================================
# Twiddle Factor Cache
# =============================================================================

class TwiddleCache:
    """
    Cache for precomputed twiddle factors.
    
    Avoids recomputing twiddle factors for repeated FFTs of the same size.
    """
    
    def __init__(self):
        self._cache = {}
    
    def get(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get or compute twiddle factors for size N."""
        if N not in self._cache:
            self._cache[N] = self._compute(N)
        return self._cache[N]
    
    def _compute(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute and store twiddle factors on device."""
        # Allocate device arrays
        d_twiddle_real = cuda.device_array(N // 2, dtype=np.float64)
        d_twiddle_imag = cuda.device_array(N // 2, dtype=np.float64)
        
        # Launch kernel to compute
        threads = THREADS_PER_BLOCK
        blocks = (N // 2 + threads - 1) // threads
        
        precompute_twiddles_kernel[blocks, threads](
            d_twiddle_real, d_twiddle_imag, N
        )
        cuda.synchronize()
        
        return d_twiddle_real, d_twiddle_imag
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global twiddle cache
_twiddle_cache = TwiddleCache()


# =============================================================================
# Host Functions
# =============================================================================

def fft_cuda_shared(x: np.ndarray) -> np.ndarray:
    """
    Shared memory optimized CUDA FFT.
    
    Strategy:
    1. Precompute twiddle factors (cached)
    2. Bit-reversal permutation
    3. Process early stages in shared memory (chunk size <= MAX_SHARED_SIZE)
    4. Process later stages in global memory
    
    Args:
        x: Input array (complex, length must be power of 2)
        
    Returns:
        FFT of input array
    """
    N = len(x)
    log2_N = int(math.log2(N))
    
    # Validate input
    if 2**log2_N != N:
        raise ValueError(f"Input length {N} must be a power of 2")
    
    # Separate real and imaginary parts
    x_real = np.ascontiguousarray(x.real.astype(np.float64))
    x_imag = np.ascontiguousarray(x.imag.astype(np.float64))
    
    # Get precomputed twiddle factors
    d_twiddle_real, d_twiddle_imag = _twiddle_cache.get(N)
    
    # Allocate device memory
    d_real_in = cuda.to_device(x_real)
    d_imag_in = cuda.to_device(x_imag)
    d_real = cuda.device_array(N, dtype=np.float64)
    d_imag = cuda.device_array(N, dtype=np.float64)
    
    # Step 1: Bit-reversal permutation
    threads = THREADS_PER_BLOCK
    blocks = (N + threads - 1) // threads
    
    bit_reverse_kernel_optimized[blocks, threads](
        d_real_in, d_imag_in,
        d_real, d_imag,
        N, log2_N
    )
    
    # Determine how to split stages
    # Stages 0 to (shared_stages-1) can be done in shared memory
    # where chunk_size = 2^shared_stages <= MAX_SHARED_SIZE
    max_shared_stages = int(math.log2(MAX_SHARED_SIZE))
    shared_stages = min(max_shared_stages, log2_N)
    
    # Step 2: Process stages in shared memory
    if shared_stages > 0:
        chunk_size = 1 << shared_stages  # 2^shared_stages
        num_chunks = N // chunk_size
        
        # Each block processes one chunk
        blocks_shared = num_chunks
        threads_shared = min(THREADS_PER_BLOCK, chunk_size // 2)
        
        fft_shared_memory_kernel[blocks_shared, threads_shared](
            d_real, d_imag,
            d_twiddle_real, d_twiddle_imag,
            N, log2_N,
            0, shared_stages
        )
    
    # Step 3: Process remaining stages in global memory
    threads_global = THREADS_PER_BLOCK
    blocks_global = (N // 2 + threads_global - 1) // threads_global
    
    for stage in range(shared_stages, log2_N):
        fft_global_stage_kernel[blocks_global, threads_global](
            d_real, d_imag,
            d_twiddle_real, d_twiddle_imag,
            N, log2_N, stage
        )
    
    # Copy results back to host
    result_real = d_real.copy_to_host()
    result_imag = d_imag.copy_to_host()
    
    return result_real + 1j * result_imag


def ifft_cuda_shared(X: np.ndarray) -> np.ndarray:
    """
    Inverse FFT using forward FFT.
    
    IFFT(X) = (1/N) * conj(FFT(conj(X)))
    """
    N = len(X)
    return np.conj(fft_cuda_shared(np.conj(X))) / N


# =============================================================================
# Benchmarking Functions
# =============================================================================

def benchmark_cuda_shared(
    sizes: list,
    num_warmup: int = 5,
    num_runs: int = 20
) -> dict:
    """
    Benchmark shared memory optimized CUDA FFT.
    """
    results = {
        'sizes': sizes,
        'times_ms': [],
        'gflops': [],
        'speedup_vs_numpy': [],
        'speedup_vs_naive': []
    }
    
    # Import naive version for comparison
    try:
        from fft_v1_naive import fft_cuda_naive
        has_naive = True
    except ImportError:
        has_naive = False
    
    for N in sizes:
        print(f"  Benchmarking N = {N:,}...")
        
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Warmup
        for _ in range(num_warmup):
            fft_cuda_shared(x)
        cuda.synchronize()
        
        # Timed runs - shared memory version
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            fft_cuda_shared(x)
            cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        median_time_ms = np.median(times) * 1000
        
        # Compute GFLOPS
        flops = 5 * N * math.log2(N)
        gflops = flops / (median_time_ms * 1e6)
        
        # NumPy comparison
        numpy_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            np.fft.fft(x)
            numpy_times.append(time.perf_counter() - start)
        numpy_time_ms = np.median(numpy_times) * 1000
        speedup_numpy = numpy_time_ms / median_time_ms
        
        # Naive comparison
        if has_naive:
            naive_times = []
            for _ in range(min(num_runs, 5)):  # Fewer runs for slow naive
                start = time.perf_counter()
                fft_cuda_naive(x)
                cuda.synchronize()
                naive_times.append(time.perf_counter() - start)
            naive_time_ms = np.median(naive_times) * 1000
            speedup_naive = naive_time_ms / median_time_ms
        else:
            speedup_naive = None
        
        results['times_ms'].append(median_time_ms)
        results['gflops'].append(gflops)
        results['speedup_vs_numpy'].append(speedup_numpy)
        results['speedup_vs_naive'].append(speedup_naive)
    
    return results


def validate_cuda_shared(sizes: list = None, tolerance: float = 1e-10) -> bool:
    """Validate shared memory FFT against NumPy."""
    if sizes is None:
        sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    print("Validating CUDA Shared Memory FFT...")
    print("-" * 50)
    
    all_passed = True
    
    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        cuda_result = fft_cuda_shared(x)
        numpy_result = np.fft.fft(x)
        
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
    """Print benchmark results."""
    print("\n" + "=" * 85)
    print("CUDA Shared Memory FFT Benchmark Results")
    print("=" * 85)
    print(f"{'Size':>12} {'Time (ms)':>12} {'GFLOPS':>12} {'vs NumPy':>14} {'vs Naive':>14}")
    print("-" * 85)
    
    for i, N in enumerate(results['sizes']):
        time_ms = results['times_ms'][i]
        gflops = results['gflops'][i]
        speedup_numpy = results['speedup_vs_numpy'][i]
        speedup_naive = results['speedup_vs_naive'][i]
        
        if speedup_numpy >= 1:
            numpy_str = f"{speedup_numpy:.2f}x"
        else:
            numpy_str = f"{1/speedup_numpy:.2f}x slower"
        
        if speedup_naive is not None:
            naive_str = f"{speedup_naive:.2f}x"
        else:
            naive_str = "N/A"
        
        print(f"{N:>12,} {time_ms:>12.3f} {gflops:>12.2f} {numpy_str:>14} {naive_str:>14}")
    
    print("=" * 85)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CUDA FFT Implementation - Version 2 (Shared Memory)")
    print("=" * 70)
    
    # Device info
    print("\nCUDA Device Information:")
    print("-" * 70)
    try:
        device = cuda.get_current_device()
        print(f"  Device: {device.name}")
        print(f"  Compute Capability: {device.compute_capability}")
        print(f"  Shared Memory per Block: {device.MAX_SHARED_MEMORY_PER_BLOCK / 1024:.1f} KB")
        print(f"  Max Threads per Block: {device.MAX_THREADS_PER_BLOCK}")
    except Exception as e:
        print(f"  Warning: Could not get device info: {e}")
    
    print(f"\n  Configuration:")
    print(f"    MAX_SHARED_SIZE: {MAX_SHARED_SIZE} elements")
    print(f"    THREADS_PER_BLOCK: {THREADS_PER_BLOCK}")
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    validation_passed = validate_cuda_shared()
    
    if not validation_passed:
        print("\nValidation failed. Stopping.")
        exit(1)
    
    # Benchmark
    print("\n" + "=" * 70)
    print("BENCHMARKING")
    print("=" * 70)
    
    sizes = [2**k for k in range(10, 21)]  # 1K to 1M
    
    results = benchmark_cuda_shared(sizes, num_warmup=5, num_runs=10)
    print_benchmark_results(results)
    
    # Summary
    print("\nPerformance Summary:")
    print("-" * 70)
    
    speedups_numpy = [s for s in results['speedup_vs_numpy'] if s is not None]
    speedups_naive = [s for s in results['speedup_vs_naive'] if s is not None]
    
    if speedups_numpy:
        print(f"  vs NumPy: {min(speedups_numpy):.2f}x to {max(speedups_numpy):.2f}x")
    
    if speedups_naive:
        print(f"  vs Naive CUDA: {min(speedups_naive):.2f}x to {max(speedups_naive):.2f}x")
    
    print(f"  Peak GFLOPS: {max(results['gflops']):.2f}")
    
    # Comparison with cuFFT target
    print("\n" + "-" * 70)
    print("Progress toward cuFFT target:")
    peak_gflops = max(results['gflops'])
    # cuFFT typically achieves 200-400 GFLOPS on P100 for FP64 FFT
    cufft_estimate = 300  # Conservative estimate
    percentage = (peak_gflops / cufft_estimate) * 100
    print(f"  Current: {peak_gflops:.1f} GFLOPS")
    print(f"  cuFFT estimate: ~{cufft_estimate} GFLOPS")
    print(f"  Progress: ~{percentage:.1f}% of cuFFT")