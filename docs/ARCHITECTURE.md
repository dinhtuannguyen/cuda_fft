# Technical Architecture

This document provides detailed technical information about the FFT implementation, including algorithm details, optimization strategies, and performance characteristics.

## FFT Algorithm Overview

The Fast Fourier Transform converts a signal from time domain to frequency domain in O(N log N) operations instead of the O(N^2) required by direct computation. The Cooley-Tukey algorithm achieves this by recursively breaking down the DFT into smaller DFTs.

### Mathematical Foundation

The Discrete Fourier Transform of a sequence x[n] of length N is defined as:

```
X[k] = sum(n=0 to N-1) x[n] * W_N^(nk)
```

where W_N = exp(-2*pi*i/N) is the primitive N-th root of unity.

The Cooley-Tukey radix-2 algorithm splits this sum into even and odd indices:

```
X[k] = sum(m=0 to N/2-1) x[2m] * W_N^(2mk) + sum(m=0 to N/2-1) x[2m+1] * W_N^((2m+1)k)
     = sum(m=0 to N/2-1) x[2m] * W_(N/2)^(mk) + W_N^k * sum(m=0 to N/2-1) x[2m+1] * W_(N/2)^(mk)
     = E[k] + W_N^k * O[k]
```

This gives us the butterfly equation:
```
X[k] = E[k] + W_N^k * O[k]           for k = 0, 1, ..., N/2-1
X[k + N/2] = E[k] - W_N^k * O[k]     for k = 0, 1, ..., N/2-1
```

### Bit-Reversal Permutation

The iterative FFT algorithm requires input data in bit-reversed order. For an 8-point FFT, indices are reordered as:

```
Original:     0   1   2   3   4   5   6   7
Binary:      000 001 010 011 100 101 110 111
Reversed:    000 100 010 110 001 101 011 111
Decimal:      0   4   2   6   1   5   3   7
```

This permutation ensures that the butterfly operations combine the correct elements at each stage.

### Butterfly Operation

The core computation unit is the butterfly, which combines two complex numbers:

```
Input:  a, b
Twiddle factor: W = exp(-2*pi*i*j/m)

Output:
  a' = a + W * b
  b' = a - W * b
```

Each stage has N/2 butterflies, and there are log2(N) stages total.

## Implementation Details

### CPU Reference Implementation (cpu_fft.py)

The CPU version implements three variants for comparison and validation.

The iterative version follows the textbook algorithm with explicit bit-reversal followed by log2(N) butterfly stages. Each butterfly is computed sequentially, making it straightforward but slow.

The vectorized version uses NumPy array operations to compute all butterflies in a stage simultaneously. This provides moderate speedup through SIMD operations but is still limited by Python overhead.

The recursive version implements the classic divide-and-conquer formulation. While elegant, Python's recursion overhead makes it the slowest variant for large inputs.

### CUDA V1 Naive Implementation (fft_v1_naive.py)

The naive CUDA implementation maps the algorithm directly to GPU kernels.

The bit_reverse_kernel assigns one thread per element. Each thread computes its bit-reversed destination index and copies data accordingly. This achieves coalesced reads but scattered writes.

The butterfly_kernel processes one FFT stage per kernel launch. Each of the N/2 threads computes one butterfly operation. The twiddle factor is computed on-the-fly using CUDA's fast math intrinsics.

This approach requires log2(N) + 1 kernel launches (one for bit-reversal, log2(N) for butterfly stages), creating significant overhead for small problem sizes.

### CUDA V2 Shared Memory Implementation (fft_v2_shared.py)

The optimized implementation reduces kernel launch overhead and improves memory access patterns.

Twiddle factors are precomputed once and cached in device memory. A TwiddleCache class manages this, avoiding redundant computation for repeated FFTs of the same size.

The shared memory kernel processes multiple butterfly stages without returning to global memory. Data is loaded from global memory into shared memory, multiple stages are computed using only shared memory accesses, then results are written back to global memory. This reduces global memory bandwidth requirements significantly.

The implementation splits the FFT into two phases. Small stages (where butterfly groups fit in shared memory) are processed by the shared memory kernel. Large stages (where butterfly span exceeds shared memory) fall back to individual global memory kernel launches.

The shared memory size is limited to 2048 complex elements (32KB), allowing 11 stages to be processed in shared memory. For a 1M-point FFT with 20 stages, this means 11 stages use shared memory and 9 stages use global memory.

## Memory Access Patterns

### Global Memory Coalescing

CUDA achieves maximum memory bandwidth when threads in a warp access consecutive memory addresses. The bit-reversal kernel has inherently non-coalesced writes due to the scattered destination pattern. However, reads are coalesced since each thread reads from its sequential index.

The butterfly kernel accesses memory in a strided pattern that becomes increasingly non-coalesced in later stages. Stage 0 has stride 1 (coalesced), stage 1 has stride 2, and stage k has stride 2^k. This degradation is a fundamental characteristic of the FFT algorithm.

### Shared Memory Bank Conflicts

The Tesla P100 has 32 shared memory banks with 4-byte width. When multiple threads access the same bank, accesses are serialized. The butterfly access pattern can create bank conflicts, particularly in later stages where the stride matches the bank count.

The current implementation does not explicitly avoid bank conflicts. Adding padding to the shared memory array (e.g., allocating 2048+32 elements instead of 2048) would eliminate conflicts but was not implemented due to diminishing returns in the Numba environment.

## Performance Characteristics

### Computational Intensity

FFT has low arithmetic intensity, defined as FLOPS per byte of memory traffic. Each butterfly requires approximately 10 floating-point operations (complex multiply and two complex adds) while loading and storing 4 complex values (64 bytes for FP64).

This gives an arithmetic intensity of 10/64 = 0.156 FLOPS/byte, making FFT heavily memory-bound. The P100's 732 GB/s memory bandwidth theoretically limits performance to approximately 114 GFLOPS, far below its 4.7 TFLOPS FP64 compute capability.

### Kernel Launch Overhead

Numba CUDA kernel launches incur approximately 1ms overhead per launch. For a 1M-point FFT with the naive implementation (21 kernel launches), this adds 21ms of pure overhead, dominating the actual computation time.

The shared memory optimization reduces this to approximately 3-4 kernel launches (bit-reversal + 1 shared memory kernel + 9 global memory stages), providing meaningful improvement for smaller problem sizes.

### Occupancy Considerations

GPU occupancy measures how fully the GPU's resources are utilized. The current implementation achieves low occupancy for small problem sizes due to insufficient parallelism. Numba issues warnings for grid sizes below 128 blocks.

For large problem sizes, occupancy is limited by shared memory usage. Each block uses 32KB of shared memory, allowing only 1-2 blocks per SM on P100 (48KB shared memory per SM). This limits the ability to hide memory latency through concurrent execution.

## Validation Methodology

All implementations are validated against NumPy's FFT, which serves as a trusted reference. The test suite verifies correctness using several approaches.

Random input testing generates complex random arrays and compares results element-wise with tolerance of 1e-10. This catches gross errors but may miss subtle numerical issues.

Special input testing uses known analytical solutions. A delta impulse should produce a flat spectrum, a pure sinusoid should produce two impulses at the signal frequency, and the DC component should equal the sum of all input samples.

Mathematical property testing verifies Parseval's theorem (energy conservation), linearity, and the convolution theorem. These properties must hold for any correct FFT implementation.

Round-trip testing confirms that IFFT(FFT(x)) recovers the original signal to within numerical precision.

## Potential Optimizations Not Implemented

Several optimizations were considered but not implemented due to time constraints or Numba limitations.

Radix-4 or radix-8 algorithms reduce the number of stages by processing 4 or 8 elements per butterfly. This increases arithmetic intensity and reduces memory traffic but significantly complicates the implementation.

The Stockham auto-sort algorithm eliminates the bit-reversal permutation by using ping-pong buffers. Each stage reads from one buffer and writes to another in natural order, with the final output appearing in sorted order.

Warp-level primitives like __shfl_sync would allow threads within a warp to exchange data without shared memory, reducing latency for small butterflies. Numba does not currently expose these primitives.

Kernel fusion would combine the entire FFT into a single kernel launch, eliminating launch overhead entirely. This requires careful management of synchronization across thread blocks and is extremely complex to implement correctly.

Mixed precision computation could use FP32 for intermediate stages where precision is less critical, doubling effective memory bandwidth and compute throughput.
