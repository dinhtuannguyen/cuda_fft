"""
Utility Functions for FFT Project

Common utilities for validation, timing, and visualization.

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
"""

import numpy as np
import time
from typing import Callable, List, Dict, Any, Optional
import json
from datetime import datetime


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def validate_fft_result(
    result: np.ndarray,
    expected: np.ndarray,
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """
    Validate FFT result against expected output.
    
    Returns:
        Dictionary with validation metrics
    """
    abs_diff = np.abs(result - expected)
    
    return {
        'max_error': np.max(abs_diff),
        'mean_error': np.mean(abs_diff),
        'rms_error': np.sqrt(np.mean(abs_diff**2)),
        'passed': np.max(abs_diff) < tolerance,
        'tolerance': tolerance
    }


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
    
    @property
    def ms(self) -> float:
        return self.elapsed * 1000
    
    @property
    def us(self) -> float:
        return self.elapsed * 1e6


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    num_warmup: int = 5,
    num_runs: int = 20,
    sync_func: Callable = None
) -> Dict[str, float]:
    """
    Benchmark a function with warmup and multiple runs.
    
    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        sync_func: Optional synchronization function (for GPU)
        
    Returns:
        Dictionary with timing statistics
    """
    kwargs = kwargs or {}
    
    # Warmup
    for _ in range(num_warmup):
        func(*args, **kwargs)
        if sync_func:
            sync_func()
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        if sync_func:
            sync_func()
        times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'std_ms': np.std(times),
        'num_runs': num_runs
    }


def compute_fft_metrics(N: int, time_ms: float) -> Dict[str, float]:
    """
    Compute FFT performance metrics.
    
    Args:
        N: FFT size
        time_ms: Execution time in milliseconds
        
    Returns:
        Dictionary with performance metrics
    """
    # FFT has 5N*log2(N) floating point operations (approx)
    # Each butterfly: 10 real ops (2 complex mults + 2 complex adds)
    # N/2 butterflies per stage, log2(N) stages
    flops = 5 * N * np.log2(N)
    
    # Memory: read N complex numbers, write N complex numbers
    # Each complex = 16 bytes (2 * float64)
    bytes_accessed = 2 * N * 16  # Read + Write
    
    time_s = time_ms / 1000
    
    return {
        'N': N,
        'time_ms': time_ms,
        'gflops': flops / (time_s * 1e9),
        'bandwidth_gb_s': bytes_accessed / (time_s * 1e9),
        'throughput_mfft_s': 1 / (time_s * 1e6)  # Million FFTs per second
    }


def generate_test_signal(
    N: int,
    signal_type: str = 'random',
    **kwargs
) -> np.ndarray:
    """
    Generate test signals for FFT testing.
    
    Args:
        N: Signal length
        signal_type: One of 'random', 'sine', 'cosine', 'impulse', 
                     'chirp', 'noise', 'mixed'
        **kwargs: Additional parameters for signal generation
        
    Returns:
        Complex numpy array
    """
    if signal_type == 'random':
        return np.random.randn(N) + 1j * np.random.randn(N)
    
    elif signal_type == 'sine':
        freq = kwargs.get('freq', 8)
        n = np.arange(N)
        return np.sin(2 * np.pi * freq * n / N).astype(np.complex128)
    
    elif signal_type == 'cosine':
        freq = kwargs.get('freq', 8)
        n = np.arange(N)
        return np.cos(2 * np.pi * freq * n / N).astype(np.complex128)
    
    elif signal_type == 'impulse':
        position = kwargs.get('position', 0)
        x = np.zeros(N, dtype=np.complex128)
        x[position] = 1
        return x
    
    elif signal_type == 'chirp':
        # Frequency sweep from f0 to f1
        f0 = kwargs.get('f0', 0)
        f1 = kwargs.get('f1', N // 4)
        n = np.arange(N)
        phase = 2 * np.pi * (f0 * n / N + (f1 - f0) * n**2 / (2 * N**2))
        return np.exp(1j * phase)
    
    elif signal_type == 'mixed':
        # Sum of multiple sinusoids
        freqs = kwargs.get('freqs', [4, 16, 32])
        amps = kwargs.get('amps', [1.0] * len(freqs))
        n = np.arange(N)
        x = np.zeros(N, dtype=np.complex128)
        for f, a in zip(freqs, amps):
            x += a * np.exp(2j * np.pi * f * n / N)
        return x
    
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def save_benchmark_results(
    results: Dict[str, Any],
    filename: str,
    metadata: Dict[str, Any] = None
):
    """Save benchmark results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'results': results
    }
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filename, 'w') as f:
        json.dump(convert(output), f, indent=2)


def load_benchmark_results(filename: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def print_comparison_table(
    implementations: Dict[str, List[float]],
    sizes: List[int],
    metric: str = "Time (ms)"
):
    """
    Print a comparison table for multiple implementations.
    
    Args:
        implementations: Dict mapping name -> list of values
        sizes: List of FFT sizes
        metric: Name of the metric
    """
    names = list(implementations.keys())
    
    # Header
    header = f"{'Size':>12}"
    for name in names:
        header += f" {name:>12}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for i, N in enumerate(sizes):
        row = f"{N:>12,}"
        for name in names:
            val = implementations[name][i]
            if val is not None:
                row += f" {val:>12.3f}"
            else:
                row += f" {'N/A':>12}"
        print(row)


def format_number(n: float, precision: int = 2) -> str:
    """Format number with appropriate SI suffix."""
    if n >= 1e12:
        return f"{n/1e12:.{precision}f}T"
    elif n >= 1e9:
        return f"{n/1e9:.{precision}f}G"
    elif n >= 1e6:
        return f"{n/1e6:.{precision}f}M"
    elif n >= 1e3:
        return f"{n/1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"

