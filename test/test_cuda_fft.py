"""
CUDA FFT Validation Tests

Comprehensive tests for CUDA FFT implementations.
Tests correctness against NumPy and CPU reference.

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from numba import cuda


def check_cuda_available():
    """Check if CUDA is available."""
    try:
        cuda.detect()
        return True
    except Exception:
        return False


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add(self, name: str, passed: bool, details: str = ""):
        self.tests.append((name, passed, details))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def summary(self):
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for name, passed, details in self.tests:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {name}")
            if details and not passed:
                print(f"       {details}")
        
        print("-" * 70)
        total = self.passed + self.failed
        print(f"Results: {self.passed}/{total} tests passed")
        
        return self.failed == 0


def test_cuda_naive_correctness():
    """Test CUDA naive FFT against NumPy."""
    from fft_v1_naive import fft_cuda_naive
    
    results = TestResults()
    tolerance = 1e-10
    
    sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        cuda_result = fft_cuda_naive(x)
        numpy_result = np.fft.fft(x)
        
        max_error = np.max(np.abs(cuda_result - numpy_result))
        results.add(
            f"CUDA Naive FFT N={N}",
            max_error < tolerance,
            f"max_error = {max_error:.2e}"
        )
    
    return results


def test_cuda_naive_special_inputs():
    """Test CUDA FFT with special input patterns."""
    from fft_v1_naive import fft_cuda_naive
    
    results = TestResults()
    tolerance = 1e-10
    N = 1024
    
    # Test 1: All zeros
    x = np.zeros(N, dtype=np.complex128)
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("All zeros", error < tolerance, f"error = {error:.2e}")
    
    # Test 2: All ones
    x = np.ones(N, dtype=np.complex128)
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("All ones", error < tolerance, f"error = {error:.2e}")
    
    # Test 3: Delta impulse
    x = np.zeros(N, dtype=np.complex128)
    x[0] = 1
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("Delta impulse", error < tolerance, f"error = {error:.2e}")
    
    # Test 4: Pure cosine
    n = np.arange(N)
    freq = 8
    x = np.cos(2 * np.pi * freq * n / N).astype(np.complex128)
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("Pure cosine", error < tolerance, f"error = {error:.2e}")
    
    # Test 5: Complex exponential
    x = np.exp(2j * np.pi * freq * n / N)
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("Complex exponential", error < tolerance, f"error = {error:.2e}")
    
    # Test 6: Real-valued input
    x = np.random.randn(N).astype(np.complex128)
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("Real-valued input", error < tolerance, f"error = {error:.2e}")
    
    # Test 7: Pure imaginary input
    x = 1j * np.random.randn(N)
    cuda_result = fft_cuda_naive(x)
    numpy_result = np.fft.fft(x)
    error = np.max(np.abs(cuda_result - numpy_result))
    results.add("Pure imaginary input", error < tolerance, f"error = {error:.2e}")
    
    return results


def test_cuda_naive_ifft():
    """Test CUDA IFFT round-trip."""
    from fft_v1_naive import fft_cuda_naive, ifft_cuda_naive
    
    results = TestResults()
    tolerance = 1e-10
    
    sizes = [256, 1024, 4096, 16384]
    
    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        # Forward then inverse
        X = fft_cuda_naive(x)
        x_recovered = ifft_cuda_naive(X)
        
        error = np.max(np.abs(x_recovered - x))
        results.add(
            f"IFFT round-trip N={N}",
            error < tolerance,
            f"max_error = {error:.2e}"
        )
    
    return results


def test_cuda_naive_parseval():
    """Test Parseval's theorem with CUDA FFT."""
    from fft_v1_naive import fft_cuda_naive
    
    results = TestResults()
    tolerance = 1e-8
    
    sizes = [256, 1024, 4096]
    
    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        X = fft_cuda_naive(x)
        
        energy_time = np.sum(np.abs(x)**2)
        energy_freq = np.sum(np.abs(X)**2) / N
        
        relative_error = abs(energy_time - energy_freq) / energy_time
        results.add(
            f"Parseval's theorem N={N}",
            relative_error < tolerance,
            f"relative_error = {relative_error:.2e}"
        )
    
    return results


def test_cuda_vs_cpu():
    """Test CUDA FFT against CPU reference implementation."""
    from fft_v1_naive import fft_cuda_naive
    from cpu_fft import fft_cpu_iterative
    
    results = TestResults()
    tolerance = 1e-10
    
    sizes = [256, 1024, 4096]
    
    for N in sizes:
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        cuda_result = fft_cuda_naive(x)
        cpu_result = fft_cpu_iterative(x)
        
        error = np.max(np.abs(cuda_result - cpu_result))
        results.add(
            f"CUDA vs CPU N={N}",
            error < tolerance,
            f"max_error = {error:.2e}"
        )
    
    return results


def test_large_sizes():
    """Test with larger FFT sizes."""
    from fft_v1_naive import fft_cuda_naive
    
    results = TestResults()
    tolerance = 1e-9
    
    sizes = [32768, 65536, 131072, 262144]
    
    for N in sizes:
        print(f"  Testing N = {N:,}...")
        x = np.random.randn(N) + 1j * np.random.randn(N)
        
        cuda_result = fft_cuda_naive(x)
        numpy_result = np.fft.fft(x)
        
        error = np.max(np.abs(cuda_result - numpy_result))
        results.add(
            f"Large FFT N={N:,}",
            error < tolerance,
            f"max_error = {error:.2e}"
        )
    
    return results


def run_all_cuda_tests():
    """Run all CUDA test suites."""
    print("=" * 70)
    print("CUDA FFT Implementation Test Suite")
    print("=" * 70)
    
    # Check CUDA availability
    if not check_cuda_available():
        print("ERROR: CUDA is not available. Cannot run tests.")
        return False
    
    print("CUDA detected. Running tests...\n")
    
    all_results = TestResults()
    
    # Run each test suite
    print("[1/6] Testing CUDA naive correctness...")
    for test in test_cuda_naive_correctness().tests:
        all_results.tests.append(test)
        if test[1]: all_results.passed += 1
        else: all_results.failed += 1
    
    print("[2/6] Testing special inputs...")
    for test in test_cuda_naive_special_inputs().tests:
        all_results.tests.append(test)
        if test[1]: all_results.passed += 1
        else: all_results.failed += 1
    
    print("[3/6] Testing IFFT round-trip...")
    for test in test_cuda_naive_ifft().tests:
        all_results.tests.append(test)
        if test[1]: all_results.passed += 1
        else: all_results.failed += 1
    
    print("[4/6] Testing Parseval's theorem...")
    for test in test_cuda_naive_parseval().tests:
        all_results.tests.append(test)
        if test[1]: all_results.passed += 1
        else: all_results.failed += 1
    
    print("[5/6] Testing CUDA vs CPU reference...")
    for test in test_cuda_vs_cpu().tests:
        all_results.tests.append(test)
        if test[1]: all_results.passed += 1
        else: all_results.failed += 1
    
    print("[6/6] Testing large sizes...")
    for test in test_large_sizes().tests:
        all_results.tests.append(test)
        if test[1]: all_results.passed += 1
        else: all_results.failed += 1
    
    return all_results.summary()


if __name__ == "__main__":
    success = run_all_cuda_tests()
    sys.exit(0 if success else 1)

