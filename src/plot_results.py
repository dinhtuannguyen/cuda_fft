"""
FFT Benchmark Visualization

Creates performance comparison charts for all FFT implementations.

Author: Andrey Maltsev
Project: CUDA FFT Implementation for NVIDIA Portfolio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Benchmark data from your runs
sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
size_labels = ['1K', '2K', '4K', '8K', '16K', '32K', '64K', '128K', '256K', '512K', '1M']

# Execution times (ms)
numpy_times = [0.015, 0.033, 0.067, 0.171, 0.339, 0.672, 1.425, 3.231, 8.655, 16.912, 38.266]
cpu_iter_times = [5.974, 13.010, 28.356, 60.911, 130.614, 282.018, 595.580, 1262.958, 2704.078, None, None]
cuda_naive_times = [1.859, 1.945, 1.989, 2.114, 2.256, 2.573, 3.171, 3.881, 6.995, 11.524, 23.245]
cuda_shared_times = [1.259, 1.310, 1.391, 1.516, 1.752, 2.208, 2.617, 3.786, 6.552, 11.245, 22.025]

# GFLOPS
numpy_gflops = [3.32, 3.46, 3.68, 3.11, 3.38, 3.66, 3.68, 3.45, 2.73, 2.95, 2.74]
cuda_naive_gflops = [0.03, 0.06, 0.12, 0.25, 0.51, 0.95, 1.65, 2.87, 3.37, 4.32, 4.51]
cuda_shared_gflops = [0.04, 0.09, 0.18, 0.35, 0.65, 1.11, 2.00, 2.94, 3.60, 4.43, 4.76]

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 12))


# =============================================================================
# Plot 1: Execution Time Comparison (Log Scale)
# =============================================================================
ax1 = fig.add_subplot(2, 2, 1)

x = np.arange(len(sizes))
width = 0.2

# Filter out None values for CPU
cpu_times_filtered = [t if t is not None else 0 for t in cpu_iter_times]
cpu_mask = [t is not None for t in cpu_iter_times]

ax1.semilogy(x, numpy_times, 'o-', linewidth=2, markersize=8, label='NumPy (MKL)', color='#2ecc71')
ax1.semilogy(x[cpu_mask], [cpu_iter_times[i] for i in range(len(sizes)) if cpu_mask[i]], 
             's--', linewidth=2, markersize=8, label='CPU Iterative', color='#e74c3c')
ax1.semilogy(x, cuda_naive_times, '^-', linewidth=2, markersize=8, label='CUDA V1 (Naive)', color='#3498db')
ax1.semilogy(x, cuda_shared_times, 'D-', linewidth=2, markersize=8, label='CUDA V2 (Shared)', color='#9b59b6')

ax1.set_xlabel('FFT Size', fontsize=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=12)
ax1.set_title('FFT Execution Time Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(size_labels)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotation for key insight
ax1.annotate('CUDA beats NumPy\nfor N >= 256K', 
             xy=(8, 6.5), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# =============================================================================
# Plot 2: GFLOPS Performance
# =============================================================================
ax2 = fig.add_subplot(2, 2, 2)

ax2.plot(x, numpy_gflops, 'o-', linewidth=2, markersize=8, label='NumPy (MKL)', color='#2ecc71')
ax2.plot(x, cuda_naive_gflops, '^-', linewidth=2, markersize=8, label='CUDA V1 (Naive)', color='#3498db')
ax2.plot(x, cuda_shared_gflops, 'D-', linewidth=2, markersize=8, label='CUDA V2 (Shared)', color='#9b59b6')

# Add cuFFT reference line
ax2.axhline(y=300, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label='cuFFT (~300 GFLOPS)')

ax2.set_xlabel('FFT Size', fontsize=12)
ax2.set_ylabel('Performance (GFLOPS)', fontsize=12)
ax2.set_title('FFT Performance (GFLOPS)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(size_labels)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 10)

# Add annotation
ax2.annotate(f'Peak: {max(cuda_shared_gflops):.2f} GFLOPS\n(~1.6% of cuFFT)', 
             xy=(10, max(cuda_shared_gflops)), xytext=(7, 7),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='gray'),
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# =============================================================================
# Plot 3: Speedup vs CPU
# =============================================================================
ax3 = fig.add_subplot(2, 2, 3)

# Calculate speedups vs CPU
speedup_naive_cpu = []
speedup_shared_cpu = []
valid_indices = []

for i in range(len(sizes)):
    if cpu_iter_times[i] is not None:
        speedup_naive_cpu.append(cpu_iter_times[i] / cuda_naive_times[i])
        speedup_shared_cpu.append(cpu_iter_times[i] / cuda_shared_times[i])
        valid_indices.append(i)

x_valid = np.array(valid_indices)

ax3.bar(x_valid - 0.15, speedup_naive_cpu, 0.3, label='CUDA V1 (Naive)', color='#3498db', alpha=0.8)
ax3.bar(x_valid + 0.15, speedup_shared_cpu, 0.3, label='CUDA V2 (Shared)', color='#9b59b6', alpha=0.8)

ax3.set_xlabel('FFT Size', fontsize=12)
ax3.set_ylabel('Speedup vs CPU', fontsize=12)
ax3.set_title('GPU Speedup over CPU Implementation', fontsize=14, fontweight='bold')
ax3.set_xticks(x_valid)
ax3.set_xticklabels([size_labels[i] for i in valid_indices])
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (v1, v2) in enumerate(zip(speedup_naive_cpu, speedup_shared_cpu)):
    ax3.text(valid_indices[i] - 0.15, v1 + 10, f'{v1:.0f}x', ha='center', fontsize=8)
    ax3.text(valid_indices[i] + 0.15, v2 + 10, f'{v2:.0f}x', ha='center', fontsize=8)


# =============================================================================
# Plot 4: V2 vs V1 Improvement
# =============================================================================
ax4 = fig.add_subplot(2, 2, 4)

# Calculate V2/V1 speedup
v2_vs_v1 = [cuda_naive_times[i] / cuda_shared_times[i] for i in range(len(sizes))]

colors = ['#27ae60' if s > 1.2 else '#f39c12' if s > 1.1 else '#e74c3c' for s in v2_vs_v1]
bars = ax4.bar(x, v2_vs_v1, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

ax4.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
ax4.axhline(y=1.24, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: 1.24x')

ax4.set_xlabel('FFT Size', fontsize=12)
ax4.set_ylabel('Speedup (V2/V1)', fontsize=12)
ax4.set_title('CUDA V2 (Shared) vs V1 (Naive) Speedup', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(size_labels)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0.8, 1.6)

# Add value labels
for i, v in enumerate(v2_vs_v1):
    ax4.text(i, v + 0.02, f'{v:.2f}x', ha='center', fontsize=8)


# =============================================================================
# Finalize
# =============================================================================
plt.suptitle('CUDA FFT Implementation - Performance Analysis\nTesla P100-PCIE-16GB', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'fft_benchmark_results.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Chart saved to: {output_path}")

# Also save to results directory
results_path = os.path.join(output_dir, 'results', 'fft_benchmark_results.png')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
plt.savefig(results_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Chart also saved to: {results_path}")

plt.show()
