#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results
df = pd.read_csv('bk_tuning_large_matrices.csv')

print("=" * 80)
print("BK Parameter Analysis - Comprehensive Visualization")
print("=" * 80)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color scheme
colors_bk = {8: '#3498db', 16: '#e74c3c', 32: '#2ecc71'}
matrix_sizes = df['size'].unique()

# Plot 1: GFLOPS comparison (large plot)
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(matrix_sizes))
width = 0.35

for i, bk in enumerate(sorted(df['BK'].unique())):
    gflops_values = []
    for size in matrix_sizes:
        size_df = df[df['size'] == size]
        bk_data = size_df[size_df['BK'] == bk]
        if len(bk_data) > 0:
            gflops_values.append(bk_data['gflops'].values[0])
        else:
            gflops_values.append(0)

    offset = width * (i - 0.5)
    bars = ax1.bar(x + offset, gflops_values, width, label=f'BK={bk}',
                   alpha=0.8, color=colors_bk.get(bk, '#95a5a6'))

    # Add value labels on bars
    for bar, val in zip(bars, gflops_values):
        if val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
ax1.set_ylabel('Performance (GFLOPS)', fontsize=13, fontweight='bold')
ax1.set_title('Performance Comparison: BK Parameter Impact', fontsize=15, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(matrix_sizes, rotation=15, ha='right')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Plot 2: Performance improvement (BK=16 vs BK=8)
ax2 = fig.add_subplot(gs[0, 2])
improvements = []
for size in matrix_sizes:
    size_df = df[df['size'] == size]
    bk8 = size_df[size_df['BK'] == 8]['gflops'].values[0]
    bk16 = size_df[size_df['BK'] == 16]['gflops'].values[0]
    improvement = ((bk16 / bk8) - 1) * 100
    improvements.append(improvement)

bars = ax2.barh(range(len(matrix_sizes)), improvements, alpha=0.8, color='#27ae60')
ax2.set_yticks(range(len(matrix_sizes)))
ax2.set_yticklabels(matrix_sizes, fontsize=10)
ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
ax2.set_title('BK=16 vs BK=8\nPerformance Gain', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2.,
            f'+{val:.1f}%',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Plot 3: Execution time trend
ax3 = fig.add_subplot(gs[1, :2])
for bk in sorted(df['BK'].unique()):
    bk_df = df[df['BK'] == bk].sort_values('M')
    ax3.plot(bk_df['size'], bk_df['time_ms'], marker='o', linewidth=2.5,
            markersize=10, label=f'BK={bk}', color=colors_bk.get(bk, '#95a5a6'))

ax3.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
ax3.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
ax3.set_title('Execution Time vs Matrix Size', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_xticks(range(len(matrix_sizes)))
ax3.set_xticklabels(matrix_sizes, rotation=15, ha='right')
ax3.set_yscale('log')

# Plot 4: Shared memory usage
ax4 = fig.add_subplot(gs[1, 2])
bk_values = sorted(df['BK'].unique())
shared_mem_kb = [df[df['BK'] == bk]['shared_mem_bytes'].iloc[0] / 1024 for bk in bk_values]
bars = ax4.bar(range(len(bk_values)), shared_mem_kb, alpha=0.8,
               color=[colors_bk.get(bk, '#95a5a6') for bk in bk_values])
ax4.set_xticks(range(len(bk_values)))
ax4.set_xticklabels([f'BK={bk}' for bk in bk_values])
ax4.set_ylabel('Shared Memory (KB)', fontsize=11, fontweight='bold')
ax4.set_title('Shared Memory Usage', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.axhline(y=48, color='r', linestyle='--', linewidth=2, alpha=0.7, label='48KB Limit')
ax4.legend(fontsize=9)

# Add value labels
for bar, val in zip(bars, shared_mem_kb):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 5: Performance scaling
ax5 = fig.add_subplot(gs[2, :2])
for bk in sorted(df['BK'].unique()):
    bk_df = df[df['BK'] == bk].sort_values('M')
    # Normalize to smallest size
    baseline = bk_df['gflops'].iloc[0]
    normalized = (bk_df['gflops'] / baseline) * 100
    ax5.plot(bk_df['size'], normalized, marker='s', linewidth=2.5,
            markersize=10, label=f'BK={bk}', color=colors_bk.get(bk, '#95a5a6'))

ax5.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
ax5.set_ylabel('Relative Performance (%)', fontsize=13, fontweight='bold')
ax5.set_title('Performance Scaling (Normalized to 2048x2048x2048)', fontsize=15, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(alpha=0.3, linestyle='--')
ax5.set_xticks(range(len(matrix_sizes)))
ax5.set_xticklabels(matrix_sizes, rotation=15, ha='right')
ax5.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

# Plot 6: Summary table
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

# Create summary text
summary_text = "Key Findings:\n\n"
summary_text += "BK=16 is optimal for\nall matrix sizes\n\n"
summary_text += "Performance Impact:\n"
for size in matrix_sizes:
    size_df = df[df['size'] == size]
    bk8 = size_df[size_df['BK'] == 8]['gflops'].values[0]
    bk16 = size_df[size_df['BK'] == 16]['gflops'].values[0]
    improvement = ((bk16 / bk8) - 1) * 100
    summary_text += f"  {size.split('x')[0]}: +{improvement:.1f}%\n"

summary_text += "\nBK=32: Exceeds\nshared memory limit\n(66.6 KB > 48 KB)"

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        family='monospace')

plt.suptitle('BK Parameter Impact Analysis on A100 GPU',
             fontsize=17, fontweight='bold', y=0.995)

plt.savefig('bk_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved to bk_comprehensive_analysis.png")

# Print detailed statistics
print("\n" + "=" * 80)
print("Detailed Statistics")
print("=" * 80)

for size in matrix_sizes:
    size_df = df[df['size'] == size]
    print(f"\n{size}:")
    print(f"  {'BK':<5} {'Time (ms)':<12} {'GFLOPS':<12} {'vs Best':<10}")
    print(f"  {'-'*40}")
    best_gflops = size_df['gflops'].max()
    for _, row in size_df.iterrows():
        ratio = best_gflops / row['gflops']
        print(f"  {row['BK']:<5} {row['time_ms']:<12.3f} {row['gflops']:<12.2f} {ratio:.3f}x")

print("\n" + "=" * 80)
print("Recommendations")
print("=" * 80)
print("""
Based on comprehensive testing across matrix sizes 2048-16384:

1. OPTIMAL PARAMETER: BK=16
   - Consistently best performance across all sizes
   - 0.2% - 7.4% improvement over BK=8
   - Shared memory usage: 32.5 KB (well within 48KB limit)

2. BK=8 ALTERNATIVE:
   - Acceptable fallback if memory is constrained
   - Uses only 16.2 KB shared memory
   - Performance penalty: 0.2% - 7.4%

3. BK=32 NOT VIABLE:
   - Exceeds shared memory limit (66.6 KB > 48 KB)
   - Cannot be used on A100

4. MATRIX SIZE IMPACT:
   - Largest BK benefit at 4096x4096x4096 (+7.4%)
   - Smaller impact at very large sizes (1-2%)
   - BK=16 maintains advantage across all scales

CONCLUSION: Use BK=16 for optimal performance on A100 GPU.
""")
