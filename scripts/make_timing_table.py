#!/usr/bin/env python3
"""
Merge timing results and create a nice LaTeX table.
"""

import pandas as pd

# Load the timing results
cuda_df = pd.read_csv("data/whitehead/processed/direct_timing_results_cuda_1.csv")
cpu_df = pd.read_csv("data/whitehead/processed/direct_timing_results_cpu.csv")

# Merge on model name
merged = pd.merge(cuda_df, cpu_df, on='model')

# Create a clean table with per-sequence times
table_df = pd.DataFrame({
    'Model': merged['model'].str.replace('ESM2-650M', 'ESM2'),
    'CPU (s/seq)': merged['per_sequence_seconds_cpu'].round(4),
    'GPU (s/seq)': merged['per_sequence_seconds_cuda_1'].round(4)
})

print("Timing Results:")
print(table_df.to_string(index=False))

print("\nLaTeX Table:")
latex_table = table_df.to_latex(index=False, escape=False, float_format="%.4f")
print(latex_table)

# Save to file
with open("data/whitehead/processed/timing_table.tex", "w") as f:
    f.write(latex_table)
print("\nLaTeX table saved to: data/whitehead/processed/timing_table.tex")