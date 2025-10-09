#!/usr/bin/env python3
"""
Generic preprocessing pipeline for MAGMA-seq antibody binding data.

Handles replicate aggregation and quality filtering for any MAGMA-seq dataset:
1. Filters out high-variance replicates (CV > 0.5)
2. Aggregates replicates using geometric mean in log10 space
3. Adds standardized columns for downstream analysis

This ensures consistent preprocessing across all datasets before merging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def calculate_geometric_mean(values):
    """Calculate geometric mean of values (antilog of mean log)."""
    log_values = np.log10(values)
    mean_log = np.mean(log_values)
    return 10 ** mean_log

def preprocess_magma_seq(input_path, output_path=None, cv_threshold=0.5, 
                        dataset_name=None, verbose=True):
    """
    Preprocess MAGMA-seq data with replicate handling and quality filtering.
    
    Args:
        input_path: Path to input CSV with VH, VL, KD columns
        output_path: Path for output CSV (auto-generated if None)
        cv_threshold: Maximum coefficient of variation for reliable measurements
        dataset_name: Name to identify the dataset
        verbose: Print processing statistics
        
    Returns:
        DataFrame with preprocessed data
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Validate required columns
    required_cols = ['VH', 'VL', 'KD']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    dataset_label = dataset_name or Path(input_path).stem
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_label}")
        print(f"{'='*60}")
        print(f"Original data: {len(df)} measurements")
        print(f"Unique VH/VL pairs: {len(df.drop_duplicates(['VH', 'VL']))}")
    
    # Analyze replicates
    duplicates = df[df.duplicated(subset=['VH', 'VL'], keep=False)]
    
    if verbose and len(duplicates) > 0:
        print(f"\nReplicate analysis:")
        print(f"  Total replicate measurements: {len(duplicates)}")
        
        # Group duplicates for analysis
        dup_grouped = duplicates.groupby(['VH', 'VL']).agg({
            'KD': ['count', 'mean', 'std', 'min', 'max']
        }).reset_index()
        dup_grouped.columns = ['VH', 'VL', 'count', 'mean', 'std', 'min', 'max']
        dup_grouped['cv'] = dup_grouped['std'] / dup_grouped['mean']
        
        print(f"  Replicate count distribution:")
        rep_counts = dup_grouped['count'].value_counts().sort_index()
        for count, freq in rep_counts.items():
            print(f"    {count} replicates: {freq} sequences")
        
        mean_cv = dup_grouped['cv'].mean()
        print(f"  Mean CV for replicates: {mean_cv:.3f}")
    
    # Calculate statistics for all groups
    grouped = df.groupby(['VH', 'VL'])
    group_stats = grouped.agg({
        'KD': ['count', 'mean', 'std', 'min', 'max', list]
    }).reset_index()
    group_stats.columns = ['VH', 'VL', 'count', 'mean', 'std', 'min', 'max', 'kd_values']
    
    # Calculate CV (set to 0 for single measurements)
    group_stats['cv'] = group_stats['std'] / group_stats['mean']
    group_stats['cv'] = group_stats['cv'].fillna(0)
    
    # Filter by CV threshold
    reliable_groups = group_stats[group_stats['cv'] <= cv_threshold].copy()
    unreliable_groups = group_stats[group_stats['cv'] > cv_threshold]
    
    if verbose:
        print(f"\nQuality filtering (CV threshold = {cv_threshold}):")
        print(f"  Reliable sequences: {len(reliable_groups)}")
        print(f"  High-variance sequences removed: {len(unreliable_groups)}")
        
        if len(unreliable_groups) > 0:
            print(f"  Mean CV of removed sequences: {unreliable_groups['cv'].mean():.3f}")
            print(f"  Max CV of removed sequences: {unreliable_groups['cv'].max():.3f}")
    
    # Calculate geometric mean KD for reliable groups
    reliable_groups['KD'] = reliable_groups['kd_values'].apply(calculate_geometric_mean)
    
    # Calculate additional metrics
    reliable_groups['binding_score'] = -np.log10(reliable_groups['KD'] * 1e-9)
    reliable_groups['replicate_count'] = reliable_groups['count']
    reliable_groups['measurement_cv'] = reliable_groups['cv']
    
    # Select final columns
    final_df = reliable_groups[['VH', 'VL', 'KD', 'binding_score', 
                                'replicate_count', 'measurement_cv']].copy()
    
    # Add dataset metadata
    if dataset_name:
        final_df['dataset'] = dataset_name
    
    # Sort by KD for consistent ordering
    final_df = final_df.sort_values('KD').reset_index(drop=True)
    
    if verbose:
        print(f"\nFinal preprocessed data:")
        print(f"  Unique sequences: {len(final_df)}")
        print(f"  KD range: {final_df['KD'].min():.2f} - {final_df['KD'].max():.2f} nM")
        print(f"  Binding score range: {final_df['binding_score'].min():.2f} - {final_df['binding_score'].max():.2f}")
        
        # Affinity distribution
        bins = [0, 10, 100, 1000, np.inf]
        labels = ['High (<10 nM)', 'Medium (10-100 nM)', 'Low (100-1000 nM)', 'Very Low (>1000 nM)']
        final_df['affinity_category'] = pd.cut(final_df['KD'], bins=bins, labels=labels)
        
        print(f"\n  Affinity distribution:")
        for category in labels:
            count = (final_df['affinity_category'] == category).sum()
            pct = count / len(final_df) * 100
            print(f"    {category}: {count} ({pct:.1f}%)")
        
        # Remove temporary column
        final_df = final_df.drop('affinity_category', axis=1)
    
    # Save output
    if output_path:
        final_df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n✅ Saved preprocessed data to: {output_path}")
    
    return final_df

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MAGMA-seq antibody binding data"
    )
    parser.add_argument(
        "input_path",
        help="Path to input CSV file with VH, VL, KD columns"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: adds '_preprocessed' to input filename)"
    )
    parser.add_argument(
        "-c", "--cv-threshold",
        type=float,
        default=0.5,
        help="CV threshold for filtering (default: 0.5)"
    )
    parser.add_argument(
        "-n", "--name",
        help="Dataset name for identification"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input_path)
        output_path = input_path.parent / f"{input_path.stem}_preprocessed.csv"
    else:
        output_path = args.output
    
    # Run preprocessing
    preprocess_magma_seq(
        args.input_path,
        output_path,
        cv_threshold=args.cv_threshold,
        dataset_name=args.name,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()