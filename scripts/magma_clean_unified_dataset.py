#!/usr/bin/env python3
"""
Clean the unified MAGMA dataset by removing invalid amino acids and aggregating replicates.
Based on replicate aggregation logic from kirby_uca_baseline.py.
"""

import csv
import pandas as pd
import numpy as np

def is_valid_amino_acid_sequence(seq):
    """Check if sequence contains only valid amino acids."""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa in valid_aa for aa in seq)

def geometric_mean_kd(kd_values):
    """Calculate geometric mean of KD values (antilog of mean log10)"""
    if len(kd_values) == 1:
        return kd_values.iloc[0]
    log_kd = np.log10(kd_values)
    mean_log_kd = np.mean(log_kd)
    return 10 ** mean_log_kd

def main():
    print("=" * 80)
    print("CLEANING UNIFIED MAGMA DATASET WITH REPLICATE AGGREGATION")
    print("=" * 80)
    
    input_file = "data/whitehead/unified/magma_unified_dataset.csv"
    output_file = "data/whitehead/unified/magma_unified_dataset_clean.csv"
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} sequences")
    
    # Step 1: Remove invalid amino acid sequences
    print("\\nStep 1: Removing invalid amino acid sequences...")
    initial_count = len(df)
    invalid_mask = ~(df['VH'].apply(is_valid_amino_acid_sequence) & 
                     df['VL'].apply(is_valid_amino_acid_sequence))
    
    if invalid_mask.sum() > 0:
        print(f"Removing {invalid_mask.sum()} sequences with invalid amino acids:")
        invalid_sequences = df[invalid_mask]
        for _, seq in invalid_sequences.iterrows():
            print(f"  - {seq['antibody']} KD={seq['KD']}")
        
        df = df[~invalid_mask].copy()
        print(f"Kept {len(df)} valid sequences")
    else:
        print("No invalid sequences found")
    
    # Step 2: Identify and aggregate replicates (same VH/VL pairs)
    print("\\nStep 2: Identifying replicate sequences...")
    
    # Find duplicates (excluding dummy KD=1500 values)
    non_dummy_df = df[df['KD'] != 1500.0].copy()
    dummy_df = df[df['KD'] == 1500.0].copy()
    
    print(f"Non-dummy sequences: {len(non_dummy_df)}")
    print(f"Dummy reference sequences (KD=1500): {len(dummy_df)}")
    
    # Check for duplicate VH/VL pairs in non-dummy data
    duplicate_mask = non_dummy_df.duplicated(subset=['VH', 'VL'], keep=False)
    duplicates = non_dummy_df[duplicate_mask]
    
    if len(duplicates) > 0:
        print(f"\\nFound {len(duplicates)} sequences in {len(duplicates) // 2} replicate pairs:")
        
        # Show replicate pairs
        for (vh, vl), group in duplicates.groupby(['VH', 'VL']):
            antibody = group['antibody'].iloc[0]
            kd_values = group['KD'].tolist()
            print(f"  {antibody}: {len(kd_values)} replicates, KD = {kd_values}")
        
        print("\\nAggregating replicates using geometric mean in log space...")
        
        # Aggregate non-dummy sequences
        aggregated_non_dummy = non_dummy_df.groupby(['VH', 'VL']).agg({
            'sequence_id': 'first',  # Keep first sequence_id
            'KD': geometric_mean_kd,
            'antibody': 'first',
            'dataset': 'first',
            'similarity_score': 'first',
            'mutations_from_reference': 'first',
            'heavy_mutations': 'first',
            'light_mutations': 'first',
            'reference_VH': 'first',
            'reference_VL': 'first',
            'binding_score': lambda x: -geometric_mean_kd(-x),  # Recalculate binding score
            'experimental_design': 'first'
        }).reset_index()
        
        print(f"Aggregated {len(non_dummy_df)} -> {len(aggregated_non_dummy)} unique non-dummy sequences")
        
        # Combine with dummy sequences
        cleaned_df = pd.concat([aggregated_non_dummy, dummy_df], ignore_index=True)
        
    else:
        print("No replicate sequences found - no aggregation needed")
        cleaned_df = df.copy()
    
    print(f"\\nFinal dataset summary:")
    print(f"  Original sequences: {initial_count}")
    print(f"  After removing invalid AA: {len(df)}")
    print(f"  After replicate aggregation: {len(cleaned_df)}")
    print(f"  Success rate: {len(cleaned_df)/initial_count*100:.1f}%")
    
    # Verify no duplicates remain in non-dummy data
    final_non_dummy = cleaned_df[cleaned_df['KD'] != 1500.0]
    final_duplicates = final_non_dummy.duplicated(subset=['VH', 'VL']).sum()
    print(f"  Remaining duplicates in non-dummy data: {final_duplicates}")
    
    # Write cleaned dataset
    print(f"\\nSaving cleaned dataset to {output_file}...")
    cleaned_df.to_csv(output_file, index=False)
    
    print(f"\\n✅ Dataset cleaning complete!")
    print(f"Ready for model scoring with {len(cleaned_df)} sequences")
    
    return cleaned_df

if __name__ == "__main__":
    main()