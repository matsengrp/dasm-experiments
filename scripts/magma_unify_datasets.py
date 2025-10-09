#!/usr/bin/env python3
"""
Unify Kirby and Petersen MAGMA-seq datasets for consistent model scoring.

This script combines all partition files from both datasets into a single
unified dataset with consistent column naming and metadata.
"""

import os
import sys
import csv
from pathlib import Path

def load_partition_file(file_path, dataset_name):
    """Load a partition file and add dataset metadata."""
    sequences = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create unified sequence entry
            seq_entry = {
                'sequence_id': row['sequence_id'],
                'VH': row['VH'],
                'VL': row['VL'], 
                'KD': float(row['KD']),
                'antibody': row['assigned_antibody'],
                'dataset': dataset_name,
                'similarity_score': float(row['similarity_score']),
                'mutations_from_reference': int(row['mutations_from_uca']),  # Rename for clarity
                'heavy_mutations': row.get('heavy_differences', '[]'),
                'light_mutations': row.get('light_differences', '[]'),
                'reference_VH': row['WT_VH'],
                'reference_VL': row['WT_VL'],
                'binding_score': -float(row['KD']) if row['KD'] != '0' else 0,  # -log10(KD) for consistency
                'experimental_design': 'UCA_to_mature' if dataset_name == 'Kirby' else 'mature_CDR_saturation'
            }
            sequences.append(seq_entry)
    
    return sequences

def main():
    print("=" * 80)
    print("UNIFYING KIRBY AND PETERSEN MAGMA-SEQ DATASETS")
    print("=" * 80)
    
    # Define input paths
    kirby_dir = Path("data/whitehead/kirby/processed")
    petersen_dir = Path("data/whitehead/petersen/processed")
    output_dir = Path("data/whitehead/unified")
    output_dir.mkdir(exist_ok=True)
    
    all_sequences = []
    dataset_stats = {}
    
    # Load all Kirby partition files
    print("\\nLoading Kirby datasets...")
    kirby_files = list(kirby_dir.glob("*_partition.csv"))
    print(f"Found {len(kirby_files)} Kirby partition files")
    
    for file_path in kirby_files:
        antibody_name = file_path.stem.replace("_partition", "")
        print(f"  Loading {antibody_name}...")
        
        sequences = load_partition_file(file_path, "Kirby")
        all_sequences.extend(sequences)
        
        dataset_stats[f"Kirby_{antibody_name}"] = {
            'count': len(sequences),
            'kd_range': (min(s['KD'] for s in sequences), max(s['KD'] for s in sequences)),
            'mutation_range': (min(s['mutations_from_reference'] for s in sequences), 
                             max(s['mutations_from_reference'] for s in sequences))
        }
    
    # Load all Petersen partition files  
    print("\\nLoading Petersen datasets...")
    petersen_files = list(petersen_dir.glob("*_partition.csv"))
    print(f"Found {len(petersen_files)} Petersen partition files")
    
    for file_path in petersen_files:
        antibody_name = file_path.stem.replace("_partition", "")
        print(f"  Loading {antibody_name}...")
        
        sequences = load_partition_file(file_path, "Petersen")
        all_sequences.extend(sequences)
        
        dataset_stats[f"Petersen_{antibody_name}"] = {
            'count': len(sequences),
            'kd_range': (min(s['KD'] for s in sequences), max(s['KD'] for s in sequences)),
            'mutation_range': (min(s['mutations_from_reference'] for s in sequences),
                             max(s['mutations_from_reference'] for s in sequences))
        }
    
    print(f"\\n📊 UNIFIED DATASET SUMMARY")
    print("=" * 60)
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Total antibodies: {len(dataset_stats)}")
    
    # Print detailed stats
    kirby_total = sum(stats['count'] for name, stats in dataset_stats.items() if name.startswith('Kirby'))
    petersen_total = sum(stats['count'] for name, stats in dataset_stats.items() if name.startswith('Petersen'))
    
    print(f"\\nDataset breakdown:")
    print(f"  Kirby: {kirby_total} sequences across {len([n for n in dataset_stats if n.startswith('Kirby')])} antibodies")
    print(f"  Petersen: {petersen_total} sequences across {len([n for n in dataset_stats if n.startswith('Petersen')])} antibodies")
    
    print(f"\\nPer-antibody statistics:")
    for name, stats in dataset_stats.items():
        kd_min, kd_max = stats['kd_range']
        mut_min, mut_max = stats['mutation_range']
        print(f"  {name}: {stats['count']} seqs, KD {kd_min:.1f}-{kd_max:.1f} nM, {mut_min}-{mut_max} mutations")
    
    # Write unified dataset
    output_file = output_dir / "magma_unified_dataset.csv"
    
    print(f"\\n💾 Writing unified dataset to: {output_file}")
    
    fieldnames = [
        'sequence_id', 'VH', 'VL', 'KD', 'antibody', 'dataset', 
        'similarity_score', 'mutations_from_reference', 'heavy_mutations', 'light_mutations',
        'reference_VH', 'reference_VL', 'binding_score', 'experimental_design'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sequences)
    
    # Create metadata file
    metadata_file = output_dir / "dataset_metadata.txt"
    
    print(f"💾 Writing metadata to: {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        f.write("MAGMA-seq Unified Dataset Metadata\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Created: {__file__}\\n")
        f.write(f"Total sequences: {len(all_sequences)}\\n")
        f.write(f"Total antibodies: {len(dataset_stats)}\\n\\n")
        
        f.write("Dataset Sources:\\n")
        f.write("- Kirby et al. PNAS 2025: UCA-to-mature evolutionary trajectories\\n")
        f.write("- Petersen et al. Nature Comm 2024: CDR-targeted site-saturation mutagenesis\\n\\n")
        
        f.write("Column Descriptions:\\n")
        f.write("- sequence_id: Unique identifier (VH|VL concatenation)\\n")
        f.write("- VH/VL: Heavy and light chain amino acid sequences\\n")
        f.write("- KD: Binding dissociation constant (nM)\\n")
        f.write("- antibody: Antibody name\\n")
        f.write("- dataset: Kirby or Petersen\\n")
        f.write("- mutations_from_reference: Number of mutations from reference sequence\\n")
        f.write("- experimental_design: UCA_to_mature or mature_CDR_saturation\\n\\n")
        
        f.write("Per-antibody Statistics:\\n")
        for name, stats in dataset_stats.items():
            kd_min, kd_max = stats['kd_range']
            mut_min, mut_max = stats['mutation_range']
            f.write(f"{name}: {stats['count']} sequences, KD {kd_min:.1f}-{kd_max:.1f} nM, {mut_min}-{mut_max} mutations\\n")
    
    print("\\n✅ UNIFICATION COMPLETE!")
    print(f"Ready for model scoring with {len(all_sequences)} sequences")
    print(f"Next step: Run model scoring pipeline on unified dataset")

if __name__ == "__main__":
    main()