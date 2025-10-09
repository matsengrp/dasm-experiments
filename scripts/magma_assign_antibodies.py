#!/usr/bin/env python3
"""
Unified MAGMA-seq antibody assignment script using Dr. Whitehead's authoritative reference sequences.

This script processes both Kirby and Petersen MAGMA-seq datasets and assigns sequences
to specific antibody systems based on similarity to authoritative reference sequences
provided by Dr. Whitehead.

Usage:
    python scripts/magma_assign_antibodies.py --dataset kirby
    python scripts/magma_assign_antibodies.py --dataset petersen
    python scripts/magma_assign_antibodies.py --dataset both
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import difflib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)


def load_reference_sequences(reference_file_path):
    """Load reference sequences from Dr. Whitehead's AntibodySequences files."""
    print(f"🧬 Loading reference sequences from {reference_file_path}...")
    
    if not os.path.exists(reference_file_path):
        raise FileNotFoundError(f"Reference file not found: {reference_file_path}")
    
    df = pd.read_csv(reference_file_path)
    references = {}
    
    for _, row in df.iterrows():
        antibody = row['Antibody']
        chain = row['Chain'] 
        sequence = row['Sequence']
        
        if antibody not in references:
            references[antibody] = {}
        references[antibody][chain] = sequence
    
    print(f"📊 Loaded {len(references)} antibody systems:")
    for antibody, chains in references.items():
        chain_types = list(chains.keys())
        print(f"  ✅ {antibody}: {', '.join(chain_types)}")
    
    return references


def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """Calculate percentage similarity between two amino acid sequences."""
    if len(seq1) != len(seq2):
        # Use difflib for sequences of different lengths
        matcher = difflib.SequenceMatcher(None, seq1, seq2)
        return matcher.ratio()
    else:
        # Simple identity for same-length sequences
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)


def load_flab_data(dataset_name: str) -> pd.DataFrame:
    """Load FLAB data for the specified dataset."""
    
    if dataset_name.lower() == 'kirby':
        flab_path = "data/whitehead/kirby/original/Kirby_PNAS2025_FLAB_filtered_preprocessed.csv"
        paper = "Kirby et al. 2025"
    elif dataset_name.lower() == 'petersen':
        flab_path = "data/whitehead/petersen/original/MAGMAseq_anchor_FLAB_preprocessed.csv"
        paper = "Petersen et al. 2024"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(flab_path):
        raise FileNotFoundError(f"FLAB data not found: {flab_path}")
    
    df = pd.read_csv(flab_path)
    df['dataset'] = dataset_name.title()
    df['paper'] = paper
    df['method'] = 'MAGMA-seq'
    df['sequence_id'] = df['VH'] + '|' + df['VL']
    
    print(f"📊 Loaded {len(df)} {dataset_name} MAGMA-seq sequences")
    return df


def assign_sequences_to_antibodies(flab_df: pd.DataFrame, references: dict, dataset_name: str) -> pd.DataFrame:
    """Assign FLAB sequences to specific antibody systems based on reference similarity."""
    
    print(f"🔍 Assigning {dataset_name} sequences to antibodies...")
    
    # Set similarity threshold based on dataset
    min_similarity_threshold = 0.85  # Higher threshold for authoritative sequences
    
    assignments = []
    unassigned_count = 0
    
    for idx, row in flab_df.iterrows():
        vh_seq = row['VH']
        vl_seq = row['VL']
        
        best_antibody = None
        best_score = 0
        antibody_scores = {}
        
        # Compare against each antibody's reference sequences
        for antibody, chains in references.items():
            if 'VH' in chains and 'VL' in chains:
                vh_similarity = calculate_sequence_similarity(vh_seq, chains['VH'])
                vl_similarity = calculate_sequence_similarity(vl_seq, chains['VL'])
                
                # Combined score (average of VH and VL similarity)
                combined_score = (vh_similarity + vl_similarity) / 2
                antibody_scores[antibody] = combined_score
                
                if combined_score > best_score and combined_score >= min_similarity_threshold:
                    best_score = combined_score
                    best_antibody = antibody
        
        # If no antibody meets the threshold, leave unassigned
        if best_antibody is None:
            unassigned_count += 1
            best_antibody = "UNASSIGNED"
            best_score = max(antibody_scores.values()) if antibody_scores else 0
        
        assignments.append({
            'sequence_id': row['sequence_id'],
            'VH': vh_seq,
            'VL': vl_seq,
            'KD': row['KD'],
            'assigned_antibody': best_antibody,
            'similarity_score': best_score,
            'all_scores': antibody_scores
        })
    
    # Convert to DataFrame
    assigned_df = pd.DataFrame(assignments)
    
    # Report assignment statistics
    print("📊 Assignment results:")
    assignment_counts = assigned_df['assigned_antibody'].value_counts()
    for antibody, count in assignment_counts.items():
        print(f"  {antibody}: {count} sequences")
    
    if unassigned_count > 0:
        print(f"⚠️  {unassigned_count} sequences below similarity threshold ({min_similarity_threshold:.1%})")
    
    print("\n🎯 Similarity score statistics:")
    for antibody in list(references.keys()) + ["UNASSIGNED"]:
        ab_scores = assigned_df[assigned_df['assigned_antibody'] == antibody]['similarity_score']
        if len(ab_scores) > 0:
            print(f"  {antibody}: mean={ab_scores.mean():.3f}, min={ab_scores.min():.3f}, max={ab_scores.max():.3f}")
    
    return assigned_df


def create_antibody_partitions(assigned_df: pd.DataFrame, references: dict, dataset_name: str):
    """Create antibody partition files for each assigned system."""
    
    # Create output directory
    if dataset_name.lower() == 'kirby':
        output_dir = Path("data/whitehead/kirby/processed")
    else:
        output_dir = Path("data/whitehead/petersen/processed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Creating {dataset_name} partition files...")
    
    for antibody in references.keys():
        antibody_sequences = assigned_df[
            (assigned_df['assigned_antibody'] == antibody) & 
            (assigned_df['assigned_antibody'] != 'UNASSIGNED')
        ].copy()
        
        if len(antibody_sequences) == 0:
            print(f"  ⚠️  No sequences assigned to {antibody}, skipping...")
            continue
        
        print(f"  📄 Creating partition for {antibody} ({len(antibody_sequences)} sequences)...")
        
        # Get reference sequences
        ref_vh = references[antibody]['VH']
        ref_vl = references[antibody]['VL']
        
        # Calculate mutations from reference for each sequence
        mutations_from_uca = []
        heavy_differences = []
        light_differences = []
        
        for idx, row in antibody_sequences.iterrows():
            # Count mutations in VH
            vh_mutations = []
            vh_mut_count = 0
            for i, (seq_aa, ref_aa) in enumerate(zip(row['VH'], ref_vh)):
                if seq_aa != ref_aa:
                    vh_mutations.append(f"{ref_aa}{i+1}{seq_aa}")
                    vh_mut_count += 1
            
            # Count mutations in VL  
            vl_mutations = []
            vl_mut_count = 0
            for i, (seq_aa, ref_aa) in enumerate(zip(row['VL'], ref_vl)):
                if seq_aa != ref_aa:
                    vl_mutations.append(f"{ref_aa}{i+1}{seq_aa}")
                    vl_mut_count += 1
            
            total_mutations = vh_mut_count + vl_mut_count
            mutations_from_uca.append(total_mutations)
            heavy_differences.append(str(vh_mutations))
            light_differences.append(str(vl_mutations))
        
        antibody_sequences['mutations_from_uca'] = mutations_from_uca
        antibody_sequences['total_difference_count'] = mutations_from_uca
        antibody_sequences['heavy_differences'] = heavy_differences
        antibody_sequences['light_differences'] = light_differences
        
        # Add reference row (UCA/WT)
        # Use 1500 nM for UCA KD as per Kirby paper: "UCA sequences had greater than 1 µM initial binding affinity"
        uca_kd = 1500.0  # nM, representing >1000 nM (>1 μM) from paper
        ref_row = pd.DataFrame([{
            'sequence_id': f"{ref_vh}|{ref_vl}",
            'VH': ref_vh,
            'VL': ref_vl,
            'KD': uca_kd,
            'assigned_antibody': antibody,
            'similarity_score': 1.0,
            'mutations_from_uca': 0,
            'total_difference_count': 0,
            'heavy_differences': "[]",
            'light_differences': "[]",
            'WT_VH': ref_vh,
            'WT_VL': ref_vl,
            'antibody': antibody,
            'dataset': dataset_name.title(),
            'is_uca': True
        }])
        
        # Add metadata columns to match existing format
        antibody_sequences['WT_VH'] = ref_vh
        antibody_sequences['WT_VL'] = ref_vl
        antibody_sequences['antibody'] = antibody
        antibody_sequences['dataset'] = dataset_name.title()
        antibody_sequences['is_uca'] = False
        
        # Combine reference and mutant sequences
        full_partition = pd.concat([ref_row, antibody_sequences], ignore_index=True)
        
        # Save partition file
        partition_filename = f"{antibody}_partition.csv"
        partition_path = output_dir / partition_filename
        full_partition.to_csv(partition_path, index=False)
        
        print(f"    ✅ Saved {len(full_partition)} sequences to {partition_path}")
        print(f"       Mutations range: {full_partition['mutations_from_uca'].min()}-{full_partition['mutations_from_uca'].max()}")
        
        # Print sample data
        sample = full_partition[['antibody', 'mutations_from_uca', 'KD', 'is_uca']].head(3)
        print(f"       Sample:\n{sample.to_string(index=False)}")


def process_dataset(dataset_name: str):
    """Process a single dataset (kirby or petersen)."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Determine file paths
    if dataset_name.lower() == 'kirby':
        reference_file = "data/whitehead/kirby/original/AntibodySequences1.csv"
    elif dataset_name.lower() == 'petersen':
        reference_file = "data/whitehead/petersen/original/AntibodySequences2.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load reference sequences
    references = load_reference_sequences(reference_file)
    
    # Load FLAB data
    flab_df = load_flab_data(dataset_name)
    
    # Assign sequences
    assigned_df = assign_sequences_to_antibodies(flab_df, references, dataset_name)
    
    # Create partition files
    create_antibody_partitions(assigned_df, references, dataset_name)
    
    print(f"\n✅ {dataset_name.title()} assignment complete!")
    assigned_systems = assigned_df[assigned_df['assigned_antibody'] != 'UNASSIGNED']['assigned_antibody'].nunique()
    print(f"Created partitions for {assigned_systems} antibody systems")


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Assign MAGMA-seq sequences to antibody systems using authoritative references')
    parser.add_argument('--dataset', choices=['kirby', 'petersen', 'both'], default='both',
                       help='Dataset to process (default: both)')
    
    args = parser.parse_args()
    
    print("=== MAGMA ANTIBODY ASSIGNMENT ===")
    print("Using Dr. Whitehead's authoritative reference sequences")
    print(f"Processing: {args.dataset}")
    
    try:
        if args.dataset == 'both':
            process_dataset('kirby')
            process_dataset('petersen')
        else:
            process_dataset(args.dataset)
        
        print(f"\n🎉 Assignment pipeline complete!")
        print("📁 Partition files created in data/whitehead/*/processed/")
        print("📊 Ready for mutation tree visualization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()