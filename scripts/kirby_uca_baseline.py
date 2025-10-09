#!/usr/bin/env python3
"""
Create UCA-based partitions using actual UCA sequences from Gene Blocks as baselines.

This approach uses the parsed WT sequences (including UCAs) as reference points
to better model natural affinity maturation trajectories.
"""

import os
import sys
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dnsmex.local import localify
from dnsmex.dms_helper import protein_differences


def load_data():
    """Load annotated variants and parsed WT sequences."""
    
    # Load annotated variants
    annotated_path = localify("DATA_DIR/whitehead/kirby/annotated_variants.csv")
    variants_df = pd.read_csv(annotated_path)
    
    # Load parsed WT sequences (updated with properly translated sequences)
    # Use local data directory to keep ~/data pristine
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt_path = os.path.join(script_dir, "data", "whitehead", "kirby", "parsed_wt_sequences_updated.csv.gz")
    wt_df = pd.read_csv(wt_path)
    
    print(f"Loaded {len(variants_df)} variants and {len(wt_df)} WT sequences")
    
    return variants_df, wt_df


def load_deduplicated_data():
    """Load deduplicated FLAB data and parsed WT sequences."""
    
    # Load deduplicated FLAB data
    flab_path = localify("DATA_DIR/whitehead/kirby/Kirby_PNAS2025_FLAB_deduplicated.csv")
    if not os.path.exists(flab_path):
        print("Deduplicated FLAB file not found. Run deduplicate_kirby_measurements.py first.")
        sys.exit(1)
    
    variants_df = pd.read_csv(flab_path)
    
    # Load parsed WT sequences (updated with properly translated sequences)
    # Use local data directory to keep ~/data pristine
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wt_path = os.path.join(script_dir, "data", "whitehead", "kirby", "parsed_wt_sequences_updated.csv.gz")
    wt_df = pd.read_csv(wt_path)
    
    print(f"Loaded {len(variants_df)} deduplicated variants and {len(wt_df)} WT sequences")
    
    return variants_df, wt_df


def create_wt_lookup(wt_df):
    """Create lookup for WT sequences by antibody and chain."""
    
    wt_lookup = {}
    
    for _, row in wt_df.iterrows():
        antibody = row['antibody']
        chain_type = row['chain_type']
        aa_sequence = row['amino_acid_sequence']
        is_uca = row['is_uca']
        
        if antibody not in wt_lookup:
            wt_lookup[antibody] = {}
        
        # Store both UCA and mature sequences
        # Standardize chain type naming
        std_chain_type = 'heavy' if chain_type in ['vh', 'VH', 'heavy'] else 'light'
        
        if is_uca:
            wt_lookup[antibody][f'{std_chain_type}_uca'] = aa_sequence
        else:
            wt_lookup[antibody][std_chain_type] = aa_sequence
    
    return wt_lookup


def find_best_uca_reference(variant_antibody, wt_lookup):
    """Find the best UCA reference sequence for a variant."""
    
    # First try exact match
    if variant_antibody in wt_lookup:
        antibody_seqs = wt_lookup[variant_antibody]
        
        # Prefer UCA sequences if available
        heavy_ref = antibody_seqs.get('heavy_uca', antibody_seqs.get('heavy'))
        light_ref = antibody_seqs.get('light_uca', antibody_seqs.get('light'))
        
        if heavy_ref and light_ref:
            return variant_antibody, heavy_ref, light_ref
    
    # If no exact match, try to find the corresponding UCA
    # e.g., if variant is "002-S21F2", look for "UCA_002-S21F2"
    uca_name = f"UCA_{variant_antibody.replace('UCA_', '')}"
    if uca_name in wt_lookup:
        antibody_seqs = wt_lookup[uca_name]
        heavy_ref = antibody_seqs.get('heavy_uca', antibody_seqs.get('heavy'))
        light_ref = antibody_seqs.get('light_uca', antibody_seqs.get('light'))
        
        if heavy_ref and light_ref:
            return uca_name, heavy_ref, light_ref
    
    # If still no match, try the reverse (mature for UCA variants)
    mature_name = variant_antibody.replace('UCA_', '')
    if mature_name in wt_lookup:
        antibody_seqs = wt_lookup[mature_name]
        heavy_ref = antibody_seqs.get('heavy_uca', antibody_seqs.get('heavy'))
        light_ref = antibody_seqs.get('light_uca', antibody_seqs.get('light'))
        
        if heavy_ref and light_ref:
            return mature_name, heavy_ref, light_ref
    
    return None, None, None


def trim_reference_to_match_variants(ref_seq, variant_seqs):
    """Trim reference sequence to match variant sequence lengths."""
    
    if not ref_seq or not variant_seqs:
        return ref_seq
    
    # Get variant length (should be consistent within a group)
    variant_length = len(variant_seqs[0])
    
    # If reference is already the right length, return as-is
    if len(ref_seq) == variant_length:
        return ref_seq
    
    # If reference is longer, trim from the end (C-terminus)
    if len(ref_seq) > variant_length:
        return ref_seq[:variant_length]
    
    # If reference is shorter, extend it with the consensus ending from variants
    if len(ref_seq) < variant_length:
        print(f"  Warning: Reference sequence ({len(ref_seq)}) shorter than variants ({variant_length})")
        
        # Try to extend the reference by finding the consensus ending from variants
        extra_length = variant_length - len(ref_seq)
        
        # Get the common ending from variants
        variant_endings = [seq[-extra_length:] for seq in variant_seqs]
        
        # Check if all variants have the same ending
        if len(set(variant_endings)) == 1:
            consensus_ending = variant_endings[0]
            extended_ref = ref_seq + consensus_ending
            print(f"  Extended reference with consensus ending: {consensus_ending}")
            return extended_ref
        else:
            print(f"  No consensus ending found among variants")
            return ref_seq
    
    return ref_seq


def find_best_alignment_offset(ref_seq, variant_seq):
    """Find the best alignment offset between reference and variant sequences."""
    from difflib import SequenceMatcher
    
    if not ref_seq or not variant_seq:
        return 0, 0
    
    best_score = 0
    best_ref_offset = 0
    best_var_offset = 0
    
    # Try different alignment offsets, including cases where sequences are same length
    # but have different starts/ends
    max_offset = min(5, max(len(ref_seq), len(variant_seq)))
    
    for ref_offset in range(max_offset + 1):
        for var_offset in range(max_offset + 1):
            # Get aligned portions
            ref_aligned = ref_seq[ref_offset:]
            var_aligned = variant_seq[var_offset:]
            
            # Calculate similarity on the overlapping region
            min_len = min(len(ref_aligned), len(var_aligned))
            if min_len > 0:
                score = SequenceMatcher(None, ref_aligned[:min_len], var_aligned[:min_len]).ratio()
                
                if score > best_score:
                    best_score = score
                    best_ref_offset = ref_offset
                    best_var_offset = var_offset
    
    return best_ref_offset, best_var_offset


def assign_variants_to_antibodies(variants_df, wt_lookup, similarity_threshold=0.92, uca_only=False):
    """Assign variants to antibodies based on sequence similarity."""
    
    print(f"\nAssigning variants to antibodies (similarity threshold: {similarity_threshold}, uca_only: {uca_only})...")
    
    # Add antibody assignment columns
    variants_df['antibody'] = None
    variants_df['heavy_similarity'] = 0.0
    variants_df['light_similarity'] = 0.0
    variants_df['assignment_confidence'] = 'unassigned'
    
    from difflib import SequenceMatcher
    
    # Filter antibodies to only UCA ones if uca_only is True
    if uca_only:
        filtered_lookup = {k: v for k, v in wt_lookup.items() if k.startswith('UCA_')}
        print(f"  UCA-only mode: using {len(filtered_lookup)} UCA antibodies out of {len(wt_lookup)} total")
    else:
        filtered_lookup = wt_lookup
    
    for idx, variant in variants_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing variant {idx}/{len(variants_df)}")
        
        variant_heavy = variant['VH']
        variant_light = variant['VL']
        
        best_antibody = None
        best_heavy_sim = 0.0
        best_light_sim = 0.0
        best_confidence = 'unassigned'
        
        # Compare against filtered antibodies
        for antibody, seqs in filtered_lookup.items():
            if 'heavy' not in seqs or 'light' not in seqs:
                continue
            
            # Calculate similarity for both chains
            heavy_sim = SequenceMatcher(None, variant_heavy, seqs['heavy']).ratio()
            light_sim = SequenceMatcher(None, variant_light, seqs['light']).ratio()
            
            # Only consider this a potential match if at least one chain meets threshold
            if heavy_sim >= similarity_threshold or light_sim >= similarity_threshold:
                # Check if this is the best match
                avg_sim = (heavy_sim + light_sim) / 2
                if avg_sim > (best_heavy_sim + best_light_sim) / 2:
                    best_antibody = antibody
                    best_heavy_sim = heavy_sim
                    best_light_sim = light_sim
                    
                    # Determine confidence
                    if heavy_sim >= similarity_threshold and light_sim >= similarity_threshold:
                        best_confidence = 'both_chains_match'
                    elif heavy_sim >= similarity_threshold:
                        best_confidence = 'heavy_chain_best'
                    elif light_sim >= similarity_threshold:
                        best_confidence = 'light_chain_best'
                    else:
                        best_confidence = 'low_similarity'
        
        # Only assign if we meet the threshold
        if best_heavy_sim >= similarity_threshold or best_light_sim >= similarity_threshold:
            variants_df.loc[idx, 'antibody'] = best_antibody
            variants_df.loc[idx, 'heavy_similarity'] = best_heavy_sim
            variants_df.loc[idx, 'light_similarity'] = best_light_sim
            variants_df.loc[idx, 'assignment_confidence'] = best_confidence
    
    # Report assignment results
    assigned_count = variants_df['antibody'].notna().sum()
    print(f"  Assigned {assigned_count}/{len(variants_df)} variants to antibodies")
    
    assignment_counts = variants_df['antibody'].value_counts()
    print(f"  Assignment counts: {dict(assignment_counts)}")
    
    return variants_df


def create_uca_baseline_partitions(variants_df, wt_lookup, uca_only=False, similarity_threshold=0.92):
    """Create partitions using UCA sequences as baselines."""
    
    print(f"\nCreating UCA baseline partitions (uca_only: {uca_only}, threshold: {similarity_threshold})...")
    
    # First assign variants to antibodies by similarity
    variants_df = assign_variants_to_antibodies(variants_df, wt_lookup, similarity_threshold=similarity_threshold, uca_only=uca_only)
    
    # Merge non-UCA groups into their corresponding UCA groups
    print("\nMerging non-UCA groups into corresponding UCA groups...")
    antibody_mapping = {}
    all_antibodies = set(variants_df['antibody'].unique())
    
    for antibody in all_antibodies:
        if antibody is not None and not antibody.startswith('UCA_'):
            uca_version = f'UCA_{antibody}'
            if uca_version in all_antibodies:
                antibody_mapping[antibody] = uca_version
                print(f"  {antibody} -> {uca_version}")
            else:
                print(f"  {antibody} has no UCA version, keeping as is")
    
    # Apply the mapping
    variants_df['original_antibody'] = variants_df['antibody'].copy()
    variants_df['antibody'] = variants_df['antibody'].replace(antibody_mapping)
    
    # Get main antibody groups (≥30 variants for correlation analysis)
    antibody_counts = variants_df['antibody'].value_counts()
    print("\nAntibody counts after merging:", antibody_counts)
    main_antibodies = antibody_counts[antibody_counts >= 30].index.tolist()
    
    partitions = {}
    
    for antibody in main_antibodies:
        print(f"\nProcessing {antibody}...")
        
        # Get variants for this antibody
        antibody_variants = variants_df[variants_df['antibody'] == antibody].copy()
        
        # Find best UCA reference
        ref_antibody, heavy_ref, light_ref = find_best_uca_reference(antibody, wt_lookup)
        
        if ref_antibody is None:
            print(f"  No UCA reference found for {antibody}, skipping...")
            continue
        
        print(f"  Using {ref_antibody} as reference")
        print(f"  Heavy reference length: {len(heavy_ref) if heavy_ref else 'N/A'}")
        print(f"  Light reference length: {len(light_ref) if light_ref else 'N/A'}")
        
        # Find best alignment between reference and variants
        sample_heavy = antibody_variants['VH'].iloc[0]
        sample_light = antibody_variants['VL'].iloc[0]
        
        heavy_ref_offset, heavy_var_offset = find_best_alignment_offset(heavy_ref, sample_heavy)
        light_ref_offset, light_var_offset = find_best_alignment_offset(light_ref, sample_light)
        
        # Apply alignment offsets
        heavy_ref_aligned = heavy_ref[heavy_ref_offset:]
        light_ref_aligned = light_ref[light_ref_offset:]
        
        if heavy_var_offset > 0:
            print(f"  Trimming first {heavy_var_offset} AA from heavy variants")
            antibody_variants['VH'] = antibody_variants['VH'].str[heavy_var_offset:]
        if light_var_offset > 0:
            print(f"  Trimming first {light_var_offset} AA from light variants")
            antibody_variants['VL'] = antibody_variants['VL'].str[light_var_offset:]
        
        # Trim reference sequences to match variant lengths after alignment
        heavy_sequences = antibody_variants['VH'].tolist()
        light_sequences = antibody_variants['VL'].tolist()
        
        heavy_ref_trimmed = trim_reference_to_match_variants(heavy_ref_aligned, heavy_sequences)
        light_ref_trimmed = trim_reference_to_match_variants(light_ref_aligned, light_sequences)
        
        # Check for and handle duplicates created by alignment/trimming
        initial_count = len(antibody_variants)
        duplicates = antibody_variants[antibody_variants.duplicated(subset=['VH', 'VL'], keep=False)]
        if len(duplicates) > 0:
            print(f"  Found {len(duplicates)} duplicate VH/VL pairs after alignment - aggregating...")
            
            # Aggregate duplicates by taking geometric mean of KD values
            def geometric_mean_kd(kd_values):
                """Calculate geometric mean of KD values (antilog of mean log10)"""
                if len(kd_values) == 1:
                    return kd_values.iloc[0]
                log_kd = np.log10(kd_values)
                mean_log_kd = np.mean(log_kd)
                return 10 ** mean_log_kd
            
            antibody_variants = antibody_variants.groupby(['VH', 'VL']).agg({
                'KD': geometric_mean_kd
            }).reset_index()
            
            print(f"  Aggregated to {len(antibody_variants)} unique VH/VL pairs ({initial_count - len(antibody_variants)} duplicates removed)")
        else:
            print(f"  No duplicates found after alignment")

        print(f"  Heavy alignment: ref_offset={heavy_ref_offset}, var_offset={heavy_var_offset}")
        print(f"  Light alignment: ref_offset={light_ref_offset}, var_offset={light_var_offset}")
        print(f"  Final heavy reference: {len(heavy_ref)} -> {len(heavy_ref_trimmed)}")
        print(f"  Final light reference: {len(light_ref)} -> {len(light_ref_trimmed)}")
        
        # Add reference sequences to dataframe
        antibody_variants['WT_VH'] = heavy_ref_trimmed
        antibody_variants['WT_VL'] = light_ref_trimmed
        antibody_variants['reference_antibody'] = ref_antibody
        
        # Calculate differences from UCA reference
        antibody_variants['heavy_differences'] = antibody_variants['VH'].apply(
            lambda x: protein_differences(heavy_ref_trimmed, x)
        )
        antibody_variants['light_differences'] = antibody_variants['VL'].apply(
            lambda x: protein_differences(light_ref_trimmed, x)
        )
        
        # Count differences
        antibody_variants['heavy_difference_count'] = antibody_variants['heavy_differences'].apply(len)
        antibody_variants['light_difference_count'] = antibody_variants['light_differences'].apply(len)
        antibody_variants['total_difference_count'] = (
            antibody_variants['heavy_difference_count'] + 
            antibody_variants['light_difference_count']
        )
        
        # Calculate similarities to UCA reference
        antibody_variants['VH_similarity_to_uca'] = antibody_variants['VH'].apply(
            lambda x: SequenceMatcher(None, x, heavy_ref_trimmed).ratio()
        )
        antibody_variants['VL_similarity_to_uca'] = antibody_variants['VL'].apply(
            lambda x: SequenceMatcher(None, x, light_ref_trimmed).ratio()
        )
        
        # Convert KD to -log10 KD
        if 'KD' in antibody_variants.columns:
            antibody_variants['molar_KD'] = antibody_variants['KD'] * 1e-9
            antibody_variants['-log10_KD'] = -np.log10(antibody_variants['molar_KD'])
        
        partitions[antibody] = {
            'df': antibody_variants,
            'reference_antibody': ref_antibody,
            'heavy_ref': heavy_ref_trimmed,
            'light_ref': light_ref_trimmed,
            'variant_count': len(antibody_variants)
        }
        
        print(f"  Created partition with {len(antibody_variants)} variants")
        print(f"  Average similarity to UCA: H={antibody_variants['VH_similarity_to_uca'].mean():.3f}, L={antibody_variants['VL_similarity_to_uca'].mean():.3f}")
    
    return partitions


def save_uca_partitions(partitions):
    """Save UCA baseline partitions to files."""
    
    print(f"\nSaving UCA baseline partitions...")
    
    # Create output directory (use local data to keep ~/data pristine)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(script_dir, "_output")
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    
    for antibody, partition in partitions.items():
        # Save partition
        output_path = f"{output_dir}/{antibody}.csv"
        partition['df'].to_csv(output_path, index=False)
        
        summary_data.append({
            'antibody': antibody,
            'variant_count': partition['variant_count'],
            'reference_antibody': partition['reference_antibody'],
            'heavy_ref_length': len(partition['heavy_ref']),
            'light_ref_length': len(partition['light_ref']),
            'avg_heavy_similarity': partition['df']['VH_similarity_to_uca'].mean(),
            'avg_light_similarity': partition['df']['VL_similarity_to_uca'].mean(),
            'file': output_path
        })
        
        print(f"  {antibody}: {partition['variant_count']} variants -> {output_path}")
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{output_dir}/uca_baseline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nSaved summary: {summary_path}")
    
    return summary_df


def analyze_maturation_trajectories(partitions):
    """Analyze affinity maturation trajectories."""
    
    print(f"\nAnalyzing affinity maturation trajectories...")
    
    for antibody, partition in partitions.items():
        df = partition['df']
        
        print(f"\n{antibody}:")
        print(f"  Reference: {partition['reference_antibody']}")
        print(f"  Variants: {len(df)}")
        
        # Analyze by number of mutations
        mutation_groups = df.groupby('total_difference_count').agg({
            'KD': ['count', 'mean', 'std'],
            '-log10_KD': ['mean', 'std']
        }).round(3)
        
        print(f"  Affinity by mutation count:")
        for mutations, group in mutation_groups.iterrows():
            count = group[('KD', 'count')]
            mean_kd = group[('KD', 'mean')]
            mean_log_kd = group[('-log10_KD', 'mean')]
            print(f"    {mutations} mutations: {count} variants, mean KD = {mean_kd:.1f} nM, -log10 KD = {mean_log_kd:.2f}")


def main(uca_only=False, similarity_threshold=0.92, use_deduplicated=True):
    """Main function to create UCA baseline partitions."""
    
    print(f"Loading data (uca_only: {uca_only}, threshold: {similarity_threshold}, deduplicated: {use_deduplicated})...")
    if use_deduplicated:
        variants_df, wt_df = load_deduplicated_data()
    else:
        variants_df, wt_df = load_data()
    
    print("\nCreating WT lookup...")
    wt_lookup = create_wt_lookup(wt_df)
    
    print(f"Available antibodies in WT lookup: {list(wt_lookup.keys())}")
    
    print("\nCreating UCA baseline partitions...")
    partitions = create_uca_baseline_partitions(variants_df, wt_lookup, uca_only=uca_only, similarity_threshold=similarity_threshold)
    
    print("\nSaving partitions...")
    summary_df = save_uca_partitions(partitions)
    
    print("\nAnalyzing maturation trajectories...")
    analyze_maturation_trajectories(partitions)
    
    print(f"\n✅ UCA baseline partitioning complete!")
    print(f"Created {len(partitions)} partitions using UCA references")
    
    return partitions, summary_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create UCA baseline partitions for Kirby data')
    parser.add_argument('--uca-only', action='store_true', help='Only match variants to UCA sequences')
    parser.add_argument('--similarity-threshold', type=float, default=0.92, help='Similarity threshold for variant assignment (default: 0.92)')
    parser.add_argument('--no-deduplicated', action='store_true', help='Use original data instead of deduplicated version')
    args = parser.parse_args()
    
    main(uca_only=args.uca_only, similarity_threshold=args.similarity_threshold, use_deduplicated=not args.no_deduplicated)