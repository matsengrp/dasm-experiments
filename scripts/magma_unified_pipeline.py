#!/usr/bin/env python3
"""
Unified MAGMA-seq preprocessing and scoring pipeline.

PIPELINE STEPS:
1. Load raw Kirby data (find original unprocessed data)
2. Load raw Petersen data  
3. Apply uniform preprocessing (dedup, CV filtering, geometric mean)
4. Apply early filtering (invalid AA, incomplete sequences, length validation)
5. Score all clean sequences with all models (with caching)
6. Merge datasets
7. Final validation and save

This ensures consistent processing and early filtering for clean data.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from netam.framework import load_crepe
from dnsmex.local import localify
from dnsmex.cached_model_wrappers import get_cached_esm_wrapper, get_cached_ablang_wrapper
from scripts.preprocess_magma_seq import preprocess_magma_seq
from scripts.filter_invalid_sequences import filter_dataset

def score_magma_sequences(df, dataset_name):
    """Score all sequences in a dataset with all models using caching."""
    print(f"\n🔬 Scoring {len(df)} {dataset_name} sequences with all models...")
    
    # Copy dataframe to avoid modifying original
    scored_df = df.copy()
    
    # Load DASM models
    print("  Loading DASM models...")
    dasm_crepe = load_crepe(localify("DASM_TRAINED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint"))
    dasm_ft_crepe = load_crepe(localify("DASM_UPDATED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint-UPDATED-v1jaffePairedCC@0.5@v2kimTrain"))
    
    # Score with DASM
    print("  🏠 Scoring with DASM...")
    dasm_scores = []
    for _, row in tqdm(scored_df.iterrows(), total=len(scored_df), desc="DASM"):
        try:
            [[dasm_heavy, dasm_light]] = dasm_crepe([[row['VH'], row['VL']]])
            log_dasm_heavy = torch.log(dasm_heavy).mean().item()
            log_dasm_light = torch.log(dasm_light).mean().item()
            dasm_scores.append(log_dasm_heavy + log_dasm_light)
        except Exception as e:
            dasm_scores.append(np.nan)
    scored_df['dasm'] = dasm_scores
    
    # Score with DASM-FT
    print("  🏠 Scoring with DASM-FT...")
    dasm_ft_scores = []
    for _, row in tqdm(scored_df.iterrows(), total=len(scored_df), desc="DASM-FT"):
        try:
            [[dasm_heavy, dasm_light]] = dasm_ft_crepe([[row['VH'], row['VL']]])
            log_dasm_heavy = torch.log(dasm_heavy).mean().item()
            log_dasm_light = torch.log(dasm_light).mean().item()
            dasm_ft_scores.append(log_dasm_heavy + log_dasm_light)
        except Exception as e:
            dasm_ft_scores.append(np.nan)
    scored_df['dasm_ft'] = dasm_ft_scores
    
    # Score with ESM and AbLang
    sequences = [(row['VH'], row['VL']) for _, row in scored_df.iterrows()]
    
    print("  🚀 Scoring with ESM...")
    try:
        esm_wrapper = get_cached_esm_wrapper(model_size="650M", use_remote=True)
        esm_scores = esm_wrapper.evaluate_antibodies(sequences)
        scored_df['esm'] = esm_scores
    except Exception as e:
        print(f"    ❌ ESM failed: {e}")
        scored_df['esm'] = np.nan
    
    print("  🏠 Scoring with AbLang...")
    try:
        ablang_wrapper = get_cached_ablang_wrapper(use_remote=True)
        ablang_scores = ablang_wrapper.evaluate_antibodies(sequences)
        scored_df['ablang'] = ablang_scores
    except Exception as e:
        print(f"    ❌ AbLang failed: {e}")
        scored_df['ablang'] = np.nan
    
    # Report coverage
    for col in ['dasm', 'dasm_ft', 'esm', 'ablang']:
        valid = scored_df[col].notna().sum()
        print(f"    {col.upper()}: {valid}/{len(scored_df)} ({valid/len(scored_df)*100:.1f}%)")
    
    return scored_df

def main():
    print("=" * 80)
    print("MAGMA-SEQ UNIFIED PIPELINE (KIRBY + PETERSEN)")
    print("=" * 80)
    print("Pipeline steps:")
    print("1. Load raw data")
    print("2. Apply preprocessing (dedup, CV filtering)")
    print("3. Apply early filtering (invalid AA, incomplete sequences)")
    print("4. Score all clean sequences (with caching)")
    print("5. Merge datasets")
    print("6. Final validation and save")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("data/whitehead/merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load raw data
    print("\n1️⃣ LOADING RAW DATA")
    print("-" * 40)
    
    # Kirby: Find original unprocessed data if available, otherwise use deduplicated
    print("Loading Kirby data...")
    kirby_raw_path = localify("DATA_DIR/whitehead/kirby/Kirby_PNAS2025_FLAB_filtered.csv")
    if os.path.exists(kirby_raw_path):
        print(f"  📁 Loading raw Kirby data from: {kirby_raw_path}")
        kirby_raw = pd.read_csv(kirby_raw_path)
    else:
        print("  📁 Raw data not found, using deduplicated version")
        kirby_raw = pd.read_csv("data/whitehead/kirby/Kirby_PNAS2025_FLAB_deduplicated.csv.gz")
        
    kirby_raw['dataset'] = 'Kirby'
    kirby_raw['paper'] = 'Kirby et al. PNAS 2025'
    kirby_raw['method'] = 'MAGMA-seq DMS'
    print(f"  Loaded {len(kirby_raw)} Kirby sequences")
    
    # Petersen: Load raw data
    print("Loading Petersen data...")
    petersen_raw_path = "data/whitehead/petersen/MAGMAseq_anchor_FLAB.csv"
    petersen_raw = pd.read_csv(petersen_raw_path)
    petersen_raw['dataset'] = 'Petersen'
    petersen_raw['paper'] = 'Petersen et al. 2024'
    petersen_raw['method'] = 'MAGMA-seq'
    print(f"  Loaded {len(petersen_raw)} Petersen sequences")
    
    # Step 2: Apply preprocessing
    print("\n2️⃣ APPLYING PREPROCESSING")
    print("-" * 40)
    
    # Kirby preprocessing (if raw data was available)
    if 'FLAB_filtered.csv' in str(kirby_raw_path):
        print("Preprocessing Kirby data...")
        kirby_preprocessed = preprocess_magma_seq(
            kirby_raw_path, 
            output_dir / "kirby_preprocessed.csv",
            dataset_name="Kirby"
        )
        # Add metadata back
        kirby_preprocessed['dataset'] = 'Kirby'
        kirby_preprocessed['paper'] = 'Kirby et al. PNAS 2025'
        kirby_preprocessed['method'] = 'MAGMA-seq DMS'
    else:
        print("Kirby already preprocessed, skipping...")
        kirby_preprocessed = kirby_raw
        kirby_preprocessed['binding_score'] = -np.log10(kirby_preprocessed['KD'] * 1e-9)
    
    # Petersen preprocessing
    print("Preprocessing Petersen data...")
    petersen_preprocessed = preprocess_magma_seq(
        petersen_raw_path,
        output_dir / "petersen_preprocessed.csv", 
        dataset_name="Petersen"
    )
    # Add metadata back
    petersen_preprocessed['dataset'] = 'Petersen'
    petersen_preprocessed['paper'] = 'Petersen et al. 2024'
    petersen_preprocessed['method'] = 'MAGMA-seq'
    
    # Step 3: Apply early filtering
    print("\n3️⃣ APPLYING EARLY FILTERING")
    print("-" * 40)
    
    # Save preprocessed data first, then filter
    print("Filtering kirby data...")
    kirby_preprocessed_path = output_dir / "kirby_preprocessed.csv"
    kirby_preprocessed.to_csv(kirby_preprocessed_path, index=False)
    kirby_filtered_path = output_dir / "kirby_filtered.csv"
    kirby_filtered = filter_dataset(kirby_preprocessed_path, kirby_filtered_path)
    
    print("Filtering petersen data...")
    petersen_preprocessed_path = output_dir / "petersen_preprocessed.csv"
    petersen_preprocessed.to_csv(petersen_preprocessed_path, index=False)
    petersen_filtered_path = output_dir / "petersen_filtered.csv"
    petersen_filtered = filter_dataset(petersen_preprocessed_path, petersen_filtered_path)
    
    print(f"Clean sequences: Kirby={len(kirby_filtered)}, Petersen={len(petersen_filtered)}")
    
    # Step 4: Score all clean sequences
    print("\n4️⃣ SCORING ALL CLEAN SEQUENCES")
    print("-" * 40)
    
    # Score Kirby sequences (with caching)
    kirby_scores_path = "data/whitehead/kirby/kirby_magma_seq_scores.csv"
    if os.path.exists(kirby_scores_path):
        print("🔄 Loading cached Kirby scores...")
        kirby_cached = pd.read_csv(kirby_scores_path)
        kirby_scored = kirby_filtered.merge(
            kirby_cached[['VH', 'VL', 'dasm', 'dasm_ft', 'esm', 'ablang']],
            on=['VH', 'VL'], how='left'
        )
    else:
        print("🔬 Scoring Kirby sequences (no cache)...")
        kirby_scored = score_magma_sequences(kirby_filtered, "Kirby")
        # Cache results
        kirby_scored.to_csv(kirby_scores_path, index=False)
        print(f"  💾 Cached to: {kirby_scores_path}")
    
    # Score Petersen sequences (with caching)
    petersen_scores_path = "data/whitehead/petersen/petersen_magma_seq_scores.csv"
    if os.path.exists(petersen_scores_path):
        print("🔄 Loading cached Petersen scores...")
        petersen_cached = pd.read_csv(petersen_scores_path)
        petersen_scored = petersen_filtered.merge(
            petersen_cached[['VH', 'VL', 'dasm', 'dasm_ft', 'esm', 'ablang']],
            on=['VH', 'VL'], how='left'
        )
    else:
        print("🔬 Scoring Petersen sequences (no cache)...")
        petersen_scored = score_magma_sequences(petersen_filtered, "Petersen") 
        # Cache results
        petersen_scored.to_csv(petersen_scores_path, index=False)
        print(f"  💾 Cached to: {petersen_scores_path}")
    
    # Step 5: Merge datasets
    print("\n5️⃣ MERGING DATASETS")
    print("-" * 40)
    
    # Merge the scored datasets
    merged_df = pd.concat([kirby_scored, petersen_scored], ignore_index=True, sort=False)
    merged_df = merged_df.sort_values(['KD', 'dataset', 'VH', 'VL']).reset_index(drop=True)
    
    print(f"Merged dataset: {len(merged_df)} total sequences")
    print(f"  Kirby: {(merged_df['dataset'] == 'Kirby').sum()}")
    print(f"  Petersen: {(merged_df['dataset'] == 'Petersen').sum()}")
    
    # Step 6: Final validation and save
    print("\n6️⃣ FINAL VALIDATION AND SAVE")
    print("-" * 40)
    
    # Add sequence identifier
    merged_df['sequence_id'] = merged_df['VH'] + '|' + merged_df['VL']
    
    # Check model coverage
    print("Final model coverage:")
    for col in ['dasm', 'dasm_ft', 'esm', 'ablang']:
        if col in merged_df.columns:
            valid = merged_df[col].notna().sum()
            print(f"  {col.upper()}: {valid}/{len(merged_df)} ({valid/len(merged_df)*100:.1f}%)")
    
    # Save final dataset
    final_path = output_dir / "magma_unified_dataset.csv"
    merged_df.to_csv(final_path, index=False)
    print(f"\n✅ Saved final dataset: {final_path}")
    
    # Save summary
    summary_path = output_dir / "magma_pipeline_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MAGMA-seq Unified Dataset (Kirby + Petersen)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total sequences: {len(merged_df)}\n")
        f.write(f"Unique sequences: {len(merged_df['sequence_id'].unique())}\n\n")
        
        f.write("Dataset breakdown:\n")
        for dataset in merged_df['dataset'].unique():
            count = (merged_df['dataset'] == dataset).sum()
            f.write(f"  {dataset}: {count} sequences\n")
        
        f.write(f"\nKD range: {merged_df['KD'].min():.1f} - {merged_df['KD'].max():.1f} nM\n")
        f.write(f"Binding score range: {merged_df['binding_score'].min():.2f} - {merged_df['binding_score'].max():.2f}\n\n")
        
        f.write("Model score coverage:\n")
        for col in ['dasm', 'dasm_ft', 'esm', 'ablang']:
            if col in merged_df.columns:
                valid = merged_df[col].notna().sum()
                f.write(f"  {col}: {valid}/{len(merged_df)} ({valid/len(merged_df)*100:.1f}%)\n")
    
    print(f"✅ Saved summary: {summary_path}")
    print(f"\n🎉 Pipeline complete! {len(merged_df)} sequences ready for analysis")

if __name__ == "__main__":
    main()