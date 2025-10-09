#!/usr/bin/env python3
"""
Cache model scores for Kirby binary classification analysis.

Generates and caches expensive model computations:
- DASM: Local processing (fast)
- ESM: Remote GPU processing (ermine) 
- AbLang: Local processing

Saves scored partition CSV files for notebooks/dasm_paper/kirby_binary.ipynb
"""
import os
import sys
import pandas as pd
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Change to project root directory
os.chdir(project_root)

from netam.framework import load_crepe
from dnsmex.local import localify
from dnsmex.kirby_helper import KirbyBinaryPartition

def main():
    print("=" * 80)
    print("COMPLETE KIRBY BINARY ANALYSIS")
    print("DASM + DASM-FT-mild + DASM-FT-strong (Local) + ESM (Remote GPU) + AbLang (Local) + ProGen2 (Remote GPU)")
    print("=" * 80)
    
    # Track whether we're using cache or computing
    cache_status = {}
    
    # Load KD-based binary classification data
    binary_csv_path = "_output/kirby_kd_based_binary.csv"
    binary_df = pd.read_csv(binary_csv_path)
    print(f"\nDataset: {len(binary_df)} sequences (KD-based binary classification)")
    print(f"Binders: {(binary_df['binary_label'] == 1).sum()} ({(binary_df['binary_label'] == 1).mean():.1%})")
    print(f"Non-binders: {(binary_df['binary_label'] == 0).sum()} ({(binary_df['binary_label'] == 0).mean():.1%})")
    print(f"KD threshold: {binary_df['KD_continuous'].median():.1f} nM")
    
    # Load DASM models
    print(f"\nLoading DASM models...")
    crepe = load_crepe(localify("DASM_TRAINED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint"))
    print(f"   ✅ DASM base model loaded")
    
    crepe_ft_mild = load_crepe(localify("DASM_UPDATED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint-UPDATED-v1jaffePairedCC@0.5@v2kimTrain-1e-1c-0.0001lr"))
    print(f"   ✅ DASM-FT-mild model loaded")
    
    crepe_ft_strong = load_crepe(localify("DASM_UPDATED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint-UPDATED-v1jaffePairedCC@0.5@v2kimTrain"))
    print(f"   ✅ DASM-FT-strong model loaded")
    
    # Process all UCA partitions
    partition_summary = pd.read_csv(localify("DATA_DIR/whitehead/kirby/uca_baseline_partitions/uca_baseline_summary.csv"))
    antibodies = [ab for ab in partition_summary['antibody'].tolist() if ab.startswith('UCA_')]
    print(f"\nProcessing {len(antibodies)} UCA partitions: {antibodies}")
    timing_results = []
    
    for antibody in antibodies:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING {antibody}")
        print(f"{'=' * 60}")
        
        # Load partition
        partition_file = localify(f"DATA_DIR/whitehead/kirby/uca_baseline_partitions/{antibody}.csv")
        df = pd.read_csv(partition_file)
        
        partition = KirbyBinaryPartition(antibody=antibody, df=df)
        partition.load_binary_labels(binary_csv_path)
        
        if len(partition.df) == 0:
            print(f"No sequences found for {antibody}")
            continue
        
        binder_count = (partition.df['binary_label'] == 1).sum()
        print(f"Sequences: {len(partition.df)}")
        print(f"Binders: {binder_count} ({binder_count/len(partition.df):.1%})")
        
        # DASM - Local processing
        print(f"\n🏠 Adding DASM scores (local)...")
        start_time = time.time()
        partition.add_dasm_scores(crepe)
        dasm_time = time.time() - start_time
        print(f"   ✅ DASM completed in {dasm_time:.1f}s")
        
        # DASM-FT-mild - Local processing
        print(f"\n🏠 Adding DASM-FT-mild scores (local)...")
        start_time = time.time()
        partition.add_dasm_ft_mild_scores(crepe_ft_mild)
        dasm_ft_mild_time = time.time() - start_time
        print(f"   ✅ DASM-FT-mild completed in {dasm_ft_mild_time:.1f}s")
        
        # DASM-FT-strong - Local processing
        print(f"\n🏠 Adding DASM-FT-strong scores (local)...")
        start_time = time.time()
        partition.add_dasm_ft_strong_scores(crepe_ft_strong)
        dasm_ft_strong_time = time.time() - start_time
        print(f"   ✅ DASM-FT-strong completed in {dasm_ft_strong_time:.1f}s")
        
        # ESM + AbLang - Both use remote GPU processing
        print(f"\n🚀 Adding ESM and AbLang scores (both remote GPU)...")
        start_time = time.time()
        partition.add_cached_model_scores(use_remote_esm=True, use_remote_ablang=True)
        model_time = time.time() - start_time
        print(f"   ✅ ESM + AbLang completed in {model_time:.1f}s")
        
        # ProGen2 - Remote GPU processing  
        print(f"\n🚀 Adding ProGen2 scores (remote GPU)...")
        start_time = time.time()
        partition.add_progen_scores(model_version="progen2-small")
        progen_time = time.time() - start_time
        print(f"   ✅ ProGen2 completed in {progen_time:.1f}s")
        
        timing_results.append({
            'antibody': antibody,
            'sequences': len(partition.df),
            'dasm_time': dasm_time,
            'dasm_ft_mild_time': dasm_ft_mild_time,
            'dasm_ft_strong_time': dasm_ft_strong_time,
            'esm_ablang_time': model_time,
            'progen_time': progen_time,
            'total_time': dasm_time + dasm_ft_mild_time + dasm_ft_strong_time + model_time + progen_time
        })
        
        # Save partition with all model scores
        print(f"\n💾 Saving scored partition data...")
        
        # Save partition data with model scores
        output_file = f"_output/{antibody}_model_scores.csv"
        partition.df.to_csv(output_file, index=False)
        print(f"   ✅ Saved to {output_file}")
        
    
    # Final summary
    print("\n" + "=" * 80)
    print("MODEL SCORING COMPLETE")
    print("=" * 80)
    
    # Timing summary
    print("\n⏱️  Processing Times:")
    timing_df = pd.DataFrame(timing_results)
    print(timing_df.to_string(index=False))
    
    # Save timing results
    os.makedirs("_output", exist_ok=True)
    timing_df.to_csv("_output/model_scoring_times.csv", index=False)
    
    print("\n📁 Model scores saved:")
    for antibody in antibodies:
        print(f"  • _output/{antibody}_model_scores.csv")
    print("  • _output/model_scoring_times.csv")
    
    print("\n✅ Model scoring complete!")
    print("Now run: notebooks/dasm_paper/kirby_binary.ipynb")

if __name__ == "__main__":
    main()