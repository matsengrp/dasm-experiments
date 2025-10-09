#!/usr/bin/env python3
"""
Score the cleaned unified MAGMA-seq dataset with all models.
Uses the dataset with invalid amino acid sequences removed.
"""

import os
import sys
import csv
from pathlib import Path

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

def load_unified_dataset(file_path):
    """Load the unified dataset."""
    sequences = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row)
    
    return sequences

def score_with_all_models(sequences):
    """Score sequences with all models using proper MAGMA infrastructure."""
    print("🏠 Using proper MAGMA scoring infrastructure for all models...")
    
    try:
        from netam.framework import load_crepe
        from dnsmex.local import localify
        from dnsmex.magma_helper import score_unified_dataset_with_all_models
        import pandas as pd
        
        # Load DASM model
        dasm_base = load_crepe(localify("DASM_TRAINED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint"))
        
        print(f"Scoring {len(sequences)} sequences with proper partition-based infrastructure...")
        
        # Convert sequences to DataFrame for proper scoring
        df = pd.DataFrame(sequences)
        
        # Use the proper unified scoring infrastructure for all models
        scored_df = score_unified_dataset_with_all_models(df, dasm_base, use_remote_esm=True, use_remote_ablang=True)
        
        # Copy all model scores back to original sequence dictionaries
        score_columns = ['dasm', 'esm', 'ablang', 'progen']
        output_columns = ['dasm_base', 'esm', 'ablang', 'progen']
        
        for i, seq in enumerate(sequences):
            if i < len(scored_df):
                for score_col, output_col in zip(score_columns, output_columns):
                    seq[output_col] = scored_df.iloc[i].get(score_col, None)
            else:
                for output_col in output_columns:
                    seq[output_col] = None
        
        print("✅ All model scoring completed using proper infrastructure")
        return True
        
    except Exception as e:
        print(f"❌ Model scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 80)
    print("SCORING CLEANED UNIFIED MAGMA-SEQ DATASET")
    print("=" * 80)
    
    # Load cleaned unified dataset
    input_file = "data/whitehead/unified/magma_unified_dataset_clean.csv"
    output_file = "data/whitehead/unified/magma_unified_scored.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Cleaned dataset not found: {input_file}")
        print("Please run: python3 scripts/clean_unified_dataset.py")
        return
    
    print(f"📂 Loading cleaned dataset from: {input_file}")
    sequences = load_unified_dataset(input_file)
    print(f"Loaded {len(sequences)} valid sequences from {len(set(s['antibody'] for s in sequences))} antibodies")
    
    # Score with all models using proper infrastructure  
    print(f"\\n{'='*60}")
    print("UNIFIED MODEL SCORING")
    print(f"{'='*60}")
    
    scoring_success = score_with_all_models(sequences)
    if scoring_success:
        print("✅ All model scoring completed")
    else:
        print("❌ Model scoring failed")
    
    # Save scored dataset
    print(f"\\n{'='*60}")
    print("SAVING SCORED DATASET")
    print(f"{'='*60}")
    
    print(f"💾 Writing scored dataset to: {output_file}")
    
    # Define output columns
    fieldnames = [
        'sequence_id', 'VH', 'VL', 'KD', 'antibody', 'dataset',
        'similarity_score', 'mutations_from_reference', 'heavy_mutations', 'light_mutations', 
        'reference_VH', 'reference_VL', 'binding_score', 'experimental_design',
        'dasm_base', 'esm', 'ablang', 'progen'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sequences)
    
    # Print summary statistics
    print(f"\\n📊 SCORING SUMMARY")
    print("=" * 50)
    
    models = ['dasm_base', 'esm', 'ablang', 'progen']
    
    for model in models:
        valid_scores = [s[model] for s in sequences if s.get(model) is not None]
        coverage = len(valid_scores) / len(sequences) * 100
        
        if valid_scores:
            mean_score = sum(float(score) for score in valid_scores if score is not None) / len(valid_scores)
            print(f"{model.upper():>12}: {len(valid_scores):>4}/{len(sequences)} ({coverage:>5.1f}%) | Mean: {mean_score:>8.3f}")
        else:
            print(f"{model.upper():>12}: {len(valid_scores):>4}/{len(sequences)} ({coverage:>5.1f}%) | Mean: {'N/A':>8}")
    
    print(f"\\n✅ UNIFIED MODEL SCORING COMPLETE!")
    print(f"Scored dataset saved to: {output_file}")
    print(f"Ready for analysis and model comparison!")

if __name__ == "__main__":
    main()