#!/usr/bin/env python3
"""
Direct GPU timing script for model benchmarking.

Times sequences from MAGMA dataset through each model directly on GPU:
- DASM (using load_crepe)
- ESM2-650M (using esm2_wrapper_of_size)
- AbLang2 (using AbLangWrapper)

Outputs simple timing results (model_name, wall_time_seconds).
For testing: uses 1 sequence. For production: uses 100 sequences.
"""

import os
import sys
import time
import pandas as pd
import torch
from pathlib import Path
import argparse

# Imports

from netam.framework import load_crepe
from dnsmex.ablang_wrapper import AbLangWrapper
from dnsmex.esm_wrapper import esm2_wrapper_of_size
from dnsmex.dxsm_zoo import dxsm_pick_device

# Configuration
SEQUENCE_COUNT = 100  # For production benchmark
TEST_SEQUENCE_COUNT = 1  # For testing
MAGMA_DATASET = "data/whitehead/unified/magma_unified_dataset_clean.csv"

def get_device(device_arg=None):
    """Get device based on argument or auto-select."""
    if device_arg:
        if device_arg.lower() == "cpu":
            return torch.device("cpu")
        elif device_arg.lower() == "gpu":
            # Use dxsm_pick_device for GPU selection (handles MPS issues)
            return dxsm_pick_device()
        else:
            return torch.device(device_arg)
    else:
        # Auto-select using dxsm_pick_device
        return dxsm_pick_device()

def load_test_sequences(count):
    """Load sequences from MAGMA dataset for testing.
    
    Args:
        count: Number of sequences to load
    """
    # Check if MAGMA dataset exists
    if not Path(MAGMA_DATASET).exists():
        raise FileNotFoundError(f"❌ MAGMA dataset not found at {MAGMA_DATASET}")
    
    df = pd.read_csv(MAGMA_DATASET)
    # Sample sequences with fixed random seed for reproducibility
    test_df = df.sample(n=count, random_state=42)
    return [(row['VH'], row['VL']) for _, row in test_df.iterrows()]

def time_dasm(sequences, device):
    """Time DASM model using load_crepe."""
    print(f"🚀 Timing DASM on {device}...")
    
    # Use specific DASM model
    model_path = "dasm-train/trained_models/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint"
    
    if not Path(f"{model_path}.pth").exists():
        raise FileNotFoundError(f"❌ DASM model not found at {model_path}.pth")
    
    print(f"Loading DASM model from {model_path}")
    crepe = load_crepe(model_path, device=device)
    
    # Time inference only
    start_time = time.time()
    
    for heavy, light in sequences:
        # DASM expects list of [heavy, light] pairs
        with torch.no_grad():
            _ = crepe([[heavy, light]])
    
    elapsed = time.time() - start_time
    return elapsed

def time_ablang2(sequences, device):
    """Time AbLang2 model using AbLangWrapper."""
    print(f"🚀 Timing AbLang2 on {device}...")
    
    # Initialize wrapper
    wrapper = AbLangWrapper(device=str(device))
    
    # Time inference only
    start_time = time.time()
    
    for heavy, light in sequences:
        _ = wrapper.masked_logits(heavy, light)
    
    elapsed = time.time() - start_time
    return elapsed

def time_esm2(sequences, device):
    """Time ESM2 model using esm2_wrapper_of_size."""
    print(f"🚀 Timing ESM2-650M on {device}...")
    
    # Initialize wrapper
    wrapper = esm2_wrapper_of_size("650M", device=device)
    
    # Time inference only
    start_time = time.time()
    
    for heavy, light in sequences:
        # Concatenate heavy + light chains as ESM2 expects single sequence
        concat_seq = heavy + light
        _ = wrapper.masked_logits(concat_seq)
    
    elapsed = time.time() - start_time
    return elapsed

def main(num_sequences, device_arg=None):
    print("=" * 80)
    print(f"DIRECT GPU TIMING BENCHMARK ({num_sequences} sequences)")
    print("=" * 80)
    
    # Get device
    device = get_device(device_arg)
    device_str = str(device).replace(":", "_")  # e.g., "cuda:0" -> "cuda_0"
    print(f"Using device: {device}")
    
    # Create output file with device info
    output_file = f"data/whitehead/processed/direct_timing_results_{device_str}.csv"
    
    # Load test sequences
    sequences = load_test_sequences(num_sequences)
    print(f"Loaded {len(sequences)} test sequences")
    
    # Define models to test
    models = {
        'DASM': time_dasm,
        'AbLang2': time_ablang2,
        'ESM2-650M': time_esm2
    }
    
    # Run timing for each model
    results = []
    timing_column = f"wall_time_seconds_{device_str}"
    per_seq_column = f"per_sequence_seconds_{device_str}"
    
    for model_name, timing_func in models.items():
        try:
            elapsed_time = timing_func(sequences, device)
            per_seq_time = elapsed_time / len(sequences)
            results.append({
                'model': model_name,
                timing_column: elapsed_time,
                per_seq_column: per_seq_time
            })
            print(f"✅ {model_name}: {elapsed_time:.3f}s ({per_seq_time:.3f}s/seq)")
        except Exception as e:
            print(f"❌ {model_name}: Failed with error: {e}")
            results.append({
                'model': model_name,
                timing_column: None,
                per_seq_column: None
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(Path(output_file).parent, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\n📊 Results saved to: {output_file}")
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL TIMING RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct GPU timing script for model benchmarking")
    parser.add_argument("--sequences", type=int, default=100, 
                       help="Number of sequences to test (default: 100)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], 
                       help="Device to use: 'cpu' or 'gpu' (default: auto-select)")
    
    args = parser.parse_args()
    main(args.sequences, args.device)