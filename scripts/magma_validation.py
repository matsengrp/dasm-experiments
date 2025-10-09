#!/usr/bin/env python3
"""
MAGMA-seq Pipeline Validation Script

This script validates five key aspects of the MAGMA-seq pipeline:
1. Replicate aggregation using geometric mean
2. Antibody assignment based on sequence similarity
3. AbLang model scoring with perplexity negation
4. ESM model scoring with perplexity negation
5. DASM model scoring using mutation-based approach
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Define test sequences
ALICE_VH = "QEVQLVQSGAEVKKPGESLKISCKASGYNFAYYWIGWVRQMPGKGLEWMGIIYPGDSDTRYSPSFQGQVTISADKSISTAYLQWSSLKASDTAMYYCARGEMTAVFGDYWGQGTLVTVSS"
ALICE_VL = "DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTLTISSLQPEDIATYYCQQYKILLTWTFGQGTKVEIKRTV"
ALICE_EXPECTED_ANTIBODY = "002-S21F2_UCA"

BOB_VH = "EVQLQESGGGLVRPGGTLRLSCAASGFSFSNYNMYWVRQAPGKGLEWVSSISGSGLSTYYADSVKGRFTISRDKSKNTVYLHMNSLRAEDTALYYCTKDFSTYIPMTGTFDSWGQGTQVTVSS"
BOB_VL = "EIVMTQSPATLSVSPGERATLSCRASQSVNSNLAWYQQRPGQAPRLLIYTASTRATGIPARFSGSGSGTEFTLTISSIQPEDFAVYYCQQYSNWPQLTFGGGTKVEIK"
BOB_RAW_KDS = [21.53, 19.49, 18.55]
BOB_EXPECTED_ANTIBODY = "319-345"


def load_pipeline_output():
    """Load the pipeline's final scored output file."""
    scored_file = "data/whitehead/unified/magma_unified_scored.csv"
    if not os.path.exists(scored_file):
        raise FileNotFoundError(f"Pipeline output not found: {scored_file}")
    return pd.read_csv(scored_file)


def load_reference_sequences():
    """Load antibody reference sequences from both datasets."""
    references = {}
    
    # Kirby references (UCAs)
    kirby_refs = pd.read_csv("data/whitehead/kirby/original/AntibodySequences1.csv")
    for antibody in kirby_refs['Antibody'].unique():
        vh = kirby_refs[(kirby_refs['Antibody'] == antibody) & (kirby_refs['Chain'] == 'VH')]['Sequence'].iloc[0]
        vl = kirby_refs[(kirby_refs['Antibody'] == antibody) & (kirby_refs['Chain'] == 'VL')]['Sequence'].iloc[0]
        references[antibody] = (vh, vl)
    
    # Petersen references (mature antibodies)
    petersen_refs = pd.read_csv("data/whitehead/petersen/original/AntibodySequences2.csv")
    for antibody in petersen_refs['Antibody'].unique():
        vh = petersen_refs[(petersen_refs['Antibody'] == antibody) & (petersen_refs['Chain'] == 'VH')]['Sequence'].iloc[0]
        vl = petersen_refs[(petersen_refs['Antibody'] == antibody) & (petersen_refs['Chain'] == 'VL')]['Sequence'].iloc[0]
        references[antibody] = (vh, vl)
    
    return references


def count_differences(seq1, seq2):
    """Count amino acid differences between two equal-length sequences."""
    if len(seq1) != len(seq2):
        return float('inf')  # Can't compare different lengths
    
    differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
    return differences


def find_closest_antibody(test_vh, test_vl, references):
    """Find the antibody with minimum sequence differences."""
    min_distance = float('inf')
    best_antibody = None
    
    for antibody, (ref_vh, ref_vl) in references.items():
        # Only compare sequences of the same length
        if len(ref_vh) == len(test_vh) and len(ref_vl) == len(test_vl):
            vh_diff = count_differences(ref_vh, test_vh)
            vl_diff = count_differences(ref_vl, test_vl)
            total_diff = vh_diff + vl_diff
            
            if total_diff < min_distance:
                min_distance = total_diff
                best_antibody = antibody
    
    return best_antibody, min_distance


def validate_replicate_aggregation(scored_df):
    """Test 1: Verify BOB's KD aggregation using geometric mean."""
    print("\n=== TEST 1: Replicate Aggregation ===")
    
    # Calculate expected geometric mean for BOB
    expected_kd = 10 ** np.mean(np.log10(BOB_RAW_KDS))
    print(f"BOB's raw KD values: {BOB_RAW_KDS}")
    print(f"Expected geometric mean: {expected_kd:.4f}")
    
    # Find BOB in pipeline output
    bob_mask = (scored_df['VH'] == BOB_VH) & (scored_df['VL'] == BOB_VL)
    if not bob_mask.any():
        raise ValueError("BOB sequence not found in pipeline output!")
    
    pipeline_kd = scored_df.loc[bob_mask, 'KD'].iloc[0]
    print(f"Pipeline KD value: {pipeline_kd:.4f}")
    
    # Calculate actual difference
    difference = abs(expected_kd - pipeline_kd)
    print(f"Absolute difference: {difference:.6f}")
    
    # Check if they match
    if difference < 0.001:
        print("✓ PASS: KD aggregation matches expected value")
        return True, difference
    else:
        print(f"✗ FAIL: KD mismatch! Expected {expected_kd:.4f}, got {pipeline_kd:.4f}")
        return False, difference


def validate_antibody_assignment(scored_df, references):
    """Test 2: Verify antibody assignment based on sequence similarity."""
    print("\n=== TEST 2: Antibody Assignment ===")
    
    # Test ALICE assignment
    alice_mask = (scored_df['VH'] == ALICE_VH) & (scored_df['VL'] == ALICE_VL)
    if not alice_mask.any():
        raise ValueError("ALICE sequence not found in pipeline output!")
    
    alice_antibody = scored_df.loc[alice_mask, 'antibody'].iloc[0]
    print(f"ALICE assigned to: {alice_antibody} (expected: {ALICE_EXPECTED_ANTIBODY})")
    
    # Independently verify ALICE assignment
    alice_best, alice_dist = find_closest_antibody(ALICE_VH, ALICE_VL, references)
    print(f"Independent calculation: ALICE closest to {alice_best} (distance: {alice_dist})")
    
    # Test BOB assignment
    bob_mask = (scored_df['VH'] == BOB_VH) & (scored_df['VL'] == BOB_VL)
    bob_antibody = scored_df.loc[bob_mask, 'antibody'].iloc[0]
    print(f"BOB assigned to: {bob_antibody} (expected: {BOB_EXPECTED_ANTIBODY})")
    
    # Independently verify BOB assignment
    bob_best, bob_dist = find_closest_antibody(BOB_VH, BOB_VL, references)
    print(f"Independent calculation: BOB closest to {bob_best} (distance: {bob_dist})")
    
    # Verify assignments match our independent calculation
    alice_pass = alice_antibody == alice_best == ALICE_EXPECTED_ANTIBODY
    bob_pass = bob_antibody == bob_best == BOB_EXPECTED_ANTIBODY
    
    if alice_pass and bob_pass:
        print("✓ PASS: Both antibody assignments are correct")
        return True, (alice_dist, bob_dist)
    else:
        print("✗ FAIL: Antibody assignment mismatch")
        return False, (alice_dist, bob_dist)


def validate_ablang_score(scored_df):
    """Test 3: Verify AbLang scoring with perplexity negation."""
    print("\n=== TEST 3: AbLang Model Scoring ===")
    
    try:
        # Import AbLang wrapper
        from dnsmex.ablang_wrapper import AbLangWrapper
        
        # Initialize AbLang locally
        print("Loading AbLang model...")
        ablang_wrapper = AbLangWrapper(device="cpu")
        
        # Calculate ALICE's score independently
        print("Computing ALICE's AbLang score...")
        raw_perplexity = ablang_wrapper.pseudo_perplexity([[ALICE_VH, ALICE_VL]])[0]
        expected_score = -raw_perplexity  # Pipeline negates perplexity
        print(f"Raw perplexity: {raw_perplexity:.4f}")
        print(f"Expected score (negated): {expected_score:.4f}")
        
        # Get pipeline score
        alice_mask = (scored_df['VH'] == ALICE_VH) & (scored_df['VL'] == ALICE_VL)
        pipeline_score = scored_df.loc[alice_mask, 'ablang'].iloc[0]
        print(f"Pipeline score: {pipeline_score:.4f}")
        
        # Calculate actual difference
        difference = abs(expected_score - pipeline_score)
        print(f"Absolute difference: {difference:.6f}")
        
        # Check if they match (with small tolerance for floating point)
        if difference < 0.01:
            print("✓ PASS: AbLang score matches expected value")
            return True, difference
        else:
            print(f"✗ FAIL: AbLang score mismatch! Expected {expected_score:.4f}, got {pipeline_score:.4f}")
            return False, difference
            
    except Exception as e:
        print(f"✗ ERROR: Failed to validate AbLang scoring: {e}")
        return False, float('inf')


def validate_esm_score(scored_df):
    """Test 4: Verify ESM scoring with perplexity negation."""
    print("\n=== TEST 4: ESM Model Scoring ===")
    
    try:
        # Import ESM wrapper
        from dnsmex.esm_wrapper import esm2_wrapper_of_size
        
        # Initialize ESM locally (650M model to match pipeline)
        print("Loading ESM-650M model...")
        esm_wrapper = esm2_wrapper_of_size("650M")
        
        # Calculate ALICE's score independently
        print("Computing ALICE's ESM score...")
        vh_perplexity = esm_wrapper.pseudo_perplexity(ALICE_VH)
        vl_perplexity = esm_wrapper.pseudo_perplexity(ALICE_VL)
        avg_perplexity = (vh_perplexity + vl_perplexity) / 2  # Pipeline averages VH+VL
        expected_score = -avg_perplexity  # Pipeline negates perplexity
        
        print(f"VH perplexity: {vh_perplexity:.4f}")
        print(f"VL perplexity: {vl_perplexity:.4f}")
        print(f"Average perplexity: {avg_perplexity:.4f}")
        print(f"Expected score (negated): {expected_score:.4f}")
        
        # Get pipeline score
        alice_mask = (scored_df['VH'] == ALICE_VH) & (scored_df['VL'] == ALICE_VL)
        pipeline_score = scored_df.loc[alice_mask, 'esm'].iloc[0]
        print(f"Pipeline score: {pipeline_score:.4f}")
        
        # Calculate actual difference
        difference = abs(expected_score - pipeline_score)
        print(f"Absolute difference: {difference:.6f}")
        
        # Check if they match (with small tolerance for floating point)
        if difference < 0.01:
            print("✓ PASS: ESM score matches expected value")
            return True, difference
        else:
            print(f"✗ FAIL: ESM score mismatch! Expected {expected_score:.4f}, got {pipeline_score:.4f}")
            return False, difference
            
    except Exception as e:
        print(f"✗ ERROR: Failed to validate ESM scoring: {e}")
        return False, float('inf')


def validate_dasm_score(scored_df, references):
    """Test 5: Verify DASM scoring using mutation-based approach."""
    print("\n=== TEST 5: DASM Model Scoring ===")
    
    try:
        # Import DASM infrastructure
        from netam.framework import load_crepe
        from dnsmex.local import localify
        from dnsmex.dms_helper import protein_differences, sel_score_of_differences
        import torch
        
        # Load DASM model (production model used in pipeline)
        print("Loading DASM production model...")
        dasm_base = load_crepe(localify("DASM_TRAINED_MODELS_DIR/dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint"), device="cpu")
        
        # Find ALICE's assigned antibody and get reference sequence
        alice_mask = (scored_df['VH'] == ALICE_VH) & (scored_df['VL'] == ALICE_VL)
        alice_antibody = scored_df.loc[alice_mask, 'antibody'].iloc[0]
        print(f"ALICE assigned to antibody: {alice_antibody}")
        
        if alice_antibody not in references:
            print(f"✗ FAIL: Reference sequence not found for {alice_antibody}")
            return False, float('inf')
        
        ref_vh, ref_vl = references[alice_antibody]
        print(f"Reference VH length: {len(ref_vh)}, Test VH length: {len(ALICE_VH)}")
        print(f"Reference VL length: {len(ref_vl)}, Test VL length: {len(ALICE_VL)}")
        
        # Calculate mutations between reference and test sequence
        vh_differences = protein_differences(ref_vh, ALICE_VH)
        vl_differences = protein_differences(ref_vl, ALICE_VL)
        print(f"VH mutations: {len(vh_differences)}")
        print(f"VL mutations: {len(vl_differences)}")
        print(f"Total mutations: {len(vh_differences) + len(vl_differences)}")
        
        if len(vh_differences) > 0:
            print(f"VH mutation examples: {vh_differences[:3]}")
        if len(vl_differences) > 0:
            print(f"VL mutation examples: {vl_differences[:3]}")
        
        # Get DASM scores for reference sequence (as in pipeline)
        print("Computing DASM scores for reference sequence...")
        [[dasm_heavy, dasm_light]] = dasm_base([[ref_vh, ref_vl]])
        
        # Transpose and log transform (as in pipeline)
        log_dasm_heavy = torch.log(dasm_heavy.T)
        log_dasm_light = torch.log(dasm_light.T)
        
        # Calculate selection scores for mutations
        print("Computing selection scores for mutations...")
        dasm_heavy_score = sel_score_of_differences(log_dasm_heavy, vh_differences)
        dasm_light_score = sel_score_of_differences(log_dasm_light, vl_differences)
        expected_score = dasm_heavy_score + dasm_light_score
        
        print(f"Heavy chain selection score: {dasm_heavy_score:.4f}")
        print(f"Light chain selection score: {dasm_light_score:.4f}")
        print(f"Expected total score: {expected_score:.4f}")
        
        # Get pipeline score
        dasm_column = 'dasm_base'  # We know this exists
        pipeline_score = scored_df.loc[alice_mask, dasm_column].iloc[0]
        print(f"Pipeline score: {pipeline_score:.4f}")
        
        # Calculate actual difference
        difference = abs(expected_score - pipeline_score)
        print(f"Absolute difference: {difference:.6f}")
        
        # Check if they match (with small tolerance for floating point)
        if difference < 0.01:
            print("✓ PASS: DASM score matches expected value")
            return True, difference
        else:
            print(f"✗ FAIL: DASM score mismatch! Expected {expected_score:.4f}, got {pipeline_score:.4f}")
            return False, difference
            
    except Exception as e:
        print(f"✗ ERROR: Failed to validate DASM scoring: {e}")
        return False, float('inf')


def main():
    """Run all validation tests."""
    print("="*60)
    print("MAGMA-seq Pipeline Validation")
    print("="*60)
    
    # Load pipeline output
    print("\nLoading pipeline output...")
    scored_df = load_pipeline_output()
    print(f"Loaded {len(scored_df)} sequences from pipeline")
    
    # Load reference sequences
    print("\nLoading reference sequences...")
    references = load_reference_sequences()
    print(f"Loaded {len(references)} reference antibodies")
    
    # Run tests
    results = []
    differences = []
    
    test1_pass, test1_diff = validate_replicate_aggregation(scored_df)
    results.append(test1_pass)
    differences.append(("KD Aggregation", test1_diff))
    
    test2_pass, test2_diff = validate_antibody_assignment(scored_df, references)
    results.append(test2_pass)
    differences.append(("Antibody Assignment", f"ALICE: {test2_diff[0]}, BOB: {test2_diff[1]} differences"))
    
    test3_pass, test3_diff = validate_ablang_score(scored_df)
    results.append(test3_pass)
    differences.append(("AbLang Score", test3_diff))
    
    test4_pass, test4_diff = validate_esm_score(scored_df)
    results.append(test4_pass)
    differences.append(("ESM Score", test4_diff))
    
    test5_pass, test5_diff = validate_dasm_score(scored_df, references)
    results.append(test5_pass)
    differences.append(("DASM Score", test5_diff))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    print("\nDetailed differences:")
    for test_name, diff in differences:
        if isinstance(diff, float):
            if diff == float('inf'):
                print(f"  {test_name}: ERROR (test failed to run)")
            else:
                print(f"  {test_name}: {diff:.6f}")
        else:
            print(f"  {test_name}: {diff}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Pipeline validated successfully!")
        return 0
    else:
        print("\n✗ VALIDATION FAILED - Pipeline has issues!")
        return 1


if __name__ == "__main__":
    sys.exit(main())