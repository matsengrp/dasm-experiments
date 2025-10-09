# Petersen MAGMA-seq Dataset Analysis

## Overview

This directory contains processed data and analysis results for the Petersen et al. 2024 MAGMA-seq antibody dataset. The analysis reveals important insights about the experimental design and evolutionary context of the studied antibodies.

## Key Findings

### Experimental Design Clarification

The Petersen MAGMA-seq experiment explores **local mutagenesis around mature antibodies**, not full evolutionary trajectories from germline:

- **Reference point**: Mature, optimized antibodies (from Dr. Whitehead's `AntibodySequences2.csv`)
- **Mutation space**: 0-4 amino acid changes from mature sequence
- **Direction**: Mature antibody → variants (not UCA → mature)
- **Expected trend**: KD increases with mutations (binding gets worse as you deviate from optimized sequence)

### UCA Reconstruction Results

Using IgBLAST analysis of mature sequences, we successfully reconstructed true germline UCAs:

#### 222-1C06 Antibody
- **Germline assignment**: IGHV3-48*01 + IGKV3-15*01
- **Total mutations from germline**: 15 mutations
  - VH: 7 mutations (D10G, V23A, V24A, R58T, Y59I, L83Y, L86M)
  - VL: 8 mutations (T28S, I29V, R30S, D32N, P43A, I47L, N92F, P95Y)
- **IgBLAST identity**: 89.8% (VH), 91.6% (VL)
- **Assigned sequences**: 373 total

#### 319-345 Antibody  
- **Germline assignment**: IGHV3-23*01 + IGKV3-15*01
- **Total mutations from germline**: 11 mutations
  - VH: 11 mutations (Q5L, T17Q, L20S, F29T, Y32A, N33S, S50A, K74N, V79L, H82Q, T97A)
  - VL: 0 mutations (already at germline for VL)
- **IgBLAST identity**: 85.7% (VH), high (VL)
- **Assigned sequences**: 669 total

### Evolutionary Context

Both antibodies are **highly evolved** from their germline origins:
- 222-1C06: 15 mutations of somatic hypermutation
- 319-345: 11 mutations of somatic hypermutation

The experimental library captures only the **final optimization phase**:
```
True Germline UCA --[11-15 mutations]--> Mature Antibody --[0-4 mutations]--> Library Variants
                   [Not explored]                          [Experimental space]
```

## Files in This Directory

### Data Files
- `222-1C06_partition.csv` - 373 sequences assigned to 222-1C06
- `319-345_partition.csv` - 669 sequences assigned to 319-345  
- `igblast_results.txt` - IgBLAST analysis results
- `petersen_reconstructed_ucas.fasta` - Initial UCA reconstruction (deprecated)
- `petersen_complete_ucas.fasta` - Complete reconstructed UCA sequences

### Analysis Results
- `../processed/petersen_uca_mature_distance_regression.svg` - UCA distance regression plot

### Analysis Reports
- `petersen_assignment_report.md` - Comprehensive assignment statistics
- `README.md` - This overview document

### Reference Data
- `../original/AntibodySequences2.csv` - Dr. Whitehead's authoritative mature sequences
- `../original/MAGMAseq_anchor_FLAB.csv` - Original FLAB experimental data

## Mutation Distribution Analysis

The experimental data shows limited mutational exploration:

### 222-1C06 (373 sequences)
- 0 mutations: 100 sequences (26.8%)
- 1 mutation: 171 sequences (45.8%) 
- 2 mutations: 85 sequences (22.8%)
- 3 mutations: 15 sequences (4.0%)
- 4 mutations: 2 sequences (0.5%)

### 319-345 (669 sequences)
- 0 mutations: 172 sequences (25.7%)
- 1 mutation: 286 sequences (42.8%)
- 2 mutations: 167 sequences (25.0%)
- 3 mutations: 35 sequences (5.2%)
- 4 mutations: 9 sequences (1.3%)

## KD Trends Explained

**Why KD values increase with mutations**: The experimental design starts with mature, optimized antibodies and introduces random mutations. Most random changes to an already-optimized protein will decrease performance, hence higher KD (worse binding).

This is the **opposite** of natural antibody evolution, where mutations are selected for improved binding.

## UCA Distance Regression Analysis

### Reversion Hypothesis Test

**Question**: Do targeted mutations represent reversions toward UCA or random deviations?

**Method**: Plot mutations from mature (MAGMA data) vs estimated mutations from true UCA (using triangle inequality)

**Results**:
- **222-1C06**: r = 0.278, p < 0.001 (n = 191)
- **319-345**: r = 0.278, p < 0.001 (n = 348) 
- **Combined**: r = 0.278, p < 0.001 (n = 539)

**Conclusion**: **Reversion hypothesis REJECTED**
- Positive correlation indicates mutations are random deviations, not reversions toward UCA
- Targeted mutagenesis creates variants that move away from both mature antibody AND UCA
- This confirms the experimental design: mature → random variants (not UCA-directed evolution)

**Visualization**: `../processed/petersen_uca_mature_distance_regression.svg`

## Scripts Used

### UCA Reconstruction
- `../../scripts/extract_petersen_uca_from_igblast.py` - Automated UCA extraction
- `../../scripts/manual_petersen_uca_extraction.py` - Manual UCA reconstruction
- `../../compare_mature_uca_mutations.py` - 222-1C06 mutation analysis (throwaway)
- `../../compare_mature_uca_mutations_319.py` - 319-345 mutation analysis (throwaway)

### Assignment Pipeline
- `../../scripts/magma_assign_antibodies.py` - Main assignment script
- `../../scripts/magma_generate_mutation_tree_report.py` - Visualization generator

### UCA Distance Regression Analysis
- `../../scripts/petersen_uca_mature_distance_regression.py` - **Main analysis**: Tests reversion hypothesis using UCA-mature distance regression
- `../../scripts/verify_petersen_mutations.py` - Validates mutation counts via direct sequence comparison
- `../../scripts/petersen_reconstruct_complete_ucas.py` - Generates complete UCA sequences from known mutations

## Dataset Selection

The analysis focuses on **2 specific antibodies (222-1C06 and 319-345)** that were selected by Dr. Whitehead from the larger Petersen dataset. These antibodies represent:

- **Influenza neutralizing antibodies**: Selected for studying "preliminary rules of recognition for an emerging class of influenza neutralizing antibodies" (Petersen et al.)
- **High-quality data**: 1,042 total sequences with complete KD measurements
- **Comprehensive coverage**: All sequences have mutation tracking and binding measurements
- **Successful UCA reconstruction**: Germline origins identified for both antibodies

This represents a targeted subset rather than a technical limitation of the assignment process.

## Implications for Analysis

1. **Limited evolutionary scope**: Only explores final 0-4 mutations of antibody development
2. **Optimization context**: Variants are deviations from optimized mature sequences
3. **Model training**: Ideal for studying local sequence-function relationships around mature binders
4. **Comparison studies**: Different experimental paradigm than typical UCA→mature evolution

---

*Analysis completed: July 2025*  
*Dataset: Petersen et al. 2024 MAGMA-seq*  
*Reference: "An integrated technology for quantitative wide mutational scanning of human antibody Fab libraries"*