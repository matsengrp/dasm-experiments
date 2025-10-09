# Petersen Paper UCA Analysis

This directory contains scripts for inferring and reconstructing UCA sequences for antibodies from the Petersen et al. 2024 MAGMA-seq paper.

## Paper Reference
**Petersen et al. 2024** - "An integrated technology for quantitative wide mutational scanning of human antibody Fab libraries" - *Nature Communications*

## Overview

UCAs represent the germline sequences before somatic hypermutation. These are essential for:
- Understanding antibody evolution trajectories  
- Creating true parent-child pairs for mutation analysis
- Testing model predictions against real evolutionary pathways

## Workflow

### 1. Prepare Sequences for IgBLAST
Create FASTA files with mature antibody sequences (nucleotides) for IgBLAST analysis.

**Input**: Mature antibody sequences from GenBank
**Tool**: IgBLAST web interface (https://www.ncbi.nlm.nih.gov/igblast/)

### 2. Parse IgBLAST Results
Extract germline gene assignments and alignments from IgBLAST output.

**Scripts**:
- `parse_igblast.py` - Convert MHTML to clean text format
- Creates: `igblast_results_clean.txt`

### 3. Reconstruct UCAs
Apply germline mutations to reconstruct UCA sequences.

**Script**: `reconstruct_ucas.py`

**Method**: 
- Parse IgBLAST alignments showing Query vs Germline
- Apply pattern: "." = keep query, letter = change to germline
- Revert somatic hypermutations back to germline state

**Output**: `data/petersen_uca_results/uca_sequences_for_parent_child.fasta`

### 4. Verify Reconstruction
Validate UCA reconstruction against IgBLAST reported mutation counts.

**Script**: `verify_uca_reconstruction.py`

**Verification**: Compare mature vs UCA sequences, check mutation counts match IgBLAST

## Results Summary

### Antibody Sequences Analyzed
- **4A8** (NTD-targeting): VH + VL
- **CC12.1** (RBD-targeting): VH + VL  
- **CC6.31** (weak binder): VH + VL

### Somatic Hypermutations Found
| Antibody | VH Mutations | VL Mutations | Total |
|----------|--------------|--------------|-------|
| 4A8      | 4           | 4            | 8     |
| CC12.1   | 3           | 3            | 6     |
| CC6.31   | 4           | 0*           | 4     |
| **Total**| **11**      | **7**        | **18**|

*CC6.31 VL is 100% identical to germline

### Verification Results
✅ **All mutation counts match IgBLAST exactly**
✅ **18 total mutations found vs 18 expected**
✅ **Perfect UCA reconstruction validated**

## Key Files

- `data/petersen_uca_results/igblast_results_clean.txt` - Clean IgBLAST alignments 
- `data/petersen_uca_results/uca_sequences_for_parent_child.fasta` - Verified UCA sequences
- `mature_antibodies_nucleotides_for_igblast.fasta` - Original input sequences

## IgBLAST Gene Assignments

### Heavy Chains (VH)
- **4A8_VH**: IGHV1-24*01 + IGHD6-19*01 + IGHJ6*02 (98.6% identity)
- **CC12.1_VH**: IGHV3-53*01 + IGHD5-24*01 + IGHJ6*02 (99.0% identity)
- **CC6.31_VH**: IGHV1-46*01 + IGHD3-22*01 + IGHJ4*02 (98.6% identity)

### Light Chains (VL)
- **4A8_VL**: IGKV2-24*02 + IGKJ2*01 (98.7% identity)
- **CC12.1_VL**: IGKV1-9*01 + IGKJ3*01 (99.0% identity)
- **CC6.31_VL**: IGKV1-17*01 + IGKJ4*01 (100% identity - no mutations!)

## Usage

```bash
# After running IgBLAST web interface
python scripts/petersen_uca_analysis/parse_igblast.py
python scripts/petersen_uca_analysis/reconstruct_ucas.py  
python scripts/petersen_uca_analysis/verify_uca_reconstruction.py
```

## Applications

These UCA sequences enable:
1. **Real evolutionary analysis**: UCA → mature antibody trajectories
2. **Parent-child mutation studies**: Each mutation represents natural selection
3. **Model validation**: Test if ML models can predict beneficial mutations
4. **Functional analysis**: Understand which mutations improve binding affinity