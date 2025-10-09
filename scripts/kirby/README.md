# Kirby Parent-Child Mutation Analysis Pipeline

This directory contains scripts for analyzing parent-child mutation effects using the Kirby combinatorial library data and MAGMA-seq data from the Petersen et al. paper.

## Overview

The pipeline tests whether DASM, ESM, and AbLang models can predict beneficial mutations by analyzing parent-child pairs where sequences differ by exactly one amino acid mutation.

## Scripts (Run in Order)

### 1. `kirby_prepare_parent_child_data.py`
**Purpose**: Load and prepare data for parent-child analysis

**Input**: 
- `_output/UCA_*_model_scores.csv` (Kirby scored sequences)
- `data/whitehead/petersen/MAGMAseq_anchor_FLAB.csv` (MAGMA-seq data)

**Output**: 
- `_output/kirby_parent_child_prepared_data.csv`

**What it does**:
- Loads model scores for each antibody group (UCA_2-15, UCA_002-S21F2, UCA_C118)
- Adds MAGMA-seq data as additional group
- Converts KD to binding score: `binding_score = -log10(KD_nM * 1e-9)`
- Combines all data with consistent format

### 2. `kirby_identify_parent_child_pairs.py`
**Purpose**: Find parent-child pairs differing by exactly one amino acid

**Input**: `_output/kirby_parent_child_prepared_data.csv`

**Output**: `_output/kirby_parent_child_mutation_pairs.csv`

**Algorithm**:
1. For each group, compute pairwise Hamming distances between all sequences
2. Identify pairs differing by exactly 1 amino acid (across VH+VL combined)
3. Determine parent→child directionality using distance to UCA:
   - Parent = sequence closer to UCA
   - Child = sequence farther from UCA
4. Extract mutation information (chain, position, amino acids)

### 3. `kirby_score_mutation_effects.py`
**Purpose**: Score mutation effects using different models

**Input**: `_output/kirby_parent_child_mutation_pairs.csv`

**Output**: `_output/kirby_parent_child_predictions.csv`

**Models**:
- **DASM**: Parent-relative selection factors using `score_mutation_relative_to_parent`
- **ESM**: Delta perplexity (child - parent)
- **AbLang**: Delta perplexity (child - parent)
- **ProGen2**: Delta perplexity (child - parent)

### 4. `kirby_analyze_mutation_predictions.py`
**Purpose**: Analyze and visualize model performance

**Input**: `_output/kirby_parent_child_predictions.csv`

**Outputs**:
- `figures/kirby_parent_child_scatter.svg` - Lattice scatter plots
- `figures/kirby_parent_child_performance.svg` - Performance comparison
- `_output/kirby_parent_child_performance.csv` - Metrics table

**Metrics**:
- Pearson correlation between predicted and observed ΔKd
- Spearman rank correlation
- Direction accuracy (% beneficial mutations predicted correctly)

### 5. `kirby_inspect_parent_child_examples.py` 
**Purpose**: Debug and inspect specific parent-child pairs

**Use**: For debugging and understanding specific mutation examples

## Key Data Formats

### Parent-Child Pairs CSV
Contains one row per parent-child mutation pair:
- `group`: Antibody system (e.g., UCA_2-15, MAGMA-seq_mixed)
- `parent_VH`, `parent_VL`: Parent sequences
- `child_VH`, `child_VL`: Child sequences (1 mutation different)
- `mutation_chain`: 'VH' or 'VL'
- `mutation_pos`: 1-indexed position in chain
- `from_aa`, `to_aa`: Amino acid change
- `delta_binding_score`: Observed ΔKd (child - parent, bigger = better binding)
- Model predictions: `dasm_delta`, `esm_delta`, etc.

## Usage Example

```bash
# Run the complete pipeline
python scripts/kirby/kirby_prepare_parent_child_data.py
python scripts/kirby/kirby_identify_parent_child_pairs.py  
python scripts/kirby/kirby_score_mutation_effects.py
python scripts/kirby/kirby_analyze_mutation_predictions.py
```

## Expected Results

- **>500 parent-child pairs** across Kirby and MAGMA-seq data
- **Correlation analysis** showing which models best predict beneficial mutations
- **Biological insights** about positions/mutations where models succeed/fail

## Notes

- All scripts require the netam environment: `source ../netam/.venv/bin/activate`
- MAGMA-seq data is treated as single mixed group until UCAs are inferred
- Uses distance to UCA for consistent parent→child directionality
- Filters sequences for quality (removes outliers, invalid KD values)