# MAGMA-seq Analysis Pipeline

This document describes the MAGMA-seq analysis pipeline for processing antibody mutation data from two major studies.

## Overview

The pipeline combines data from:
- **Kirby et al. (PNAS 2025)**: UCA → mature antibody evolution trajectories 
- **Petersen et al. (Nature Communications 2024)**: CDR-targeted mutagenesis around mature antibodies

## Core Pipeline Scripts

### Data Processing
- `scripts/magma_assign_antibodies.py` - Assigns sequences to antibodies using reference matching
- `scripts/magma_unify_datasets.py` - Combines Kirby and Petersen datasets
- `scripts/magma_clean_unified_dataset.py` - Data cleaning with replicate aggregation
- `scripts/magma_score_unified_dataset_clean.py` - Multi-model scoring (DASM, ESM, AbLang, ProGen)

### Visualization and Analysis
- `scripts/magma_generate_mutation_tree_report.py` - Creates interactive mutation tree visualization
- `scripts/magma_unified_model_correlation_analysis.py` - Creates correlation plots and tables with corrected -log10(KD) calculation
- `dnsmex/magma_tree_analysis.py` - Tree analysis and parent-child identification functions

### Infrastructure
- `scripts/flab_progen.py` - ProGen2 scoring script
- `scripts/setup_progen_env.sh` - ProGen2 environment setup
- `scripts/setup_progen_scripts.sh` - ProGen2 script deployment
- `scripts/README_progen_setup.md` - ProGen2 setup documentation
- `scripts/timing.py` - Model timing infrastructure
- `scripts/show_model_timings.py` - Cache timing analysis

### Quality Assurance
- `scripts/magma_validation.py` - Comprehensive pipeline validation script that independently verifies 5 core operations: replicate aggregation, antibody assignment, AbLang/ESM/DASM scoring

## Data Files

### Source Data
- `data/whitehead/kirby/original/AntibodySequences1.csv` - Kirby reference sequences
- `data/whitehead/kirby/original/Kirby_PNAS2025_FLAB_*.csv` - Kirby experimental data
- `data/whitehead/petersen/original/AntibodySequences2.csv` - Petersen reference sequences
- `data/whitehead/petersen/original/MAGMAseq_anchor_FLAB*.csv` - Petersen experimental data

### Processed Data
- `data/whitehead/kirby/processed/*_partition.csv` - Kirby antibody partitions (4 files)
- `data/whitehead/petersen/processed/*_partition.csv` - Petersen antibody partitions (2 files)
- `data/whitehead/unified/magma_unified_dataset.csv` - Combined dataset (1,130 sequences)
- `data/whitehead/unified/magma_unified_dataset_clean.csv` - Cleaned dataset (1,128 sequences)
- `data/whitehead/unified/magma_unified_scored.csv` - Dataset with model predictions

### Output Files
- `data/whitehead/processed/magma_mutation_trees.html` - Interactive mutation tree visualization
- `data/whitehead/processed/magma_correlation_table.csv` - Model correlation results
- `data/whitehead/processed/magma_correlation_table.tex` - LaTeX correlation table
- `data/whitehead/processed/magma_unified_model_correlations.svg` - Correlation plots

## Pipeline Workflow

1. **Data Assignment**: `magma_assign_antibodies.py` assigns sequences to antibodies using reference matching
2. **Dataset Unification**: `magma_unify_datasets.py` combines Kirby and Petersen data (1,130 sequences)
3. **Data Cleaning**: `magma_clean_unified_dataset.py` removes invalid sequences and handles replicates (→1,128 sequences)
4. **Model Scoring**: `magma_score_unified_dataset_clean.py` scores all sequences with 4 models
5. **Visualization**: `magma_generate_mutation_tree_report.py` creates interactive mutation trees
6. **Analysis**: Correlation scripts generate publication-ready tables and figures

## Dataset Statistics

- **6 antibody systems**: 4 Kirby UCAs + 2 Petersen mature antibodies
- **1,128 sequences** after cleaning
- **4 models**: DASM, ESM-650M, AbLang, ProGen2
- **Complete experimental data**: KD measurements and mutation tracking

## Usage

To run the complete pipeline:

```bash
# Activate environment
source ../netam/.venv/bin/activate

# Run data processing
python scripts/magma_unify_datasets.py
python scripts/magma_clean_unified_dataset.py

# Score with models
python scripts/magma_score_unified_dataset_clean.py

# Generate outputs
python scripts/magma_generate_mutation_tree_report.py
python scripts/magma_unified_model_correlation_analysis.py

# Validate pipeline (optional)
python scripts/magma_validation.py
```

## Key Features

- **Interactive visualization** with mutation tooltips
- **Cross-dataset analysis** comparing natural evolution vs. targeted mutagenesis
- **Statistical rigor** with replicate aggregation using geometric mean in log space
- **Publication-ready outputs** including LaTeX tables and SVG figures
- **Comprehensive model comparison** across 4 different approaches