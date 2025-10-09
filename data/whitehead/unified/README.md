# MAGMA-seq Unified Dataset

## Overview

This directory contains the unified MAGMA-seq dataset combining both Kirby and Petersen datasets for consistent model scoring and analysis.

## Dataset Contents

- **Total sequences**: 1,915
- **Total antibodies**: 6
- **Kirby dataset**: 873 sequences (UCA→mature evolution)
- **Petersen dataset**: 1,042 sequences (mature CDR saturation)

## Key Files

- `magma_unified_dataset.csv` - Unified dataset ready for model scoring
- `magma_unified_scored.csv` - Dataset with all model predictions (created after scoring)
- `dataset_metadata.txt` - Detailed statistics and source information
- `README.md` - This documentation

## Column Descriptions

### Core Sequence Data
- `sequence_id`: Unique identifier (VH|VL concatenation)
- `VH`: Heavy chain amino acid sequence
- `VL`: Light chain amino acid sequence
- `KD`: Binding dissociation constant (nM)
- `antibody`: Antibody name
- `dataset`: "Kirby" or "Petersen"

### Reference and Mutation Information
- `reference_VH`: **Reference heavy chain sequence** (baseline for mutation counting)
- `reference_VL`: **Reference light chain sequence** (baseline for mutation counting)  
- `mutations_from_reference`: Number of mutations from reference sequence
- `heavy_mutations`: List of mutations in heavy chain (e.g., ['D10G', 'V23A'])
- `light_mutations`: List of mutations in light chain

### **CRITICAL NOTE: Reference Sequences Differ by Dataset**

#### Kirby Dataset:
- `reference_VH/VL` = **UCA sequences** (unmutated common ancestor)
- `mutations_from_reference` = mutations **away from UCA** (evolutionary progression)
- **Interpretation**: More mutations = more evolved antibody (generally better binding)

#### Petersen Dataset:
- `reference_VH/VL` = **Mature antibody sequences** (optimized, from Dr. Whitehead)
- `mutations_from_reference` = mutations **away from mature** (CDR mutagenesis)
- **Interpretation**: More mutations = more deviation from optimized sequence (generally worse binding)

### Additional Metadata
- `similarity_score`: Assignment confidence score
- `binding_score`: Derived binding metric
- `experimental_design`: "UCA_to_mature" or "mature_CDR_saturation"

### Model Predictions (after scoring)
- `dasm_base`: Base DASM model prediction
- `dasm_ft`: Fine-tuned DASM model prediction  
- `esm`: ESM-650M model prediction
- `ablang`: AbLang model prediction
- `progen`: ProGen2-small model prediction

## Experimental Design Context

### Kirby et al. PNAS 2025
- **Design**: Reconstructed evolutionary trajectories from UCA to mature antibodies
- **Coverage**: 0-12 mutations from UCA
- **Pattern**: KD generally decreases with mutations (better binding through evolution)
- **Purpose**: Study natural antibody evolution and affinity maturation

### Petersen et al. Nature Communications 2024  
- **Design**: CDR-targeted site-saturation mutagenesis around mature antibodies
- **Coverage**: 0-4 mutations from mature sequence
- **Pattern**: KD generally increases with mutations (worse binding from optimized baseline)
- **Purpose**: Systematic exploration of CDR sequence-function relationships

## Usage Notes

1. **Reference interpretation**: Always consider the dataset when interpreting `mutations_from_reference`
2. **Binding trends**: Expect opposite KD trends between datasets due to different reference points
3. **Model evaluation**: Can compare model performance across both evolutionary (Kirby) and mutagenic (Petersen) contexts
4. **Unified analysis**: Enables cross-dataset comparison of sequence-function relationships

## Analysis Potential

This unified dataset enables:
- **Model benchmarking**: Consistent evaluation of DASM, ESM, AbLang, ProGen across experimental paradigms
- **Evolutionary analysis**: Compare natural evolution (Kirby) vs. systematic mutagenesis (Petersen)
- **CDR importance**: Analyze CDR-specific patterns across different antibodies
- **Binding prediction**: Train and evaluate models on diverse antibody-antigen interactions

---

*Created by magma_unify_datasets.py*  
*Model scoring via magma_score_unified_dataset_clean.py*