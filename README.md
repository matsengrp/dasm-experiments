# Inviting Darwin into antibody language models

This repository contains the code and data for reproducing the results presented in:

**"Inviting Darwin into antibody language models"** by Frederick A. Matsen IV, Will Dumm, Kevin Sung, Mackenzie M. Johnson, David Rich, Tyler Starr, Yun S Song, Julia Fukuyama, Hugh K. Haddox

**For peer reviewers: this repository will be opened to the public after the paper is accepted. We provide a version of the repository as a `.tar.gz` file for the purposes of peer review.**

## Overview

This repository implements **Deep Amino acid Selection Models (DASM)**, a novel approach that separates mutation and selection processes in antibody evolution. Traditional language models conflate these processes, but our method models them separately:

- **Mutation Model**: Captures neutral somatic hypermutation using nucleotide-level biases
- **Selection Model**: Uses transformer networks to predict amino acid-specific selection factors

## Repository Structure

```
├── notebooks/dasm_paper/          # Analysis notebooks for all figures and tables
│   ├── nt_process_in_llms.ipynb   # Figure: Nucleotide process in LLMs
│   ├── koenig.ipynb               # Koenig et al. analysis (Table & Figures)
│   ├── perplexity.ipynb           # Perplexity comparison analysis
│   ├── shanehsazzadeh.ipynb       # Shanehsazzadeh et al. analysis
│   └── data_summaries.ipynb       # Dataset summary tables
├── dnsmex/                        # Helper code and analysis utilities
├── scripts/                       # Analysis and processing scripts
│   ├── timing_direct_gpu.py       # Model timing analysis
│   ├── make_timing_table.py       # Timing table generation
│   └── magma_*.py                 # MAGMA-seq analysis pipeline
├── dasm-train/                    # Training configuration and models
│   ├── Snakefile                  # Training pipeline
│   ├── config.yml                 # Training configuration
│   └── trained_models/            # Pre-trained DASM models (2 key models)
├── data/                          # Reference data and experimental datasets
└── README.md                      # This file
```

## Installation

First follow the instructions to do a developer install of [netam](https://github.com/matsengrp/netam) into a virtual environment (`venv` or `conda` both work).

Then install this package from its root directory in that virtual environment:

```bash
cd dasm-experiments-submit
make install
```

See `dnsmex/local.py` and modify it according to your local setup.

## Reproducing Results

### Running Notebooks

Execute notebooks to reproduce figures and tables:

```bash
cd notebooks/dasm_paper/
jupyter notebook
```

Each notebook corresponds to specific figures/tables mentioned in the manuscript:

- `nt_process_in_llms.ipynb` → Figure showing nucleotide process analysis
- `koenig.ipynb` → Koenig et al. binding/expression analysis 
- `perplexity.ipynb` → Perplexity comparison plots
- `shanehsazzadeh.ipynb` → Shanehsazzadeh et al. benchmark
- `data_summaries.ipynb` → Dataset statistics tables

### MAGMA-seq Analysis

The MAGMA-seq analysis pipeline (scatter plots and correlation tables) can be run using:

```bash
cd scripts/
python magma_unified_model_correlation_analysis.py
```

See `data/whitehead/MAGMA_PIPELINE_STRUCTURE.md` for detailed pipeline documentation.

This repository includes a complete analysis pipeline for MAGMA-seq antibody datasets (Kirby et al. 2025, Petersen et al. 2024):

- **Unified dataset**: 1,915 sequences from 6 antibody systems (Kirby + Petersen datasets)
- **Model scoring**: DASM, ESM, AbLang, and ProGen2 predictions for all sequences
- **Interactive visualization**: Complete mutation tree analysis with binding affinity data
- **UCA analysis**: Complete reconstruction and evolutionary analysis of Petersen antibodies

## Models

This repository includes two pre-trained DASM models:

1. **dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint**
   - 4M parameter model trained on multiple paired datasets
   - Used for Koenig et al. analysis

2. **dasm_1m-v1jaffeCC+v1tangCC-joint** 
   - 1M parameter model for perplexity comparisons
   - Trained on combined heavy chain datasets

Models can be loaded using the netam framework:

```python
from netam.framework import load_crepe
crepe = load_crepe("dasm-train/trained_models/[model_name]")
```

## Training Models

If you are training models with parallel branch length optimization, we suggest that you first increase the number of file descriptors using `ulimit`, e.g.

```bash
ulimit -n 8192
```

This will avoid `OSError: [Errno 24] Too many open files` triggered by the `multiprocessing` module.
You will need to do this in every shell that you are using for training models, or put it in a login `.*rc` file.

## Contact

For questions or issues, please contact [Erick Matsen](https://matsen.group) or open an issue in this repository.