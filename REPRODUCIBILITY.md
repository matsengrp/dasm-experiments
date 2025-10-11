# Reproducibility Checklist for DASM Experiments

This document ensures all files and dependencies are present for reproducing the DASM manuscript results.

## ✅ Core Files Verification

### Notebooks (all present in `notebooks/dasm_paper/`)
- [x] `nt_process_in_llms.ipynb` → Figure 1 (nucleotide process in LLMs)
- [x] `koenig.ipynb` → Table 1 & Figures 2, S1-S5 (Koenig et al. analysis)
- [x] `perplexity.ipynb` → Figure 3 (perplexity comparison)
- [x] `shanehsazzadeh.ipynb` → Table 2 & Figure S6 (Shanehsazzadeh et al. analysis)
- [x] `data_summaries.ipynb` → Table S1 (dataset summaries)

### Scripts (all present in `scripts/`)
- [x] `timing_direct_gpu.py` → Generates timing data
- [x] `make_timing_table.py` → Creates Table 3 (timing table)
- [x] MAGMA-seq pipeline scripts (see MAGMA_PIPELINE_STRUCTURE.md)

### Model Files (all present in `dasm-train/trained_models/`)
- [x] `dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint.pth` (4M parameter model)
- [x] `dasm_4m-v1tangCC+v1vanwinkleheavyTrainCC+v1jaffePairedCC+v1vanwinklelightTrainCC1m-joint.yml` (config)
- [x] `dasm_1m-v1jaffeCC+v1tangCC-joint.pth` (1M parameter model for perplexity)
- [x] `dasm_1m-v1jaffeCC+v1tangCC-joint.yml` (config)

### Data Files (in `data/`)
- [x] `whitehead/MAGMA_PIPELINE_STRUCTURE.md` → Documents MAGMA-seq analysis
- [x] `whitehead/unified/magma_unified_dataset_clean.csv` → MAGMA-seq unified dataset
- [x] `whitehead/kirby/` → Kirby et al. 2025 data (4 antibody systems)
- [x] `whitehead/petersen/` → Petersen et al. 2024 data (2 antibody systems)
- [x] `sabdab_summary_2024-01-26_abid_info_resnums.tsv.gz` → SAbDab data for visualization

## 📦 External Data (Zenodo)

Due to size constraints (~128 MB), benchmark data, training data, and Rodriguez perplexity data are distributed separately via Zenodo.

**Zenodo DOI**: [10.5281/zenodo.17322891](https://doi.org/10.5281/zenodo.17322891)

**Download and Setup**:
```bash
# 1. Download from Zenodo
wget https://zenodo.org/records/17322891/files/dasm-experiments-data.tar.gz

# 2. Extract to your preferred location
tar -xzf dasm-experiments-data.tar.gz

# 3. Configure path in dnsmex/local_config.py
cp dnsmex/local_config.py.template dnsmex/local_config.py
# Edit local_config.py to set DATA_DIR="/path/to/dasm-experiments-data"
```

**Package Contents** (~128 MB uncompressed):

*Benchmark data (~3.9 MB)*:
- `FLAb/data/binding/Koenig2017_g6_Kd.csv` - Koenig binding measurements
- `FLAb/data/expression/Koenig2017_g6_er.csv` - Koenig expression measurements
- `FLAb/data/binding/Shanehsazzadeh2023_trastuzumab_zero_kd.csv` - Shanehsazzadeh binding
- `Koenig2017_g6_er.progen.csv` - ProGen2 scores (Koenig)
- `Shanehsazzadeh2023_trastuzumab_zero_kd.progen2-small.csv` - ProGen2 scores (Shanehsazzadeh)

*Training data (~121 MB)*:
- `v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_ConsCys_no-naive_DXSMVALID.csv.gz` (38 MB)
- `v3/v3convert_vanwinkle-170-igh_pcp_2025-03-05_MASKED_NI_train_no-naive_DXSMVALID_ConsCys.csv.gz` (8.3 MB)
- `v3/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-merged_pcp_2024-11-21_DXSMVALID_no-naive_ConsCys_HL.csv.gz` (23 MB)
- `v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k.csv.gz` (52 MB)

*Rodriguez perplexity data (~2.8 MB)*:
- `v3/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_noN_no-naive.csv.gz` (1.7 MB)
- `loris/rodriguez-igm/W-117_PRCONS-IGM_igblast.tsv` (1.1 MB)

*Documentation*:
- `README.md` - Detailed documentation with citations

**Note**: FLAb data is from commit [67738ee](https://github.com/Graylab/FLAb/tree/67738eea4841a1777b73609d56ddfa39de8d7360) (April 17, 2024). The FLAb repository has continued to receive updates; we use this specific commit for reproducibility.

## 🔧 Python Dependencies

### Core Dependencies
- `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `biopython`, `scipy`, `scikit-learn`

### External Model Packages
- **AbLang2**: `pip install ablang2` (used in: nt_process_in_llms, koenig, perplexity, shanehsazzadeh)
- **ESM**: `pip install fair-esm` (used in: koenig, shanehsazzadeh, timing scripts)
- **ProGen2** (optional): Requires separate environment with old dependencies
  - Pre-computed scores included in Zenodo package - **you can skip ProGen2 setup**
  - To generate new scores: See [`scripts/README_progen_setup.md`](scripts/README_progen_setup.md)

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   # Install netam in developer mode (in netam directory)
   pip install -e .

   # Install this package
   cd dasm-experiments
   make install

   # Install external model packages
   pip install ablang2 fair-esm
   ```

2. **Configure paths**:
   ```bash
   cp dnsmex/local_config.py.template dnsmex/local_config.py
   # Edit local_config.py:
   #   DATA_DIR = "/path/to/dasm-experiments-data"
   #   FIGURES_DIR = "/path/to/output/figures"
   #   DASM_TRAINED_MODELS_DIR = "dasm-train/trained_models"  # or absolute path
   ```

3. **Download Zenodo data** (see External Data section above)

4. **Run analyses**:
   ```bash
   # Verify models present
   ls dasm-train/trained_models/*.pth  # Should show 2 files

   # Run notebooks in order
   jupyter notebook notebooks/dasm_paper/
   # - nt_process_in_llms.ipynb
   # - koenig.ipynb
   # - perplexity.ipynb
   # - shanehsazzadeh.ipynb
   # - data_summaries.ipynb

   # Run timing benchmarks
   python scripts/timing_direct_gpu.py --device cpu --sequences 10
   python scripts/make_timing_table.py

   # MAGMA-seq analysis
   python scripts/magma_unified_model_correlation_analysis.py
   ```

## 📊 Expected Outputs

**From Notebooks**:
- SVG figures in FIGURES_DIR
- CSV files in `notebooks/dasm_paper/_output/`
- Correlation tables and statistics

**From Scripts**:
- `data/whitehead/processed/timing_table.tex`
- `data/whitehead/processed/direct_timing_results_*.csv`
- `data/whitehead/processed/magma_correlation_table.tex`
- `data/whitehead/processed/magma_unified_model_correlations.svg`

## ⚠️ Notes

- **Figure S7**: Multihit model exploration is in separate [`thrifty-experiments-1`](https://github.com/matsengrp/thrifty-experiments-1) repository
- **Memory**: DASM models (~4M parameters) run on CPU; ESM2-650M benefits from GPU
- **Troubleshooting**: Use `localify()` function to debug path resolution

## 📝 Citation

**"Separating mutation from selection in antibody language models"**
Frederick A. Matsen IV, Will Dumm, Kevin Sung, Mackenzie M. Johnson, David Rich, Tyler Starr, Yun S. Song, Julia Fukuyama, Hugh K. Haddox

---

**Last Updated**: 2025-10-09
**Repository**: https://github.com/matsengrp/dasm-experiments
