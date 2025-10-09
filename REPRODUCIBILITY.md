# Reproducibility Checklist for DASM Experiments

This document provides a comprehensive checklist to ensure all files and dependencies are present for reproducing the results in the DASM manuscript.

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

## ⚠️ External Data Dependencies

### Data Files NOT in Repository (users must configure)
The following data files are referenced via `localify()` in `dnsmex/local_config.py` and must be obtained separately:

#### FLAb Benchmark Data
- `DATA_DIR/FLAb/data/binding/Koenig2017_g6_Kd.csv`
- `DATA_DIR/FLAb/data/expression/Koenig2017_g6_er.csv`
- `DATA_DIR/Koenig2017_g6_er.progen.csv`
- `DATA_DIR/Shanehsazzadeh2023_trastuzumab_zero_kd.progen2-small.csv`

**Source**: Download from FLAb benchmark repository at https://github.com/Graylab/FLAb

#### Rodriguez Dataset (for perplexity analysis)
- `DATA_DIR/loris/rodriguez-igm/W-117_PRCONS-IGM_igblast.tsv`
- Via `pcp_df_of_nickname("v1rodriguez")`: `v3/rodriguez-airr-seq-race-prod-NoWinCheck_igh_pcp_2024-11-12_MASKED_NI_noN_no-naive.csv.gz`

**Source**: Processed from Rodriguez et al. 2023 RACE-seq data

#### Training Data (for data_summaries.ipynb)
- Tang dataset: `v3/tang-deepshm-prod-NoWinCheck_igh_pcp_2024-10-29_MASKED_NI_ConsCys_no-naive_DXSMVALID.csv.gz`
- Jaffe paired dataset: `v3/wyatt-10x-1p5m_fs-all-NoWinCheck-UnmutInv-GTR-paired-merged_pcp_2024-11-21_DXSMVALID_no-naive_ConsCys_HL.csv.gz`
- Van Winkle light chain dataset: `v3/v3convert_vanwinkle-170-igk_pcp_2025-02-22_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k_CONCAT_vanwinkle-170-igl_pcp_2025-02-25_MASKED_NI_train_DXSMVALID_ConsCys_no-naive_downsample_500k.csv.gz`

**Source**: These are processed parent-child pair (PCP) datasets generated from the raw BCR sequencing data using netam's preprocessing pipeline.

## 📦 Python Package Dependencies

### Core Dependencies (from netam)
- `torch` (PyTorch)
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `biopython`
- `scipy`
- `scikit-learn`

### External Model Packages
- **AbLang2**: Required by `dnsmex.ablang_wrapper.AbLangWrapper`
  - Install: `pip install ablang2`
  - Used in: nt_process_in_llms, koenig, perplexity, shanehsazzadeh

- **ESM (Evolutionary Scale Modeling)**: Required by `dnsmex.esm_wrapper.esm2_wrapper_of_size`
  - Install: `pip install fair-esm`
  - Used in: koenig, shanehsazzadeh, timing scripts

- **ProGen2**: Required for ProGen2 benchmarks
  - Setup instructions in `scripts/README_progen_setup.md`
  - Used in: Shanehsazzadeh analysis (via pre-computed scores)

### Optional Dependencies
- `tqdm` (progress bars)
- `altair` (alternative plotting, for some notebooks)

## 🔧 Configuration Setup

### Required Configuration Steps

1. **Copy the local config template**:
   ```bash
   cp dnsmex/local_config.py.template dnsmex/local_config.py
   ```

2. **Edit `dnsmex/local_config.py`** to set your local paths:
   ```python
   CONFIG = {
       "DATA_DIR": "~/data",  # Location of FLAb and other external data
       "FIGURES_DIR": "~/writing/dasm-tex-1/figures/",  # Output for figures
       "DASM_TRAINED_MODELS_DIR": "~/re/dasm-experiments/dasm-train/trained_models",
       # ... etc
   }
   ```

3. **Key paths to configure**:
   - `DATA_DIR`: Root directory containing FLAb benchmark data and Rodriguez data
   - `FIGURES_DIR`: Where figure outputs will be saved (typically manuscript figures directory)
   - `DASM_TRAINED_MODELS_DIR`: Location of trained DASM models (can use relative path to this repo)

## 🚀 Running the Experiments

### Prerequisites
1. Install netam in developer mode
2. Install this package: `make install` from the repository root
3. Configure `dnsmex/local_config.py` as described above
4. Obtain external data files and place them in `DATA_DIR`

### Recommended Order

1. **Model availability check**:
   ```bash
   ls dasm-train/trained_models/*.pth
   # Should show 2 model files
   ```

2. **Run notebooks** (in `notebooks/dasm_paper/`):
   ```bash
   jupyter notebook
   # Open and run each notebook in order:
   # - nt_process_in_llms.ipynb
   # - koenig.ipynb
   # - perplexity.ipynb
   # - shanehsazzadeh.ipynb
   # - data_summaries.ipynb
   ```

3. **Run timing benchmarks**:
   ```bash
   # CPU timing
   python scripts/timing_direct_gpu.py --device cpu --sequences 10

   # GPU timing (if available)
   python scripts/timing_direct_gpu.py --device gpu --sequences 100

   # Generate timing table
   python scripts/make_timing_table.py
   ```

4. **MAGMA-seq analysis**:
   ```bash
   # See data/whitehead/MAGMA_PIPELINE_STRUCTURE.md for details
   python scripts/magma_unified_model_correlation_analysis.py
   ```

## 🔍 Verification Checklist

### File Existence
- [ ] All 5 key notebooks present in `notebooks/dasm_paper/`
- [ ] Both DASM model files (.pth and .yml) present in `dasm-train/trained_models/`
- [ ] MAGMA-seq data files present in `data/whitehead/`
- [ ] Scripts `timing_direct_gpu.py` and `make_timing_table.py` present

### Dependencies
- [ ] netam installed in developer mode
- [ ] dnsmex module importable (`python -c "import dnsmex"`)
- [ ] AbLang2 installed (`python -c "from ablang import pretrained"`)
- [ ] ESM installed (`python -c "import esm"`)
- [ ] All Python package dependencies installed

### Configuration
- [ ] `dnsmex/local_config.py` created and configured
- [ ] DATA_DIR path exists and contains FLAb data
- [ ] FIGURES_DIR path exists (or will be created)

### External Data
- [ ] FLAb benchmark data downloaded and placed in DATA_DIR
- [ ] Rodriguez data available (if running perplexity analysis)
- [ ] Training PCP data available (if running data_summaries.ipynb)

## ⚠️ Known Limitations

### Data Not Included in Repository
Due to size constraints, the following are NOT included:
1. **Raw training data** (~2M parent-child pairs used to train the models)
2. **FLAb benchmark datasets** (publicly available from FLAb repository)
3. **Rodriguez RACE-seq processed data** (large processed dataset)

### Figures Requiring thrifty-experiments-1
- Figure S7 (`fig:hc` - multihit model exploration) is made in the separate `thrifty-experiments-1` repository
- Reference: `multihit_model_exploration.ipynb` in https://github.com/matsengrp/thrifty-experiments-1

## 📊 Expected Outputs

### From Notebooks
- Multiple SVG figures in FIGURES_DIR
- CSV files in `notebooks/dasm_paper/_output/`
- Correlation tables and statistics

### From Scripts
- `data/whitehead/processed/timing_table.tex`
- `data/whitehead/processed/direct_timing_results_*.csv`
- `data/whitehead/processed/magma_correlation_table.tex`
- `data/whitehead/processed/magma_unified_model_correlations.svg`

## 🆘 Troubleshooting

### Import Errors
- Ensure netam is installed: `pip install -e .` in netam directory
- Ensure this package is installed: `make install` in this directory
- Check `dnsmex/local_config.py` exists

### Missing Data Files
- Verify paths in `dnsmex/local_config.py` are correct
- Check that DATA_DIR contains the FLAb data structure
- Use `localify()` function to debug path resolution

### Model Loading Errors
- Verify .pth and .yml files both exist for each model
- Check that netam version is compatible
- Ensure correct device (CPU/GPU) is available

### Memory Issues
- DASM models are small (~4M parameters) and should run on CPU
- ESM2-650M may require GPU for reasonable performance
- Consider reducing batch sizes in notebooks if needed

## 📝 Citation

If you use this code or data, please cite:

**"Separating mutation from selection in antibody language models"**
Frederick A. Matsen IV, Will Dumm, Kevin Sung, Mackenzie M. Johnson, David Rich, Tyler Starr, Yun S. Song, Julia Fukuyama, Hugh K. Haddox

---

**Last Updated**: 2025-10-09
**Repository**: https://github.com/matsengrp/dasm-experiments (private during review)
