# dasm-experiments

This repository contains the code and data for reproducing the results presented in:

**"Separating mutation from selection in antibody language models"** by Frederick A. Matsen IV, Will Dumm, Kevin Sung, Mackenzie M. Johnson, David Rich, Tyler Starr, Yun S. Song, Julia Fukuyama, Hugh K. Haddox

## Installation

First follow the instructions to do a developer install of [netam](https://github.com/matsengrp/netam) into a virtual environment (`venv` or `conda` both work).

Then clone the `dasm-experiments` repository and pip install from its root directory in that virtual environment:

    git clone git@github.com:matsengrp/dasm-experiments.git
    cd dasm-experiments
    make install

See `dnsmex/local.py` and modify it according to your local setup.

## Training models

If you are training models with parallel branch length optimization, we suggest that you first increase the number of file descriptors using `ulimit`, e.g.

    ulimit -n 8192

This will avoid `OSError: [Errno 24] Too many open files` triggered by the `multiprocessing` module.
You will need to do this in every shell that you are using for training models, or put it in a login `.*rc` file.

## Reproducing Manuscript Results

All analyses for the manuscript can be found in the locations listed below. For detailed setup and dependencies, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

### Main Analyses (in `notebooks/dasm_paper/`)
- **Nucleotide process in language models** → [`nt_process_in_llms.ipynb`](notebooks/dasm_paper/nt_process_in_llms.ipynb) (Figure 1)
- **Koenig et al. binding/expression benchmarks** → [`koenig.ipynb`](notebooks/dasm_paper/koenig.ipynb) (Table 1, Figures 2, S1-S5)
- **Perplexity comparison on natural sequences** → [`perplexity.ipynb`](notebooks/dasm_paper/perplexity.ipynb) (Figure 3)
- **Shanehsazzadeh et al. benchmarks** → [`shanehsazzadeh.ipynb`](notebooks/dasm_paper/shanehsazzadeh.ipynb) (Table 2, Figure S6)
- **Dataset summaries** → [`data_summaries.ipynb`](notebooks/dasm_paper/data_summaries.ipynb) (Table S1)

### MAGMA-seq Analysis Pipeline
- **MAGMA-seq correlation analysis** → [`data/whitehead/MAGMA_PIPELINE_STRUCTURE.md`](data/whitehead/MAGMA_PIPELINE_STRUCTURE.md) (Figures S7, Table S2)
- Includes unified dataset of 1,128 sequences from 6 antibody systems (Kirby et al. 2025, Petersen et al. 2024)
- Scripts in `scripts/magma_*.py` for data processing and model comparison

### Timing Benchmarks
- **Model timing comparison** → [`scripts/timing_direct_gpu.py`](scripts/timing_direct_gpu.py) and [`scripts/make_timing_table.py`](scripts/make_timing_table.py) (Table 3)

## Visualizations

Interactive visualizations of DASM selection factors for all antibodies in [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab) are available at <https://matsen.group/dasm-viz/v1/>.
(Thank you to OPIG for this resource, and to Will Hannon of the Bloom lab for writing [dms-viz](https://dms-viz.github.io/).)
