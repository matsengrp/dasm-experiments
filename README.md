# dasm-experiments

This repository contains the code and data for reproducing the results presented in:

**"Separating mutation from selection in antibody language models"** by Frederick A. Matsen IV, Will Dumm, Kevin Sung, Mackenzie M. Johnson, David Rich, Tyler Starr, Yun S. Song, Julia Fukuyama, Hugh K. Haddox

**For peer reviewers: this repository will be opened to the public after the paper is accepted. We provide a version of the repository as a `.tar.gz` file for the purposes of peer review.**

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

## MAGMA-seq Analysis Pipeline

This repository includes a complete analysis pipeline for MAGMA-seq antibody datasets (Kirby et al. 2025, Petersen et al. 2024):

- **Unified dataset**: Sequences from 6 antibody systems (Kirby + Petersen datasets)
- **Model scoring**: DASM, ESM2, AbLang2, and ProGen2 predictions for all sequences
- **Interactive visualization**: Complete mutation tree analysis with binding affinity data

For detailed documentation, see [`data/whitehead/MAGMA_PIPELINE_STRUCTURE.md`](data/whitehead/MAGMA_PIPELINE_STRUCTURE.md).

## Visualizations

Please see <https://matsen.group/dasm-viz/v1/> for visualizations of DASM selection factors for all of the antibodies in the [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab).
(Thank you to OPIG for this resource, and to Will Hannon of the Bloom lab for writing [dms-viz](https://dms-viz.github.io/).)
