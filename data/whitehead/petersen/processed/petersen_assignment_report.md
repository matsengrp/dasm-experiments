# Petersen MAGMA-seq Assignment Report

## Overview

This report documents the assignment of MAGMA-seq sequences to antibody references from the Petersen dataset. This dataset uses a mature→UCA experimental design where sequences represent the evolutionary path from mature antibodies back to their unmutated common ancestors (UCAs).

**Dataset Source**: `/Users/matsen/re/dnsm-experiments-1/data/whitehead/petersen/original/AntibodySequences2.csv`
**Processed Data**: `/Users/matsen/re/dnsm-experiments-1/data/whitehead/petersen/processed/`

## Summary Statistics

- **Total Reference Antibodies**: 9
- **Successfully Assigned Antibodies**: 2
- **Assignment Success Rate**: 2/9 (22.2%)
- **Total Sequences Assigned**: 1,042

## Individual Antibody Assignment Results

### 222-1C06

**Reference Sequences:**
- **VH**: `EVQLVESGGDLVQPGGSLRLSCVVSGFTFSTYSMNWVRQAPGKGLEWVSY...` (length: 122)
- **VL**: `EIVMTQSPATLSVSPGERATLSCRASQTIRSDLAWYQQKPGQPPRLIIYG...` (length: 108)

**Assignment Statistics:**
- **Total Sequences**: 373
- **UCA Sequences**: 1
- **Maximum Mutations Observed**: 4
- **Potential UCA Candidates**: 2 (sequences with max mutations)
- **KD Range**: 8.73 - 1000.00 nM
- **KD Median**: 32.90 nM
- **Sequences with KD**: 373

**Mutation Distribution:**
- **0 mutations**: 100 sequences
- **1 mutations**: 171 sequences
- **2 mutations**: 85 sequences
- **3 mutations**: 15 sequences
- **4 mutations**: 2 sequences

---

### 319-345

**Reference Sequences:**
- **VH**: `EVQLQESGGGLVRPGGTLRLSCAASGFSFSNYNMYWVRQAPGKGLEWVSS...` (length: 123)
- **VL**: `EIVMTQSPATLSVSPGERATLSCRASQSVNSNLAWYQQRPGQAPRLLIYT...` (length: 108)

**Assignment Statistics:**
- **Total Sequences**: 669
- **UCA Sequences**: 1
- **Maximum Mutations Observed**: 4
- **Potential UCA Candidates**: 9 (sequences with max mutations)
- **KD Range**: 2.68 - 2490.81 nM
- **KD Median**: 19.05 nM
- **Sequences with KD**: 669

**Mutation Distribution:**
- **0 mutations**: 172 sequences
- **1 mutations**: 286 sequences
- **2 mutations**: 167 sequences
- **3 mutations**: 35 sequences
- **4 mutations**: 9 sequences

---

## Mutation Analysis Summary

| Mutations | Total Sequences | Percentage |
|-----------|-----------------|------------|
| 0 | 272 | 26.1% |
| 1 | 457 | 43.9% |
| 2 | 252 | 24.2% |
| 3 | 50 | 4.8% |
| 4 | 11 | 1.1% |

## Experimental Design Notes

- **Mature→UCA Design**: Sequences represent evolutionary trajectories from mature antibodies to UCAs
- **UCA Identification**: True UCAs should have maximum mutation counts and represent ancestral sequences
- **Mutation Direction**: Higher mutation counts indicate sequences closer to the true UCA
- **Binding Evolution**: KD values show how binding affinity changes along evolutionary paths

## Notes

- UCA sequences are identified by `is_uca = True` in the data
- Mutation counts represent differences from the assigned reference
- KD values are binding dissociation constants in nanomolar (nM)
- Lower KD values indicate stronger binding
