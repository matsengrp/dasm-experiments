# Kirby MAGMA-seq Assignment Report

## Overview

This report documents the assignment of MAGMA-seq sequences to antibody references from the Kirby dataset. The analysis covers sequence assignment success, binding affinity distributions, and mutation patterns.

**Dataset Source**: `/Users/matsen/re/dnsm-experiments-1/data/whitehead/kirby/original/AntibodySequences1.csv`
**Processed Data**: `/Users/matsen/re/dnsm-experiments-1/data/whitehead/kirby/processed/`

## Summary Statistics

- **Total Reference Antibodies**: 5
- **Successfully Assigned Antibodies**: 4
- **Assignment Success Rate**: 4/5 (80.0%)
- **Total Sequences Assigned**: 873

## Individual Antibody Assignment Results

### 002-S21F2_UCA

**Reference Sequences:**
- **VH**: `QEVQLVQSGAEVKKPGESLKISCKGSGYSFTSYWIGWVRQMPGKGLEWMG...` (length: 120)
- **VL**: `DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYD...` (length: 111)

**Assignment Statistics:**
- **Total Sequences**: 209
- **UCA Sequences**: 1
- **KD Range**: 25.81 - 2873.25 nM
- **KD Median**: 70.30 nM
- **Sequences with KD**: 209

**Mutation Distribution:**
- **0 mutations**: 1 sequences
- **2 mutations**: 1 sequences
- **3 mutations**: 5 sequences
- **4+ mutations**: 33 sequences

---

### Ab_1-20_UCA

**Reference Sequences:**
- **VH**: `EVQLVESGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSV...` (length: 117)
- **VL**: `DIQLTQSPSFLSASVGDRVTITCRASQGISSYLAWYQQKPGKAPKLLIYA...` (length: 106)

**Assignment Statistics:**
- **Total Sequences**: 13
- **UCA Sequences**: 1
- **KD Range**: 28.08 - 175.87 nM
- **KD Median**: 49.26 nM
- **Sequences with KD**: 13

**Mutation Distribution:**
- **0 mutations**: 1 sequences
- **1 mutations**: 4 sequences
- **2 mutations**: 6 sequences
- **3 mutations**: 2 sequences

---

### Ab_2-15_UCA

**Reference Sequences:**
- **VH**: `QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGW...` (length: 129)
- **VL**: `QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMI...` (length: 109)

**Assignment Statistics:**
- **Total Sequences**: 534
- **UCA Sequences**: 1
- **KD Range**: 3.30 - 2946.89 nM
- **KD Median**: 27.81 nM
- **Sequences with KD**: 534

**Mutation Distribution:**
- **0 mutations**: 1 sequences
- **1 mutations**: 2 sequences
- **2 mutations**: 9 sequences
- **3 mutations**: 30 sequences
- **4+ mutations**: 61 sequences

---

### C118_UCA

**Reference Sequences:**
- **VH**: `QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVAV...` (length: 127)
- **VL**: `QLVLTQSPSASASLGASVKLTCTLSSGHSSYAIAWHQQQPEKGPRYLMKL...` (length: 111)

**Assignment Statistics:**
- **Total Sequences**: 117
- **UCA Sequences**: 1
- **KD Range**: 19.47 - 470.30 nM
- **KD Median**: 60.75 nM
- **Sequences with KD**: 117

**Mutation Distribution:**
- **0 mutations**: 2 sequences
- **1 mutations**: 11 sequences
- **2 mutations**: 27 sequences
- **3 mutations**: 31 sequences
- **4+ mutations**: 31 sequences

---

## Mutation Analysis Summary

| Mutations | Total Sequences | Percentage |
|-----------|-----------------|------------|
| 0 | 5 | 0.6% |
| 1 | 17 | 1.9% |
| 2 | 43 | 4.9% |
| 3 | 68 | 7.8% |
| 4+ | 740 | 84.8% |

## Notes

- UCA sequences are identified by `is_uca = True` in the data
- Mutation counts represent differences from the assigned UCA reference
- KD values are binding dissociation constants in nanomolar (nM)
- Lower KD values indicate stronger binding
