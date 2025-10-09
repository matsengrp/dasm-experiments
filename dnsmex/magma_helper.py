#!/usr/bin/env python3
"""Helper functions for MAGMA-seq analysis supporting both Kirby and Petersen datasets.

This module provides unified functionality for:
- Proper DASM scoring with reference sequences
- Handling different experimental designs (UCA->mature vs mature->variants)
- Computing mutations and selection scores
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from .dms_helper import sel_score_of_differences, protein_differences


class MAGMAPartition:
    """Base class for MAGMA-seq partition analysis."""

    def __init__(
        self, antibody: str, df: pd.DataFrame, reference_vh: str, reference_vl: str
    ):
        """Initialize MAGMA partition.

        Args:
            antibody: Antibody name
            df: DataFrame with VH, VL, KD columns
            reference_vh: Reference VH sequence (UCA for Kirby, mature for Petersen)
            reference_vl: Reference VL sequence
        """
        self.antibody = antibody
        self.df = df.copy()
        self.reference_vh = reference_vh
        self.reference_vl = reference_vl

        # Validate sequence lengths before processing
        self.validate_sequence_lengths()

        # Calculate differences from reference if not already present
        if "heavy_differences" not in self.df.columns:
            self.calculate_differences()

    def validate_sequence_lengths(self):
        """Validate that all sequences have the same length as reference sequences."""
        ref_vh_len = len(self.reference_vh)
        ref_vl_len = len(self.reference_vl)

        # Check VH lengths
        vh_lengths = self.df["VH"].str.len().unique()
        if len(vh_lengths) > 1 or vh_lengths[0] != ref_vh_len:
            bad_seqs = self.df[self.df["VH"].str.len() != ref_vh_len]
            raise ValueError(
                f"❌ CRITICAL: VH length mismatch in {self.antibody}!\n"
                f"   Reference VH: {ref_vh_len} chars\n"
                f"   Found VH lengths: {vh_lengths.tolist()}\n"
                f"   {len(bad_seqs)} sequences have wrong length\n"
                f"   Pipeline halted - fix reference sequences or data."
            )

        # Check VL lengths
        vl_lengths = self.df["VL"].str.len().unique()
        if len(vl_lengths) > 1 or vl_lengths[0] != ref_vl_len:
            bad_seqs = self.df[self.df["VL"].str.len() != ref_vl_len]
            raise ValueError(
                f"❌ CRITICAL: VL length mismatch in {self.antibody}!\n"
                f"   Reference VL: {ref_vl_len} chars\n"
                f"   Found VL lengths: {vl_lengths.tolist()}\n"
                f"   {len(bad_seqs)} sequences have wrong length\n"
                f"   Pipeline halted - fix reference sequences or data."
            )

    def calculate_differences(self):
        """Calculate mutations from reference sequence."""

        def safe_protein_differences(ref_seq, var_seq):
            """Calculate differences with strict length check."""
            if len(ref_seq) != len(var_seq):
                raise ValueError(
                    f"❌ CRITICAL: Length mismatch detected for {self.antibody}!\n"
                    f"   Reference: {len(ref_seq)} chars\n"
                    f"   Variant:   {len(var_seq)} chars\n"
                    f"   This indicates a serious data integrity issue.\n"
                    f"   Pipeline halted - fix reference sequences or data cleaning."
                )
            return protein_differences(ref_seq, var_seq)

        self.df["heavy_differences"] = self.df["VH"].apply(
            lambda x: safe_protein_differences(self.reference_vh, x)
        )
        self.df["light_differences"] = self.df["VL"].apply(
            lambda x: safe_protein_differences(self.reference_vl, x)
        )

        # Count total mutations
        self.df["mutations_from_reference"] = self.df["heavy_differences"].apply(
            len
        ) + self.df["light_differences"].apply(len)

    def add_dasm_scores(self, crepe):
        """Add proper DASM scores using reference sequence.

        Args:
            crepe: Loaded DASM model
        """
        # Get DASM scores for reference sequence
        [[dasm_heavy, dasm_light]] = crepe([[self.reference_vh, self.reference_vl]])

        # Transpose and log transform
        log_dasm_heavy = torch.log(dasm_heavy.T)
        log_dasm_light = torch.log(dasm_light.T)

        # Calculate scores for variants based on their mutations
        self.df["dasm_heavy"] = self.df["heavy_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_heavy, x)
        )
        self.df["dasm_light"] = self.df["light_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_light, x)
        )
        self.df["dasm"] = self.df["dasm_heavy"] + self.df["dasm_light"]

        return self.df

    def add_cached_model_scores(
        self, use_remote_esm: bool = True, use_remote_ablang: bool = True
    ):
        """Add cached model scores (ESM and AbLang) with proper antibody grouping.

        Args:
            use_remote_esm: Whether to use remote GPU processing for ESM
            use_remote_ablang: Whether to use remote GPU processing for AbLang
        """
        from .cached_model_wrappers import (
            get_cached_esm_wrapper,
            get_cached_ablang_wrapper,
        )

        # Prepare sequence pairs
        sequences = list(zip(self.df["VH"], self.df["VL"]))

        # Add ESM scores with antibody grouping
        esm_wrapper = get_cached_esm_wrapper("650M", use_remote=use_remote_esm)
        esm_scores = esm_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["esm"] = -np.array(esm_scores)  # Negate perplexity (higher = better)

        # Add AbLang scores with antibody grouping
        ablang_wrapper = get_cached_ablang_wrapper(use_remote=use_remote_ablang)
        ablang_scores = ablang_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["ablang"] = -np.array(
            ablang_scores
        )  # Negate perplexity (higher = better)

        return self.df

    def add_progen_scores(self, model_version="progen2-small"):
        """Add ProGen2 scores with proper antibody grouping.

        Args:
            model_version: ProGen2 model version (small, medium, large, xlarge)
        """
        from .remote_progen import get_cached_progen_wrapper

        # Prepare sequence pairs
        sequences = list(zip(self.df["VH"], self.df["VL"]))

        # Add ProGen scores with antibody grouping
        progen_wrapper = get_cached_progen_wrapper(model_version)
        progen_scores = progen_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["progen"] = progen_scores  # Already negated in wrapper

        return self.df


def load_reference_sequences(dataset: str = "both") -> Dict[str, Tuple[str, str]]:
    """Load reference sequences for all antibodies.

    Args:
        dataset: 'kirby', 'petersen', or 'both'

    Returns:
        Dictionary mapping antibody names to (VH, VL) tuples
    """
    references = {}

    # Kirby reference sequences (UCAs)
    if dataset in ["kirby", "both"]:
        kirby_refs = pd.read_csv("data/whitehead/kirby/original/AntibodySequences1.csv")

        # Group by antibody and create VH/VL pairs
        for antibody, group in kirby_refs.groupby("Antibody"):
            vh_seq = group[group["Chain"] == "VH"]["Sequence"].iloc[0]
            vl_seq = group[group["Chain"] == "VL"]["Sequence"].iloc[0]
            references[antibody] = (vh_seq, vl_seq)

    # Petersen reference sequences (mature antibodies)
    if dataset in ["petersen", "both"]:
        petersen_refs = pd.read_csv(
            "data/whitehead/petersen/original/AntibodySequences2.csv"
        )

        # Group by antibody and create VH/VL pairs
        for antibody, group in petersen_refs.groupby("Antibody"):
            if antibody in ["222-1C06", "319-345"]:  # Only these two are in our dataset
                vh_seq = group[group["Chain"] == "VH"]["Sequence"].iloc[0]
                vl_seq = group[group["Chain"] == "VL"]["Sequence"].iloc[0]
                references[antibody] = (vh_seq, vl_seq)

    return references


def create_magma_partition(
    antibody: str, df: pd.DataFrame, references: Optional[Dict] = None
) -> MAGMAPartition:
    """Create appropriate MAGMA partition for an antibody.

    Args:
        antibody: Antibody name
        df: DataFrame with sequences and KD values
        references: Optional dictionary of reference sequences

    Returns:
        MAGMAPartition instance
    """
    if references is None:
        references = load_reference_sequences()

    if antibody not in references:
        raise ValueError(f"No reference sequence found for {antibody}")

    ref_vh, ref_vl = references[antibody]

    return MAGMAPartition(antibody, df, ref_vh, ref_vl)


def score_unified_dataset_with_dasm(df: pd.DataFrame, dasm_model) -> pd.DataFrame:
    """Score unified dataset with proper DASM scoring.

    Args:
        df: Unified dataset with VH, VL, KD, antibody columns
        dasm_model: Loaded DASM model

    Returns:
        DataFrame with added DASM scores
    """
    # Load all reference sequences
    references = load_reference_sequences("both")

    # Process each antibody group separately
    scored_dfs = []

    for antibody in df["antibody"].unique():
        print(f"  Processing {antibody}...")

        # Get subset for this antibody
        antibody_df = df[df["antibody"] == antibody].copy()

        # Create partition with proper reference
        partition = create_magma_partition(antibody, antibody_df, references)

        # Add DASM scores
        partition.add_dasm_scores(dasm_model)

        scored_dfs.append(partition.df)

    # Combine all scored dataframes
    result_df = pd.concat(scored_dfs, ignore_index=True)

    # Ensure we preserve all original columns
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = df[col]

    return result_df


def score_unified_dataset_with_all_models(
    df: pd.DataFrame,
    dasm_model,
    use_remote_esm: bool = True,
    use_remote_ablang: bool = True,
) -> pd.DataFrame:
    """Score unified dataset with all models using proper infrastructure.

    Args:
        df: Unified dataset with VH, VL, KD, antibody columns
        dasm_model: Loaded DASM model
        use_remote_esm: Whether to use remote GPU processing for ESM
        use_remote_ablang: Whether to use remote GPU processing for AbLang

    Returns:
        DataFrame with added model scores
    """
    # Load all reference sequences
    references = load_reference_sequences("both")

    # Process each antibody group separately
    scored_dfs = []

    for antibody in df["antibody"].unique():
        print(f"  Processing {antibody}...")

        # Get subset for this antibody
        antibody_df = df[df["antibody"] == antibody].copy()

        # Create partition with proper reference
        partition = create_magma_partition(antibody, antibody_df, references)

        # Add all model scores with proper antibody grouping
        partition.add_dasm_scores(dasm_model)
        partition.add_cached_model_scores(use_remote_esm, use_remote_ablang)
        partition.add_progen_scores("progen2-small")

        scored_dfs.append(partition.df)

    # Combine all scored dataframes
    result_df = pd.concat(scored_dfs, ignore_index=True)

    # Ensure we preserve all original columns
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = df[col]

    return result_df


def get_dataset_type(antibody: str) -> str:
    """Determine if antibody is from Kirby or Petersen dataset."""
    if antibody in [
        "002-S21F2_UCA",
        "Ab_1-20_UCA",
        "Ab_2-15_UCA",
        "C118_UCA",
        "CC12.1_UCA",
    ]:
        return "Kirby"
    elif antibody in ["222-1C06", "319-345"]:
        return "Petersen"
    else:
        raise ValueError(f"Unknown antibody: {antibody}")
