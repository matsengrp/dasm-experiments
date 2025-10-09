"""Helper functions for analyzing Kirby et al. (2025) SARS-CoV-2 antibody data.

Paper: Kirby et al. "Retrospective SARS-CoV-2 human antibody development trajectories
are largely sparse and permissive" PNAS 2025
DOI: https://doi.org/10.1073/pnas.2412787122

Data sources:
- Binding data: DATA_DIR/whitehead/kirby/Kirby_PNAS2025_FLAB_filtered.csv
- WT/UCA sequences: DATA_DIR/whitehead/kirby/pnas.2412787122/Gene Blocks-Table 1.csv
  (CSV export from https://www.pnas.org/doi/suppl/10.1073/pnas.2412787122/suppl_file/pnas.2412787122.sd02.xlsx)

Note: The analysis notebook uses a simpler consensus-based approach rather than the
complex sequence parsing functions below, which are kept for potential future use.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from .local import localify
from .dms_helper import (
    protein_differences,
    get_site_by_site_consensus,
    sel_score_of_differences,
)
from .cached_model_wrappers import get_cached_esm_wrapper, get_cached_ablang_wrapper
from .remote_progen import get_cached_progen_wrapper


def load_kirby_data() -> pd.DataFrame:
    """Load the Kirby binding data.

    Returns:
        DataFrame with VH, VL, and KD columns
    """
    binding_path = localify("DATA_DIR/whitehead/kirby/Kirby_PNAS2025_FLAB_filtered.csv")
    return pd.read_csv(binding_path)


def load_gene_blocks_data() -> pd.DataFrame:
    """Load the Gene Blocks sequence data.

    Returns:
        DataFrame with Name and Sequence columns (nucleotide sequences)
    """
    gene_blocks_path = localify(
        "DATA_DIR/whitehead/kirby/pnas.2412787122/Gene Blocks-Table 1.csv"
    )
    return pd.read_csv(gene_blocks_path)


@dataclass
class KirbyPartitionBase:
    """Base class to handle analysis of a partition of Kirby data by antibody group."""

    antibody: str
    df: pd.DataFrame
    heavy_consensus: str = None
    light_consensus: str = None

    def __post_init__(self):
        # Consensus sequences are already in the dataframe from kirby_uca_consensus.py
        if "WT_VH" in self.df.columns:
            self.heavy_consensus = self.df["WT_VH"].iloc[0]
            self.light_consensus = self.df["WT_VL"].iloc[0]
        else:
            # Calculate if not present
            heavy_df = pd.DataFrame({"VH": self.df["VH"].tolist()})
            light_df = pd.DataFrame({"VL": self.df["VL"].tolist()})
            self.heavy_consensus = get_site_by_site_consensus(heavy_df, "VH")
            self.light_consensus = get_site_by_site_consensus(light_df, "VL")

        # Calculate differences from consensus
        self.df["heavy_differences"] = self.df["VH"].apply(
            lambda x: protein_differences(self.heavy_consensus, x)
        )
        self.df["light_differences"] = self.df["VL"].apply(
            lambda x: protein_differences(self.light_consensus, x)
        )
        self.df["heavy_difference_count"] = self.df["heavy_differences"].apply(len)
        self.df["light_difference_count"] = self.df["light_differences"].apply(len)
        self.df["total_difference_count"] = (
            self.df["heavy_difference_count"] + self.df["light_difference_count"]
        )

        # Convert KD to -log10 KD for better correlation
        if "KD" in self.df.columns:
            self.df["molar_KD"] = self.df["KD"] * 1e-9  # Convert nM to M
            self.df["-log10_KD"] = -np.log10(self.df["molar_KD"])

    def add_dasm_scores(self, crepe):
        """Add DASM scores to the dataframe."""
        # Get DASM scores for consensus
        [[dasm_heavy, dasm_light]] = crepe(
            [[self.heavy_consensus, self.light_consensus]]
        )
        log_dasm_heavy = torch.log(dasm_heavy.T)
        log_dasm_light = torch.log(dasm_light.T)

        # Calculate scores for variants using shared helper functions
        self.df["dasm_heavy"] = self.df["heavy_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_heavy, x)
        )
        self.df["dasm_light"] = self.df["light_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_light, x)
        )
        self.df["dasm"] = self.df["dasm_heavy"] + self.df["dasm_light"]

    def add_dasm_ft_mild_scores(self, crepe_ft_mild):
        """Add DASM-FT-mild scores to the dataframe."""
        # Get DASM-FT-mild scores for consensus
        [[dasm_ft_mild_heavy, dasm_ft_mild_light]] = crepe_ft_mild(
            [[self.heavy_consensus, self.light_consensus]]
        )
        log_dasm_ft_mild_heavy = torch.log(dasm_ft_mild_heavy.T)
        log_dasm_ft_mild_light = torch.log(dasm_ft_mild_light.T)

        # Calculate scores for variants using shared helper functions
        self.df["dasm_ft_mild_heavy"] = self.df["heavy_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_ft_mild_heavy, x)
        )
        self.df["dasm_ft_mild_light"] = self.df["light_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_ft_mild_light, x)
        )
        self.df["dasm_ft_mild"] = (
            self.df["dasm_ft_mild_heavy"] + self.df["dasm_ft_mild_light"]
        )

    def add_dasm_ft_strong_scores(self, crepe_ft_strong):
        """Add DASM-FT-strong scores to the dataframe."""
        # Get DASM-FT-strong scores for consensus
        [[dasm_ft_strong_heavy, dasm_ft_strong_light]] = crepe_ft_strong(
            [[self.heavy_consensus, self.light_consensus]]
        )
        log_dasm_ft_strong_heavy = torch.log(dasm_ft_strong_heavy.T)
        log_dasm_ft_strong_light = torch.log(dasm_ft_strong_light.T)

        # Calculate scores for variants using shared helper functions
        self.df["dasm_ft_strong_heavy"] = self.df["heavy_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_ft_strong_heavy, x)
        )
        self.df["dasm_ft_strong_light"] = self.df["light_differences"].apply(
            lambda x: sel_score_of_differences(log_dasm_ft_strong_light, x)
        )
        self.df["dasm_ft_strong"] = (
            self.df["dasm_ft_strong_heavy"] + self.df["dasm_ft_strong_light"]
        )

    def add_esm_scores(self):
        """Add ESM scores only (skip AbLang for now)."""
        # Prepare sequence pairs
        sequences = list(zip(self.df["VH"], self.df["VL"]))

        # Add ESM scores (650M model with VH+VL averaging) - cached by antibody group
        esm_wrapper = get_cached_esm_wrapper("650M")
        esm_scores = esm_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["esm"] = -np.array(esm_scores)  # Negate perplexity (higher = better)

        # Calculate UCA baseline for ESM
        uca_sequences = [(self.heavy_consensus, self.light_consensus)]
        self.uca_esm_score = -esm_wrapper.evaluate_antibodies(
            uca_sequences, antibody_group=f"{self.antibody}_UCA"
        )[0]

    def add_cached_model_scores(
        self, use_remote_esm: bool = True, use_remote_ablang: bool = True
    ):
        """Add cached model scores (ESM and AbLang) to the dataframe with persistent
        caching.

        Args:
            use_remote_esm: Whether to use remote GPU processing for ESM (faster)
            use_remote_ablang: Whether to use remote GPU processing for AbLang (faster)
        """
        # Prepare sequence pairs
        sequences = list(zip(self.df["VH"], self.df["VL"]))

        # Add ESM scores (650M model with VH+VL averaging) - cached by antibody group
        esm_wrapper = get_cached_esm_wrapper("650M", use_remote=use_remote_esm)
        esm_scores = esm_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["esm"] = -np.array(esm_scores)  # Negate perplexity (higher = better)

        # Add AbLang scores - cached by antibody group
        ablang_wrapper = get_cached_ablang_wrapper(use_remote=use_remote_ablang)
        ablang_scores = ablang_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["ablang"] = -np.array(
            ablang_scores
        )  # Negate perplexity (higher = better)

        # Calculate UCA baseline perplexities for reference lines
        uca_sequences = [(self.heavy_consensus, self.light_consensus)]
        self.uca_esm_score = -esm_wrapper.evaluate_antibodies(
            uca_sequences, antibody_group=f"{self.antibody}_UCA"
        )[0]
        self.uca_ablang_score = -ablang_wrapper.evaluate_antibodies(
            uca_sequences, antibody_group=f"{self.antibody}_UCA"
        )[0]

    def add_progen_scores(self, model_version="progen2-small"):
        """Add ProGen2 scores to the dataframe.

        Args:
            model_version: ProGen2 model version (small, medium, large, xlarge)
        """
        # Prepare sequence pairs
        sequences = list(zip(self.df["VH"], self.df["VL"]))

        # Add ProGen scores - cached by antibody group
        progen_wrapper = get_cached_progen_wrapper(model_version)
        progen_scores = progen_wrapper.evaluate_antibodies(
            sequences, antibody_group=self.antibody
        )
        self.df["progen"] = progen_scores  # Already negated in wrapper

        # Calculate UCA baseline for ProGen
        uca_sequences = [(self.heavy_consensus, self.light_consensus)]
        self.uca_progen_score = progen_wrapper.evaluate_antibodies(
            uca_sequences, antibody_group=f"{self.antibody}_UCA"
        )[0]

    def save(self, prefix):
        """Save the processed dataframe."""
        self.df.to_csv(f"{prefix}_{self.antibody}.csv", index=False)

    @classmethod
    def load(cls, prefix, antibody):
        """Load a partition from a CSV file."""
        df = pd.read_csv(f"{prefix}_{antibody}.csv")
        return cls(antibody=antibody, df=df)


class KirbyBinaryPartition(KirbyPartitionBase):
    """Binary classification analysis subclass of KirbyPartitionBase."""

    def load_binary_labels(self, binary_csv_path):
        """Load KD-based binary labels and merge with partition data."""
        binary_df = pd.read_csv(binary_csv_path)

        # Check if this is the new KD-based binary file or old flawed binary file
        if "binary_label" in binary_df.columns:
            # New KD-based binary classification - use as is
            print(f"Using KD-based binary classification from {binary_csv_path}")
        elif "KD" in binary_df.columns:
            # Old flawed binary file - rename column but warn
            print(f"WARNING: Using old binary classification from {binary_csv_path}")
            print("Consider using KD-based binary classification for better results")
            binary_df = binary_df.rename(columns={"KD": "binary_label"})
        else:
            raise ValueError(f"No binary label column found in {binary_csv_path}")

        # Merge with partition data - only keep binary label columns from binary_df
        binary_cols = ["VH", "VL", "binary_label", "KD_continuous"]
        binary_subset = binary_df[binary_cols]

        # Handle case where KD_continuous already exists in partition data
        if "KD_continuous" in self.df.columns:
            # Drop the existing KD_continuous to avoid merge conflicts
            self.df = self.df.drop(columns=["KD_continuous"])

        merged_df = self.df.merge(binary_subset, on=["VH", "VL"], how="inner")
        self.df = merged_df
        return self

    def get_top_n_sequences(self, model, n=5):
        """Get top N sequences by model score."""
        if model not in self.df.columns:
            raise ValueError(
                f"Model {model} scores not found. Available: {[col for col in self.df.columns if col in ['dasm', 'esm', 'ablang', 'progen']]}"
            )

        sorted_df = self.df.sort_values(model, ascending=False)
        return sorted_df.head(n)

    def calculate_precision_at_k(self, model, k_values=[5, 10, 20]):
        """Calculate precision@K for given model."""
        results = []

        for k in k_values:
            k_actual = min(k, len(self.df))
            top_k = self.get_top_n_sequences(model, k_actual)
            binders_in_top_k = (top_k["binary_label"] == 1).sum()
            precision = binders_in_top_k / k_actual

            results.append(
                {
                    "model": model,
                    "k": k,
                    "k_actual": k_actual,
                    "binders_in_top_k": binders_in_top_k,
                    "precision": precision,
                    "antibody": self.antibody,
                }
            )

        return pd.DataFrame(results)

    def calculate_random_baseline(self, k_values=[5, 10, 20]):
        """Calculate expected precision for random selection."""
        baseline_precision = (self.df["binary_label"] == 1).mean()

        results = []
        for k in k_values:
            k_actual = min(k, len(self.df))
            expected_binders = baseline_precision * k_actual

            results.append(
                {
                    "model": "Random",
                    "k": k,
                    "k_actual": k_actual,
                    "binders_in_top_k": expected_binders,
                    "precision": baseline_precision,
                    "antibody": self.antibody,
                }
            )

        return pd.DataFrame(results)

    def summary_table(
        self, models=["dasm", "esm", "ablang", "progen"], k_values=[5, 10, 20]
    ):
        """Generate summary table of top-N performance."""
        all_results = []

        # Add results for each model
        for model in models:
            if model in self.df.columns:
                model_results = self.calculate_precision_at_k(model, k_values)
                all_results.append(model_results)

        # Add random baseline
        random_results = self.calculate_random_baseline(k_values)
        all_results.append(random_results)

        return pd.concat(all_results, ignore_index=True)


def score_mutation_relative_to_parent(parent, mutant, crepe):
    """Score a mutant sequence relative to its parent (not UCA).

    This enables fair comparison with language models by allowing DASM to
    evaluate mutations in the context of the evolved sequence rather than
    always relative to the distant UCA.

    Args:
        parent: (heavy, light) tuple of parent sequences
        mutant: (heavy, light) tuple of mutant sequences
        crepe: DASM model (loaded with load_crepe)

    Returns:
        Combined DASM score for heavy + light mutations relative to parent
    """
    # Get DASM output for parent sequence (parent goes through transformer)
    parent_heavy, parent_light = parent
    mutant_heavy, mutant_light = mutant

    # Run parent through DASM to get selection factors
    [[dasm_heavy, dasm_light]] = crepe([[parent_heavy, parent_light]])

    # Convert to log space for scoring
    log_dasm_heavy = torch.log(dasm_heavy.T)
    log_dasm_light = torch.log(dasm_light.T)

    # Calculate differences between parent and mutant
    heavy_diffs = protein_differences(parent_heavy, mutant_heavy)
    light_diffs = protein_differences(parent_light, mutant_light)

    # Score mutations relative to parent
    heavy_score = sel_score_of_differences(log_dasm_heavy, heavy_diffs)
    light_score = sel_score_of_differences(log_dasm_light, light_diffs)

    return heavy_score + light_score
