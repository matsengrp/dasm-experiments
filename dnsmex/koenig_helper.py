from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.Data import CodonTable
from typing import Tuple, Dict
import torch

from netam.codon_table import make_codon_neighbor_indicator
from netam.sequences import translate_sequence, AA_STR_SORTED
from .dms_helper import get_site_by_site_consensus, protein_differences


class KoenigDataset:
    KOENIG_AA_ORDER = list("YWFLIVAMKRHDESTNQPGC")

    def __init__(
        self,
        csv_path: str,
        fitness_label: str,
        fitness_column: str,
        log_transform: bool = True,
    ):
        """Initialize with path to either binding or expression CSV file.

        Args:
            csv_path: Path to CSV file with Koenig data
            fitness_label: Label for fitness values (for plots)
            fitness_column: Name of column with fitness values
            log_transform: Whether to log-transform fitness values
        """
        self.df = pd.read_csv(csv_path)
        self.fitness_label = fitness_label
        self.fitness_column = fitness_column
        self.log_transform = log_transform

        self.heavy_consensus = self._get_consensus("heavy")
        self.light_consensus = self._get_consensus("light")

        self._process_differences()

        self.heavy_dms_style_df = self._create_dms_style_df("heavy")
        self.light_dms_style_df = self._create_dms_style_df("light")

        if self.log_transform:
            self.heavy_dms_style_df = np.log(self.heavy_dms_style_df)
            self.light_dms_style_df = np.log(self.light_dms_style_df)

        self.heavy_consensus_backtrans = self._backtranslate_consensus("heavy")
        self.light_consensus_backtrans = self._backtranslate_consensus("light")

        self.heavy_neighbor_indicator = self._create_neighbor_indicator(
            self.heavy_consensus_backtrans, chain="heavy"
        )
        self.light_neighbor_indicator = self._create_neighbor_indicator(
            self.light_consensus_backtrans, chain="light"
        )

    def _get_consensus(self, chain: str) -> str:
        """Get consensus sequence for specified chain."""
        return get_site_by_site_consensus(self.df, chain)

    def _process_differences(self):
        """Process differences between sequences and consensus."""
        self.df["heavy_differences"] = self.df["heavy"].apply(
            lambda x: protein_differences(self.heavy_consensus, x)
        )
        self.df["light_differences"] = self.df["light"].apply(
            lambda x: protein_differences(self.light_consensus, x)
        )

        # Calculate difference counts for each chain
        self.df["heavy_difference_count"] = self.df["heavy_differences"].apply(len)
        self.df["light_difference_count"] = self.df["light_differences"].apply(len)
        self.df["difference_count"] = (
            self.df["heavy_difference_count"] + self.df["light_difference_count"]
        )

        # Filter dataframes for each chain - mutations only in that chain
        self.heavy_df = self.df[
            (self.df["heavy_difference_count"] == 1)
            & (self.df["light_difference_count"] == 0)
        ].copy()
        self.light_df = self.df[
            (self.df["light_difference_count"] == 1)
            & (self.df["heavy_difference_count"] == 0)
        ].copy()

        # Extract mutation information for each chain
        if len(self.heavy_df) > 0:
            self.heavy_df["difference"] = self.heavy_df["heavy_differences"].apply(
                lambda x: x[0]
            )
            self.heavy_df["site"] = self.heavy_df["difference"].apply(
                lambda x: int(x[1:-1])
            )
            self.heavy_df["aa"] = self.heavy_df["difference"].apply(lambda x: x[-1])

        if len(self.light_df) > 0:
            self.light_df["difference"] = self.light_df["light_differences"].apply(
                lambda x: x[0]
            )
            self.light_df["site"] = self.light_df["difference"].apply(
                lambda x: int(x[1:-1])
            )
            self.light_df["aa"] = self.light_df["difference"].apply(lambda x: x[-1])

    def _create_dms_style_df(self, chain: str) -> pd.DataFrame:
        """Create DMS-style DataFrame with amino acids as index and sites as columns for
        specified chain."""
        chain_df = getattr(self, f"{chain}_df")
        consensus = getattr(self, f"{chain}_consensus")

        dms_style = chain_df.pivot(
            index="aa", columns="site", values=self.fitness_column
        )

        # Fill in the WTs with 1s
        for site in dms_style.columns:
            aa = consensus[site - 1]
            # Assert that the WT value is NaN before setting it to 1.0
            assert pd.isna(
                dms_style.loc[aa, site]
            ), f"Expected NaN for WT {aa} at site {site} in {chain} chain, got {dms_style.loc[aa, site]}"

            dms_style.loc[aa, site] = 1.0

        return dms_style

    def plot_fitness_histogram(self, chain: str, ax=None):
        """Plot histogram of log fitness values for specified chain."""
        if ax is None:
            _, ax = plt.subplots()

        chain_df = getattr(self, f"{chain}_df")

        if self.log_transform:
            plot_values = np.log(chain_df[self.fitness_column])
            xlabel = f"log({self.fitness_label})"
        else:
            plot_values = chain_df[self.fitness_column]
            xlabel = self.fitness_label

        ax.hist(plot_values, bins=100)
        ax.axvline(0.0, color="red", linestyle="--")
        ax.set_title(f"{chain.capitalize()} Chain")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        sns.despine()
        return ax

    def plot_fitness_histograms(self, figsize=(12, 5)):
        """Plot histograms for both heavy and light chains side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        self.plot_fitness_histogram(chain="heavy", ax=ax1)
        self.plot_fitness_histogram(chain="light", ax=ax2)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_heatmap(self, chain: str, ax=None):
        """Plot heatmap of log fitness values for specified chain."""
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 4))

        dms_style_df = getattr(self, f"{chain}_dms_style_df")

        ordered_df = dms_style_df.reindex(self.KOENIG_AA_ORDER)
        sns.heatmap(ordered_df, cmap="coolwarm", center=0, vmin=-8, vmax=5, ax=ax)
        cbar = ax.collections[0].colorbar
        label = (
            f"log({self.fitness_label})" if self.log_transform else self.fitness_label
        )
        cbar.set_label(label)
        ax.set_title(f"{chain.capitalize()} Chain")
        return ax

    def plot_heatmaps(self, figsize=(14, 10)):
        """Plot heatmaps for both heavy and light chains stacked vertically."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        self.plot_heatmap(chain="heavy", ax=ax1)
        self.plot_heatmap(chain="light", ax=ax2)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def _backtranslate_consensus(self, chain: str) -> str:
        """Backtranslate consensus sequence to nucleotides using human codon preferences
        and V gene information where available."""
        consensus = getattr(self, f"{chain}_consensus")
        # Use appropriate V gene sequence based on chain
        if chain == "heavy":
            v_gene_seq = IGHV3_23_04_SEQ
        elif chain == "light":
            v_gene_seq = IGKV1_39_01_SEQ
        else:
            raise ValueError(f"Invalid chain: {chain}")

        return backtranslate_with_v_gene(consensus, v_gene_seq=v_gene_seq)

    def _create_neighbor_indicator(
        self, consensus_backtrans: str, chain: str = "heavy"
    ) -> np.ndarray:
        """Create neighbor indicator matrix from backtranslated consensus.

        Args:
            consensus_backtrans: Backtranslated consensus sequence
            chain: "heavy" or "light" - determines trimming behavior

        Returns:
            Neighbor indicator matrix with appropriate columns removed
        """
        neighbor = make_codon_neighbor_indicator(consensus_backtrans)

        if chain == "heavy":
            # Heavy chain: remove first column (position 1 not mutated)
            return neighbor[:, 1:]
        elif chain == "light":
            # Light chain: remove first and last columns (positions 1 and 108 not mutated)
            return neighbor[:, 1:-1]
        else:
            raise ValueError(f"Invalid chain type: {chain}. Must be 'heavy' or 'light'")

    def plot_comparison(
        self,
        other_df: pd.DataFrame,
        x_label: str,
        y_label: str,
        chain: str,
        ax=None,
        neighbor_only: bool = None,
    ) -> Tuple[float, plt.Axes]:
        """Plot comparison between two log-transformed datasets for specified chain."""
        if ax is None:
            _, ax = plt.subplots()

        dms_style_df = getattr(self, f"{chain}_dms_style_df")
        neighbor_indicator = getattr(self, f"{chain}_neighbor_indicator")

        if neighbor_only is not None:
            mask = neighbor_indicator if neighbor_only else ~neighbor_indicator
            df_to_use = dms_style_df.copy()
            df_to_use.values[~mask] = np.nan

        return plot_variant_comparison(
            df_to_use, f"log({x_label})", other_df, f"log({y_label})", ax=ax
        )


def plot_variant_comparison(
    x_df: pd.DataFrame,
    x_name: str,
    y_df: pd.DataFrame,
    y_name: str,
    neighbor_info=None,
    ax=None,
    x_lims=None,
    y_lims=None,
) -> Tuple[float, plt.Axes]:
    """Plot comparison between two variant effect matrices."""
    if ax is None:
        _, ax = plt.subplots()

    if x_lims:
        ax.set_xlim(x_lims)
    if y_lims:
        ax.set_ylim(y_lims)  # Apply model-specific y-axis limits

    colors = {"Codon neighbor": "#1f78b4", "Non-neighbor": "#33a02c"}

    # Melt dataframes first
    x_melted = melt_dms_like_df(x_df, x_name)
    y_melted = melt_dms_like_df(y_df, y_name)

    # Merge the melted dataframes
    compare_df = pd.merge(x_melted, y_melted, on=["aa", "site"]).dropna()

    # Add neighbor status if provided
    if neighbor_info is not None:
        if isinstance(neighbor_info, (str, np.str_)):
            compare_df["neighbor_status"] = neighbor_info
        else:
            # For the "all" case where we have a matrix of statuses
            status_indices = pd.MultiIndex.from_product(
                [x_df.index, x_df.columns], names=["aa", "site"]
            )
            status_series = pd.Series(neighbor_info.ravel(), index=status_indices)
            compare_df["neighbor_status"] = compare_df.set_index(
                ["aa", "site"]
            ).index.map(status_series)

    # Plot with or without color based on neighbor status
    if neighbor_info is not None:
        sns.scatterplot(
            data=compare_df,
            x=x_name,
            y=y_name,
            hue="neighbor_status",
            palette=colors,
            ax=ax,
            alpha=0.15,
            linewidth=0,
            s=10,
            legend=False,
        )
    else:
        sns.scatterplot(
            data=compare_df, x=x_name, y=y_name, ax=ax, alpha=0.15, linewidth=0, s=10
        )

    corr = compare_df[x_name].corr(compare_df[y_name])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.text(
        0.05,
        0.1,
        f"correlation = {corr:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        # font size as big as the title
        fontsize=15,
        color="#6a0dad",
    )
    sns.despine()

    return corr, ax


def melt_dms_like_df(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert wide DMS-style DataFrame to long format."""
    melted_df = df.reset_index().melt(
        id_vars=[df.index.name or "index"], var_name="site", value_name=value_name
    )
    melted_df.rename(columns={"index": "aa"}, inplace=True)
    return melted_df


IGHV3_23_04_SEQ = """GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTTAGCAGCTATGCCATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAGCTATTAGTGGTAGTGGTGGTAGCACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAAAGA"""
IGKV1_39_01_SEQ = """GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGAGCATTAGCAGCTATTTAAATTGGTATCAGCAGAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATGCTGCATCCAGTTTGCAAAGTGGGGTCCCATCAAGGTTCAGTGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGAGTTACAGTACCCCTCC"""


# Human codon usage frequencies from:
# https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606&aa=1
# Values are per thousand
HUMAN_CODON_USAGE = {
    "TTT": 17.6,
    "TTC": 20.3,  # Phe
    "TTA": 7.7,
    "TTG": 12.9,  # Leu
    "CTT": 13.2,
    "CTC": 19.6,  # Leu
    "CTA": 7.2,
    "CTG": 39.6,  # Leu
    "ATT": 16.0,
    "ATC": 20.8,  # Ile
    "ATA": 7.5,  # Ile
    "ATG": 22.0,  # Met
    "GTT": 11.0,
    "GTC": 14.5,  # Val
    "GTA": 7.1,
    "GTG": 28.1,  # Val
    "TCT": 15.2,
    "TCC": 17.7,  # Ser
    "TCA": 12.2,
    "TCG": 4.4,  # Ser
    "CCT": 17.5,
    "CCC": 19.8,  # Pro
    "CCA": 16.9,
    "CCG": 6.9,  # Pro
    "ACT": 13.1,
    "ACC": 18.9,  # Thr
    "ACA": 15.1,
    "ACG": 6.1,  # Thr
    "GCT": 18.4,
    "GCC": 27.7,  # Ala
    "GCA": 15.8,
    "GCG": 7.4,  # Ala
    "TAT": 12.2,
    "TAC": 15.3,  # Tyr
    "TAA": 1.0,
    "TAG": 0.8,  # Stop
    "CAT": 10.9,
    "CAC": 15.1,  # His
    "CAA": 12.3,
    "CAG": 34.2,  # Gln
    "AAT": 17.0,
    "AAC": 19.1,  # Asn
    "AAA": 24.4,
    "AAG": 31.9,  # Lys
    "GAT": 21.8,
    "GAC": 25.1,  # Asp
    "GAA": 29.0,
    "GAG": 39.6,  # Glu
    "TGT": 10.6,
    "TGC": 12.6,  # Cys
    "TGA": 1.6,  # Stop
    "TGG": 13.2,  # Trp
    "CGT": 4.5,
    "CGC": 10.4,  # Arg
    "CGA": 6.2,
    "CGG": 11.4,  # Arg
    "AGT": 12.1,
    "AGC": 19.5,  # Ser
    "AGA": 12.2,
    "AGG": 12.0,  # Arg
    "GGT": 10.8,
    "GGC": 22.2,  # Gly
    "GGA": 16.5,
    "GGG": 16.5,  # Gly
}

# Get standard genetic code from Biopython
standard_table = CodonTable.standard_dna_table
forward_table = standard_table.forward_table

# Create a mapping of amino acids to their most common codon in humans
PREFERRED_CODONS = {}
for aa in set(forward_table.values()):
    possible_codons = [codon for codon, amino in forward_table.items() if amino == aa]
    PREFERRED_CODONS[aa] = max(possible_codons, key=lambda c: HUMAN_CODON_USAGE[c])


def backtranslate(aa_sequence: str) -> str:
    """Backtranslate an amino acid sequence to nucleotides using human codon
    preferences.

    Args:
        aa_sequence: String of amino acid single letter codes (upper case)

    Returns:
        String of nucleotides representing the backtranslated sequence

    Raises:
        KeyError: If an invalid amino acid code is encountered
    """
    return "".join(PREFERRED_CODONS[aa] for aa in aa_sequence.upper())


def backtranslate_with_v_gene(aa_sequence: str, v_gene_seq: str) -> str:
    """Backtranslate protein sequence using V gene sequence where possible."""
    # Truncate v_gene_seq to codon boundary and translate
    v_gene_seq = v_gene_seq[: len(v_gene_seq) - len(v_gene_seq) % 3]
    v_gene_aa = translate_sequence(v_gene_seq)
    consensus_popular_nt = backtranslate(aa_sequence)

    result = ""
    for i, aa in enumerate(aa_sequence):
        if i < len(v_gene_aa) and aa == v_gene_aa[i]:
            result += v_gene_seq[i * 3 : i * 3 + 3]
        else:
            result += consensus_popular_nt[i * 3 : i * 3 + 3]

    assert translate_sequence(result) == aa_sequence
    return result


def assign_wt(df: pd.DataFrame, wt_seq: str, value=np.nan):
    """Assign a value to wild-type positions in a dataframe."""
    for idx, aa in enumerate(wt_seq):
        if aa in df.index and idx + 1 in df.columns:
            df.loc[aa, idx + 1] = value


def df_of_arr(arr: np.ndarray) -> pd.DataFrame:
    """Convert numpy array to dataframe with AA rows and position columns."""
    return pd.DataFrame(
        arr, columns=range(1, arr.shape[1] + 1), index=list(AA_STR_SORTED)
    )


def trim_df(df: pd.DataFrame, chain: str = "heavy") -> pd.DataFrame:
    """Trim columns from a dataframe based on chain type.

    Args:
        df: DataFrame to trim
        chain: "heavy" or "light" - determines trimming behavior
            - heavy: trim only first column (position 1 not mutated)
            - light: trim first and last columns (positions 1 and 108 not mutated)

    Returns:
        Trimmed DataFrame
    """
    if chain == "heavy":
        # Heavy chain: only position 1 is not mutated
        return df.drop(df.columns[0], axis=1)
    elif chain == "light":
        # Light chain: positions 1 and 108 are not mutated
        return df.drop([df.columns[0], df.columns[-1]], axis=1)
    else:
        raise ValueError(f"Invalid chain type: {chain}. Must be 'heavy' or 'light'")


def process_model_output(
    model_output: np.ndarray,
    wt_seq: str,
    log_transform: bool = True,
    chain: str = "heavy",
) -> pd.DataFrame:
    """Process model output into a trimmed dataframe with WT positions set to NaN.

    Args:
        model_output: Model predictions as numpy array or tensor
        wt_seq: Wild-type sequence
        log_transform: Whether to apply log transformation
        chain: "heavy" or "light" - determines trimming behavior

    Returns:
        Processed dataframe ready for comparison
    """
    if isinstance(model_output, torch.Tensor):
        model_output = model_output.cpu().numpy()

    if log_transform:
        model_output = np.log(model_output)

    df = df_of_arr(model_output.T)
    assign_wt(df, wt_seq, np.nan)
    return trim_df(df, chain=chain)


def evaluate_paired_model(
    paired_model, heavy_seq: str, light_seq: str
) -> Dict[str, np.ndarray]:
    """Evaluate a paired antibody model on heavy and light sequences.

    Args:
        paired_model: A model that accepts paired sequences
        heavy_seq: Heavy chain sequence
        light_seq: Light chain sequence

    Returns:
        Dictionary with 'heavy' and 'light' predictions
    """
    paired_output = paired_model([[heavy_seq, light_seq]])
    return {"heavy": paired_output[0][0], "light": paired_output[0][1]}


def create_model_comparison_data(
    dataset: KoenigDataset,
    model_predictions: Dict[str, Dict[str, pd.DataFrame]],
    chain: str = "heavy",
) -> pd.DataFrame:
    """Create a comparison dataframe for multiple models on a specific chain.

    Args:
        dataset: KoenigDataset instance
        model_predictions: Dict of model_name -> {chain -> prediction_df}
        chain: Which chain to analyze ('heavy' or 'light')

    Returns:
        DataFrame with model predictions aligned for comparison
    """
    dms_df = getattr(dataset, f"{chain}_dms_style_df")

    # Stack all data for comparison
    comparison_data = []

    # Add experimental data
    comparison_data.append({"model": "Experimental", "data": dms_df, "chain": chain})

    # Add model predictions
    for model_name, predictions in model_predictions.items():
        if chain in predictions:
            comparison_data.append(
                {"model": model_name, "data": predictions[chain], "chain": chain}
            )

    return comparison_data
