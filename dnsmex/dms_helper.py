"""Common helper functions for DMS-style data analysis across different datasets."""

from collections import Counter
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def get_site_by_site_consensus(df: pd.DataFrame, chain: str) -> str:
    """Get site by site consensus sequence for specified chain.

    Args:
        df: DataFrame with sequences
        chain: Column name containing sequences (e.g., 'heavy', 'light')

    Returns:
        Consensus sequence string
    """
    sequences = df[chain]
    columns = zip(*sequences)
    return "".join(Counter(col).most_common(1)[0][0] for col in columns)


def protein_differences(seq1: str, seq2: str) -> List[str]:
    """Compare two protein sequences and express differences in 1-indexed format like
    A105G.

    Args:
        seq1: First sequence (usually reference/WT)
        seq2: Second sequence (usually variant)

    Returns:
        List of mutations in format like ['A105G', 'T107S']
    """
    assert len(seq1) == len(
        seq2
    ), f"Sequences must be same length: {len(seq1)} vs {len(seq2)}"
    return [
        f"{res1}{i}{res2}"
        for i, (res1, res2) in enumerate(zip(seq1, seq2), start=1)
        if res1 != res2
    ]


def plot_correlation_scatter(
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str,
    y_label: str,
    ax: Optional[plt.Axes] = None,
    color: str = "#1f78b4",
    alpha: float = 0.5,
    show_correlation: bool = True,
    correlation_color: str = "#6a0dad",
) -> Tuple[float, plt.Axes]:
    """Create a scatter plot with correlation coefficient.

    Args:
        x_data: X-axis data
        y_data: Y-axis data
        x_label: Label for x-axis
        y_label: Label for y-axis
        ax: Matplotlib axes (creates new if None)
        color: Point color
        alpha: Point transparency
        show_correlation: Whether to show correlation text
        correlation_color: Color for correlation text

    Returns:
        Tuple of (correlation coefficient, axes)
    """
    if ax is None:
        _, ax = plt.subplots()

    # Remove NaN values
    mask = ~(x_data.isna() | y_data.isna())
    x_clean = x_data[mask]
    y_clean = y_data[mask]

    # Create scatter plot
    ax.scatter(x_clean, y_clean, alpha=alpha, linewidth=0, color=color)

    # Calculate correlation
    corr = x_clean.corr(y_clean) if len(x_clean) > 0 else 0.0

    # Add correlation text
    if show_correlation:
        ax.text(
            0.05,
            0.95,
            f"Pearson r: {corr:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=plt.rcParams["axes.titlesize"],
            color=correlation_color,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    sns.despine()

    return corr, ax


def create_correlation_grid(
    experimental_data: pd.Series,
    model_predictions: Dict[str, pd.Series],
    experimental_label: str,
    figsize: Tuple[float, float] = (6, 9),
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Create a grid of correlation plots for multiple models.

    Args:
        experimental_data: Experimental values to compare against
        model_predictions: Dict of model_name -> predictions
        experimental_label: Label for experimental data
        figsize: Figure size

    Returns:
        Tuple of (figure, DataFrame with correlations)
    """
    n_models = len(model_predictions)
    fig, axs = plt.subplots(n_models, 1, figsize=figsize, sharex=True)
    if n_models == 1:
        axs = [axs]

    correlations = []

    for i, (model_name, predictions) in enumerate(model_predictions.items()):
        corr, _ = plot_correlation_scatter(
            experimental_data,
            predictions,
            experimental_label,
            model_name,
            ax=axs[i],
            show_correlation=True,
        )
        correlations.append({"Model": model_name, "Correlation": corr})

        # Only show x-label on bottom plot
        if i < n_models - 1:
            axs[i].set_xlabel("")

    plt.tight_layout()

    # Create correlation DataFrame
    corr_df = pd.DataFrame(correlations)

    return fig, corr_df


def melt_dms_like_df(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert wide DMS-style DataFrame to long format.

    Args:
        df: Wide format DataFrame with amino acids as index and sites as columns
        value_name: Name for the value column in melted DataFrame

    Returns:
        Long format DataFrame with columns: aa, site, value_name
    """
    melted_df = df.reset_index().melt(
        id_vars=[df.index.name or "index"], var_name="site", value_name=value_name
    )
    melted_df.rename(columns={"index": "aa"}, inplace=True)
    return melted_df


def mask_tensor_of_differences(max_len: int, differences: List[str]) -> torch.Tensor:
    """Make a tensor that is 1 for the positions of the differences and 0 otherwise.

    For example, A105G would have a 1 at column 104 in the G row.

    Args:
        max_len: Maximum sequence length
        differences: List of mutations in format like ['A105G', 'T107S']

    Returns:
        Tensor of shape (20, max_len) with 1s at mutation positions
    """
    from netam.sequences import AA_STR_SORTED

    mask = torch.zeros((20, max_len))
    for diff in differences:
        site = int(diff[1:-1])
        aa2 = diff[-1]
        mask[AA_STR_SORTED.index(aa2), site - 1] = 1
    return mask


def sel_score_of_differences(
    dasm_scores: torch.Tensor, differences: List[str]
) -> float:
    """Calculate selection score for given differences using DASM scores.

    Args:
        dasm_scores: DASM log scores tensor of shape (20, seq_len)
        differences: List of mutations in format like ['A105G', 'T107S']

    Returns:
        Sum of selection scores for all mutations
    """
    return (
        (dasm_scores * mask_tensor_of_differences(dasm_scores.shape[1], differences))
        .sum()
        .item()
    )
