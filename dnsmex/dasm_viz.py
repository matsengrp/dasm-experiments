import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import altair as alt
import seaborn as sns
import umap

from netam.framework import trimmed_shm_model_outputs_of_crepe
from netam.sequences import (
    nt_idx_tensor_of_str,
    aa_index_of_codon,
    NT_STR_SORTED,
    STOP_CODONS,
    AA_STR_SORTED,
    translate_sequence,
)
from netam.hit_class import parent_specific_hit_classes
import netam.molevol as molevol


PLOT_AA_ORDER = list("RKHDEQNSTYWFAILMVGPC")


def codon_prediction_df_from_sequence(
    nt_sequence: str, neutral_crepe, dasm_crepes, branch_length: float
):
    """Create a DataFrame with one row per codon per aa site.

    The Dataframe contains the following columns:
    * aa_site: 1-indexed site number
    * hit_class: The hit class of the codon
    * aa_idx: The amino acid index
    * aa: The amino acid
    * aa_site_prediction_variance: The variance of the prediction at that site, over all the passed `dasm_crepes`
    * aa_site_prediction: The mean dasm prediction at that site
    * neutral_prob: The neutral probability of that codon, according to the neutral crepe
    * is_wildtype: Whether the codon is the wildtype codon at its site

    Args:
        nt_sequence: The nucleotide sequence
        neutral_crepe: The neutral model
        dasm_crepes: A list of crepe models for each parent
        branch_length: The branch length used to compute neutral codon probabilities
    """
    aa_sequence = translate_sequence(nt_sequence)

    stacked_outputs = torch.stack(
        [
            crepe.model.selection_factors_of_aa_str(aa_sequence)
            for _, crepe in dasm_crepes
        ],
        dim=0,
    )

    selection_factors = torch.mean(stacked_outputs, dim=0)
    selection_factor_variance = torch.var(stacked_outputs, dim=0, unbiased=False)

    hit_classes = parent_specific_hit_classes(
        nt_idx_tensor_of_str(nt_sequence).view(-1, 3)
    )
    # Issue 153 should be addressed here.
    neutral_codon_probs = neutral_codon_probs_from_sequence(
        nt_sequence, neutral_crepe, branch_length=branch_length
    )
    target_aa_idxs = target_aa_tensor_of_len(len(aa_sequence))

    neutral_codon_rows = []
    for site_idx in range(len(aa_sequence)):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    aa_idx = int(target_aa_idxs[site_idx, i, j, k])
                    if aa_idx == -1:
                        continue
                    else:
                        aa = AA_STR_SORTED[aa_idx]
                    neutral_codon_rows.append(
                        (
                            site_idx + 1,
                            int(hit_classes[site_idx, i, j, k]),
                            aa_idx,
                            aa,
                            selection_factor_variance[site_idx, aa_idx],
                            selection_factors[site_idx, aa_idx],
                            neutral_codon_probs[site_idx, i, j, k],
                            aa == aa_sequence[site_idx],
                        )
                    )

    return pd.DataFrame(
        data=neutral_codon_rows,
        columns=(
            "aa_site",
            "hit_class",
            "aa_idx",
            "aa",
            "aa_site_prediction_variance",
            "aa_site_prediction",
            "neutral_prob",
            "is_wildtype",
        ),
    )


def selection_df_from_factors(
    selection_factors: np.ndarray,
) -> pd.DataFrame:
    """Create a dataframe with one column per site and rows indexed by AA identity.

    Args:
        selection_factors: An array containing selection factors.
            Rows (first index) should correspond to sites, and columns should
            correspond to amino acids in `netam.common.AA_STR_SORTED`
            (alphabetical) order.
    Returns:
        DataFrame with rows indexed by aa in order 'RKHDEQNSTYWFAILMVGPC'
            and one named column per 1-indexed site.
    """
    df = pd.DataFrame(
        selection_factors.T,
        index=list(AA_STR_SORTED),
        columns=[i + 1 for i in range(selection_factors.shape[0])],
    )

    df = df.reindex(PLOT_AA_ORDER)

    return df


def dms_style_heatmap(
    log_selection_factors_df: pd.DataFrame,
    aa_string: str = None,
    normalize: bool = False,
    dotsize_overlay: pd.DataFrame = None,
    ax=None,
    colorbar_label: str = "Log Selection Factor",
    cb_min_center_max: tuple = (-5, 0, 1),
    plot_aa_order=PLOT_AA_ORDER,
) -> matplotlib.figure.Figure:
    """Create a DMS-style heatmap with colored and sized circle overlays from a
    selection factors dataframe.

    Args:
        log_selection_factors_df: A DataFrame as produced by
            `selection_df_from_factors`, with rows indexed by aa and one named
            column per 1-indexed site.
        aa_string: Amino acid sequence for which selection factors
            are estimated. If provided, length should match number of columns
            in selection_factors_df.
        normalize: Whether to divide each site's selection factors by the
            wildtype selection factor.
        dotsize_overlay: A DataFrame with the same shape as selection_factors_df
            that determines the circle size overlay on the heatmap.
        ax: matplotlib axes on which to plot the heatmap. If not provided,
            new axes will be created and their containing figure returned.
        colorbar_label: Label for the colorbar.
        cb_min_center_max: Tuple of (min, center, max) values for the color scale.
        None means the color scale will be determined by the data.
        plot_aa_order: A list of amino acids in the order they should be plotted.
    Returns:
        Matplotlib figure containing the heatmap. If axes were provided, then
            this is the figure instance containing the passed axes. If not,
            this is a figure instance created by this plotting function.
    """
    pscale_factor = 0.7
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(
            figsize=(
                0.32 * len(log_selection_factors_df.columns) * pscale_factor,
                6 * pscale_factor,
            )
        )

    if set(log_selection_factors_df.index) != set(plot_aa_order):
        raise ValueError(
            "Expected selection_factors_df index to contain "
            "amino acid identities, but other values were found!"
        )

    plot_df = log_selection_factors_df.copy().reindex(plot_aa_order)

    if normalize:
        if aa_string is None:
            raise ValueError("Normalizing by wildtype requires aa_string be provided.")
        wt_values = pd.Series(
            [plot_df.loc[aa, col] for col, aa in zip(plot_df.columns, aa_string)],
            index=plot_df.columns,
        )
        plot_df = plot_df.subtract(wt_values, axis=1)

    cmap_lin = LinearSegmentedColormap.from_list(
        name="dms_cmap_lin",
        colors=["red", "white", "#C8C6FF"],
    )
    cmap_lin.set_bad(color="lightgray")  # Set NaN values to gray
    if cb_min_center_max is None:
        data_min = np.nanmin(plot_df.values)
        data_max = np.nanmax(plot_df.values)
        cb_min_center_max = (
            data_min,
            (data_min + data_max) / 2,
            data_max,
        )
    (vmin, vcenter, vmax) = cb_min_center_max
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap_lin)

    if dotsize_overlay is None:
        sns.heatmap(plot_df, cmap=sm.cmap, norm=norm, cbar=False, ax=ax, annot=False)
        ax.invert_yaxis()  # Invert y-axis to have AA order from top to bottom
    else:
        # Plot the heatmap squares with color and size based on data
        max_dotsize = np.max(dotsize_overlay)
        full_dotsize = 130 * pscale_factor
        for (i, j), value in np.ndenumerate(plot_df.values):
            color = sm.to_rgba(value)
            size = (dotsize_overlay.iloc[i, j] / max_dotsize) * full_dotsize
            ax.scatter(j + 0.5, i + 0.5, s=size, marker="s", color=color, alpha=1)

    # Add color bar for reference
    cbar = fig.colorbar(
        sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01, shrink=0.9
    )
    cbar.outline.set_visible(False)

    # There's rather a lot of code here for the colorbar, but the point of all
    # of it is that we want to have nice ticks that are set independently in the
    # positive and negative ranges.

    # Function to generate nice ticks within a range
    def get_nice_ticks(start, end, target_count=3):
        # Base intervals we consider "nice"
        intervals = [0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10]

        range_width = abs(end - start)

        # Find an interval that gives approximately the target number of ticks
        ideal_interval = range_width / target_count

        # Find the closest "nice" interval
        interval = min(intervals, key=lambda x: abs(x - ideal_interval))

        # Generate ticks
        first_tick = math.ceil(start / interval) * interval
        last_tick = math.floor(end / interval) * interval

        ticks = []
        current = first_tick
        while (
            current <= last_tick + 1e-10
        ):  # Add small epsilon for floating point comparison
            ticks.append(current)
            current += interval

        return ticks

    # Generate separate nice ticks for negative and positive ranges
    neg_ticks = get_nice_ticks(vmin, vcenter) if vmin < vcenter else []
    pos_ticks = get_nice_ticks(vcenter, vmax) if vcenter < vmax else []

    # Always include the center value
    all_ticks = sorted(list(set(neg_ticks + [vcenter] + pos_ticks)))

    # Set the ticks
    cbar.set_ticks(all_ticks)

    # Format the tick labels - use integers where possible
    cbar.set_ticklabels(
        [str(int(tick)) if tick == int(tick) else f"{tick:.2g}" for tick in all_ticks]
    )
    cbar.set_label(colorbar_label)
    cbar.ax.yaxis.label.set_size(14)

    # Remove axis lines but keep tick marks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_xlabel("Site")
    ax.set_ylabel("AA")
    ax.set_xticks(np.arange(len(plot_df.columns)) + 0.5)
    ax.set_xticklabels(plot_df.columns, rotation=90)
    ax.set_yticks(np.arange(len(plot_df.index)) + 0.5)
    ax.set_yticklabels(plot_df.index)
    ax.set_xlim(-0.4, len(plot_df.columns) + 1)

    # Add vertical lines at every half-integer on the x-axis
    for x_pos in np.arange(0, len(plot_df.columns), 1):
        ax.axvline(
            x=x_pos, color="white", linestyle="-", linewidth=1.5, zorder=2, alpha=1
        )

    # Add horizontal lines at every half-integer on the y-axis
    for y_pos in np.arange(0, len(plot_df.index), 1):
        ax.axhline(
            y=y_pos, color="white", linestyle="-", linewidth=1.5, zorder=2, alpha=1
        )

    plot_aa_order_indices = {aa: idx for idx, aa in enumerate(plot_aa_order)}

    if aa_string is not None:
        for col, aa in enumerate(aa_string):
            row = plot_aa_order_indices[aa]
            ax.text(col + 0.5, row + 0.5, "*", ha="center", va="center", color="black")

    ax.invert_yaxis()  # Invert y-axis to have AA order from top to bottom
    ax.set_aspect("equal")
    return fig


def neutral_codon_probs_from_sequence(
    nt_str: str,
    neutral_crepe,
    branch_length: float,
):
    """Construct an array to adjust heatmap square sizes based on neutral codon
    probabilities.

    Args:
        nt_str: The widltype nucleotide sequence
        neutral_crepe: Crepe object containing the neutral model
        branch_length: The branch length used to compute neutral codon probabilities

    Returns:
        Numpy array containing neutral aa probabilities, with alphabetical aa order
    """
    (nt_rates,), (nt_csps,) = trimmed_shm_model_outputs_of_crepe(
        neutral_crepe, [nt_str]
    )
    parent_idxs = nt_idx_tensor_of_str(nt_str)
    parent_len = len(nt_str)

    mut_probs = 1.0 - torch.exp(-branch_length * nt_rates[:parent_len])
    nt_csps = nt_csps[:parent_len, :]
    molevol.check_csps(parent_idxs, nt_csps)

    mut_matrices = molevol.build_mutation_matrices(
        parent_idxs.reshape(-1, 3), mut_probs.reshape(-1, 3), nt_csps.reshape(-1, 3, 4)
    )
    return molevol.codon_probs_of_mutation_matrices(mut_matrices)


def aa_from_nt_idx_triple(idx_codon):
    codon = "".join([NT_STR_SORTED[i] for i in idx_codon])
    if codon in STOP_CODONS:
        return -1
    return aa_index_of_codon(codon)


def target_aa_tensor_of_len(seq_len):
    one_site_target_aa_tensor = torch.zeros(4, 4, 4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                one_site_target_aa_tensor[i, j, k] = aa_from_nt_idx_triple([i, j, k])
    return one_site_target_aa_tensor.repeat(seq_len, 1, 1, 1)


def neutral_aa_probs_from_sequence(
    nt_str: str,
    neutral_crepe,
    branch_length: float,
):
    """Construct a DataFrame to adjust heatmap square sizes based on neutral codon
    probabilities.

    Args:
        nt_str: The widltype nucleotide sequence
        neutral_crepe: Crepe object containing the neutral model
        branch_length: The branch length used to compute neutral codon probabilities

    Returns:
        Numpy array containing neutral aa probabilities, with alphabetical aa order
    """
    return (
        neutral_codon_probs_from_sequence(nt_str, neutral_crepe, branch_length).view(
            -1, 64
        )
        @ molevol.CODON_AA_INDICATOR_MATRIX
    )


def dotsize_df_from_sequence(
    nt_str: str,
    neutral_crepe,
    branch_length: float,
):
    """Construct a DataFrame to adjust heatmap square sizes based on neutral codon
    probabilities.

    Args:
        nt_str: The widltype nucleotide sequence
        neutral_crepe: Crepe object containing the neutral model
        branch_length: The branch length used to compute neutral codon probabilities

    Returns:
        DataFrame with rows indexed by aa and columns labeled by aa site.
    """
    neutral_aa_probs = neutral_aa_probs_from_sequence(
        nt_str, neutral_crepe, branch_length
    )
    # Transformation makes small dots easier to see, and makes dot area
    # proportional to probability
    shifted_neutral_aa_probs = neutral_aa_probs.sqrt()
    # Clip to avoid high-probability wildtype codons dominating sizes:
    sorted_arr = np.sort(shifted_neutral_aa_probs, axis=1)[:, ::-1]
    max_second_largest_value = sorted_arr[:, 1].max()
    neutral_aa_probs_clipped = np.minimum(
        shifted_neutral_aa_probs, max_second_largest_value
    )
    return selection_df_from_factors(neutral_aa_probs_clipped)


def umap_plot_df_of_dasm_burrito(
    burrito,
    global_mask=False,
    original_pcp_df=None,
    data_columns=[
        "sample_id",
        "family",
        "v_family",
        "v_gene",
        "depth",
        "distance",
        "child_is_leaf",
    ],
):
    """Compute UMAP embeddings of DASM representations and return a DataFrame containing
    those embeddings.

    Args:
        burrito: A trained DASM burrito object.
        global_mask: Whether to mask UMAP embeddings by removing any site which is masked in any parent sequence, before
            computing UMAP embedding.
        original_pcp_df: A DataFrame containing the original PCP data, from which columns in `data_columns` can be recovered.
        data_columns: A list of column names to include from the original pcp df data.

    Returns:
        DataFrame containing UMAP embeddings in columns named `UMAP_1` and `UMAP_2`, and any columns
        recovered from original PCP data.
    """
    burrito.model.eval()
    val_loader = burrito.build_val_loader()
    umap_plot_df = pd.DataFrame(
        columns=["aa_parents_idxs", "aa_parent_mask", "aa_parent_encoded"]
    )
    for batch in tqdm(val_loader, desc="Evaluating model"):
        representations = (
            burrito.model.represent(batch["aa_parents_idxs"], batch["mask"])
            .detach()
            .cpu()
            .numpy()
        )
        batch_df = pd.DataFrame(
            {
                "aa_parents_idxs": list(batch["aa_parents_idxs"].numpy()),
                "aa_parent_mask": list(batch["mask"].numpy()),
                "aa_parent_encoded": list(representations),
            }
        )
        if not batch_df.empty:
            umap_plot_df = pd.concat([umap_plot_df, batch_df], ignore_index=True)
        del representations, batch_df
    if global_mask:
        global_mask = np.logical_and.reduce(list(umap_plot_df["aa_parent_mask"].values))

        umap_plot_df["aa_parent_encoded_nomask"] = umap_plot_df[
            "aa_parent_encoded"
        ].copy()

        umap_plot_df["aa_parent_encoded"] = umap_plot_df.apply(
            lambda row: row["aa_parent_encoded"][global_mask], axis=1
        )
    flattened_data = umap_plot_df["aa_parent_encoded"].apply(lambda x: x.flatten())
    data_for_umap = np.stack(
        flattened_data.values
    )  # Resulting shape will be (df_length, sites * representation_dims)

    umap_model = umap.UMAP(n_components=2)
    print("Computing UMAP transformation...")
    embeddings = umap_model.fit_transform(data_for_umap)

    umap_plot_df["UMAP_1"] = embeddings[:, 0]
    umap_plot_df["UMAP_2"] = embeddings[:, 1]

    if original_pcp_df is not None:
        colset = set(original_pcp_df.columns)
        columns_to_pick = []
        for col in data_columns:
            if col not in colset:
                raise ValueError(
                    f"Data column {col} not found in original PCP dataframe"
                )
            else:
                columns_to_pick.append(col)
        # return pd.concat([umap_plot_df.reset_index(drop=True), original_pcp_df[data_columns].reset_index(drop=True)], axis=1)
        return pd.concat([umap_plot_df, original_pcp_df[data_columns]], axis=1)
    return umap_plot_df


def plot_umap_with_categories(plot_df, categories, ax=None):
    """Plot UMAP embeddings with color-coded categories.

    Args:
        plot_df: DataFrame containing UMAP embeddings in columns `UMAP_1` and `UMAP_2`, and any columns listed in `categories`.
        categories: A list of column names in `plot_df` to use for color-coding.
        ax: Matplotlib axes on which to plot the UMAP. If not provided, a new figure will be created.
    Returns:
        Matplotlib figure containing the plot.
    """
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, axs = plt.subplots(1, len(categories), figsize=(10 * len(categories), 8))
    if len(categories) == 1:
        axs = [axs]
    for category_column, ax in zip(categories, axs):
        scatter = ax.scatter(
            plot_df["UMAP_1"],
            plot_df["UMAP_2"],
            c=plot_df[category_column]
            .astype("category")
            .cat.codes,  # Color by categorical variable
            cmap="Spectral",
            s=50,
            alpha=0.7,
        )

        handles, _labels = scatter.legend_elements(prop="colors", alpha=0.6)
        category_labels = plot_df[category_column].astype("category").cat.categories
        ax.legend(handles, category_labels, title=category_column, loc="upper right")

        ax.set_title(f"DASM representation by {category_column}")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
    return fig


def plot_umap_with_categories_and_histogram_selector(
    plot_df, categories, tooltip_columns, selection_column="depth"
):
    """Plot UMAP embeddings with color-coded categories, tooltips, and a histogram data
    selector.

    Args:
        plot_df: DataFrame containing UMAP embeddings in columns `UMAP_1` and `UMAP_2`, and any columns listed in
            `categories` and `selection_column`.
        categories: A list of column names in `plot_df` to use for color-coding.
        tooltip_columns: A list of column names in `plot_df` to include in tooltips.
        selection_column: The column to plot on the histogram data selector.

    Returns:
        Altair chart containing the plot.
    """
    plots = []
    required_columns = set(
        tooltip_columns + ["UMAP_1", "UMAP_2", selection_column] + categories
    )
    plot_df = plot_df[list(sorted(required_columns))].copy()

    x_min, x_max = plot_df["UMAP_1"].min(), plot_df["UMAP_1"].max()
    y_min, y_max = plot_df["UMAP_2"].min(), plot_df["UMAP_2"].max()
    # Define a selection interval for the histogram
    depth_selection = alt.selection_interval(encodings=["x"])

    # Create the histogram with selection_column on the x-axis
    histogram = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{selection_column}:Q", bin=True, title=selection_column),
            y=alt.Y("count()", title="Count", scale=alt.Scale(type="log")),
        )
        .add_selection(depth_selection)
        .properties(
            width=800,
            height=200,
            title="Drag on this plot to select by " + selection_column,
        )
    )

    for category_column in categories:
        plot_df[category_column] = plot_df[category_column].astype(str)

        # Generate a fixed color scale for the unique values in the category column
        category_values = sorted(plot_df[category_column].unique().tolist())

        scatter_plot = (
            alt.Chart(plot_df)
            .mark_circle(size=50, opacity=0.7)
            .encode(
                x=alt.X(
                    "UMAP_1",
                    title="UMAP Dimension 1",
                    scale=alt.Scale(domain=[x_min, x_max]),
                ),
                y=alt.Y(
                    "UMAP_2",
                    title="UMAP Dimension 2",
                    scale=alt.Scale(domain=[y_min, y_max]),
                ),
                color=alt.Color(
                    f"{category_column}:N",
                    scale=alt.Scale(domain=category_values, scheme="spectral"),
                ),
                tooltip=[alt.Tooltip(col) for col in tooltip_columns],
            )
            .transform_filter(
                depth_selection  # Filter scatter plot based on histogram selection
            )
            .properties(
                title=f"DASM representation by {category_column}", width=400, height=400
            )
        )

        plots.append(scatter_plot)

    # Concatenate the scatter plots horizontally
    combined_scatter = alt.hconcat(*plots).resolve_scale(color="independent")

    # Combine histogram and scatter plots vertically
    combined_plot = alt.vconcat(histogram, combined_scatter)

    return combined_plot
