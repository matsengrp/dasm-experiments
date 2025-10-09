import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
import tqdm

from netam.common import heavy_chain_shim
from netam.framework import load_crepe
from netam.sequences import AA_STR_SORTED
from dnsmex.local import localify
from netam.sequences import AA_STR_SORTED

"""
Monte Carlo Dropout Analysis for DASM

This module implements Monte Carlo dropout techniques for uncertainty quantification
for the DASM model. It provides tools for:

1. Running MC dropout inference on protein sequences
2. Calculating uncertainty statistics on dropout distributions
3. Comparing model predictions with ground truth
4. Finding optimal dropout rates within a model so that 95% of the predictions
   fall within the 95% confidence intervals.
5. Visualizing prediction comparisons

Functions:
-------------
- run_mc_dropout_on_heavy_seq: Perform MC dropout inference on a single sequence
- calc_mc_dropout_stats_per_seq: Calculate uncertainty statistics for a single sequence
- compare_models_w_mc_dropout: Compare model predictions with ground truth (using the above two functions)
- find_optimal_dropout_rate: Calibrate dropout probability
- plot_compare_models_w_mc_dropout_prediction_scatter: Create scatter plots
- plot_compare_models_w_mc_dropout_prediction_seaborn_heatmap: Create heatmap visualizations

Notes:
------
The module assumes input sequences are protein heavy chains and works with
log-transformed selection factors. All visualization functions support both
standalone plotting and integration with existing matplotlib figures.
"""


def assign_wt(df, wt_seq, value):
    for idx, aa in enumerate(wt_seq):
        df.loc[aa, idx + 1] = value


def df_of_arr(arr):
    return pd.DataFrame(
        arr, columns=range(1, arr.shape[1] + 1), index=list(AA_STR_SORTED)
    )


def run_mc_dropout_on_heavy_seq(
    heavy_seq, model_path, mc_dropout_iterations=100, device="cpu", dropout_prob=0.1
):
    """Run Monte Carlo dropout inference on a heavy chain sequence to estimate
    prediction uncertainty.

    This function loads a model, enables dropout during inference by setting the model to train mode,
    and performs multiple forward passes with random dropout patterns. Each forward pass generates
    slightly different predictions, which collectively provide a distribution to estimate uncertainty.

    Parameters:
    -----------
    heavy_seq : str
        The heavy chain amino acid sequence to analyze
    model_path : str or model object
        Path to the saved model or model object to use for predictions
    mc_dropout_iterations : int, optional (default=100)
        Number of forward passes with dropout to perform
    device : str, optional (default='cpu')
        Device to run inference on ('cpu' or 'cuda')
    dropout_prob : float, optional (default=0.1)
        Probability of dropout to apply during inference

    Returns:
    --------
    all_data : pandas DataFrame
        Combined DataFrame containing predictions from all MC dropout iterations with columns:
        - AminoAcid: The amino acid at this position
        - Position: The sequence position
        - log_selection_factor: Log-transformed selection factor prediction
        - Sample: Identifier for which MC dropout iteration this prediction came from

    Notes:
    ------
    This function intentionally runs the model in train mode (not eval mode) to enable
    dropout during inference, which is necessary for Monte Carlo dropout uncertainty estimation.
    """
    crepe = load_crepe(localify(model_path), device=device)

    # Change dropout probability in all dropout layers
    for module in crepe.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_prob

    # put model in train mode for mc dropout
    crepe.model.train()
    crepe = heavy_chain_shim(crepe)

    # Initialize a list to store all predictions
    dasm_heavy_predictions = []

    # Run the model multiple times and save predictions
    for i in range(mc_dropout_iterations):
        # Get raw predictions
        [dasm_heavy] = crepe([heavy_seq])

        # Save log-transformed predictions
        log_dasm_heavy = np.log(dasm_heavy).T
        log_dasm_heavy_df = df_of_arr(log_dasm_heavy)
        assign_wt(log_dasm_heavy_df, heavy_seq, np.nan)
        log_dasm_heavy_df_long = log_dasm_heavy_df.stack().reset_index()
        log_dasm_heavy_df_long.columns = [
            "AminoAcid",
            "Position",
            "log_selection_factor",
        ]

        # Add sample identifier
        log_dasm_heavy_df_long["Sample"] = i

        dasm_heavy_predictions.append(log_dasm_heavy_df_long)

    # Combine all samples
    all_data = pd.concat(dasm_heavy_predictions, ignore_index=True)
    return all_data


def calc_mc_dropout_stats_per_seq(all_data, quantiles=[0.025, 0.975]):
    """Calculate statistics from Monte Carlo dropout predictions for uncertainty
    quantification.

    This function processes the raw MC dropout prediction data to compute various statistics
    that characterize the prediction uncertainty per grouped position and amino acid, including mean,
    standard deviation, and confidence intervals based on specified quantiles.

    Parameters:
    -----------
    all_data : pandas DataFrame
        Combined DataFrame containing predictions from all MC dropout iterations produced by run_mc_dropout_on_heavy_seq, with columns:
        - Position: The sequence position
        - AminoAcid: The amino acid at this position
        - log_selection_factor: Log-transformed selection factor prediction
        - Sample: Identifier for which MC dropout iteration this prediction came from
    quantiles : list of float, optional (default=[0.025, 0.975])
        Quantiles to use for confidence interval bounds, default gives 95% CI

    Returns:
    --------
    stats_df : pandas DataFrame
        DataFrame containing uncertainty statistics with columns:
        - Position: The sequence position
        - AminoAcid: The amino acid at this position
        - mean_log_selection_factor: Mean of log selection factor across MC samples
        - std_log_selection_factor: Standard deviation across MC samples
        - lower_bound_log_selection_factor: Lower bound of confidence interval
        - upper_bound_log_selection_factor: Upper bound of confidence interval
        - ci_width_log_selection_factor: Width of the confidence interval
        - normalized_ci_log_selection_factor: CI width normalized by absolute mean
        - cv_log_selection_factor: Coefficient of variation (std/|mean|)

    Notes:
    ------
    A small epsilon (1e-5) is added to denominators to prevent division by zero
    when calculating normalized metrics.
    """
    # Group by Position and AminoAcid to calculate across samples
    grouped = all_data.groupby(["Position", "AminoAcid"])

    # Calculate mean and std
    stats_df = (
        grouped["log_selection_factor"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_log_selection_factor",
                "std": "std_log_selection_factor",
            }
        )
    )

    # Add small epsilon to avoid division by zero
    epsilon = 1e-5

    # Calculate percentiles
    percentiles = grouped["log_selection_factor"].quantile(quantiles).unstack()
    percentiles.columns = ["lower_bound", "upper_bound"]
    percentiles = percentiles.reset_index().rename(
        columns={
            "lower_bound": "lower_bound_log_selection_factor",
            "upper_bound": "upper_bound_log_selection_factor",
        }
    )

    # Merge with stats_df
    stats_df = pd.merge(stats_df, percentiles, on=["Position", "AminoAcid"])

    # Step 4: Calculate all uncertainty measures
    stats_df["ci_width_log_selection_factor"] = (
        stats_df["upper_bound_log_selection_factor"]
        - stats_df["lower_bound_log_selection_factor"]
    )
    stats_df["normalized_ci_log_selection_factor"] = stats_df[
        "ci_width_log_selection_factor"
    ] / (np.abs(stats_df["mean_log_selection_factor"]) + epsilon)
    stats_df["cv_log_selection_factor"] = stats_df["std_log_selection_factor"] / (
        np.abs(stats_df["mean_log_selection_factor"]) + epsilon
    )

    return stats_df


def compare_models_w_mc_dropout(
    ground_truth_model,
    tested_model,
    test_sequence_list,
    mc_dropout_iterations=100,
    dropout_prob=0.1,
    device="cpu",
):
    """Compare the MC dropout predictions of a tested model to the predictions of a
    ground truth model.

    This function evaluates model uncertainty by performing Monte Carlo dropout on the tested model
    and comparing its predictions with a reference ground truth model. For each sequence in the test list,
    it calculates statistics on prediction distributions and determines if ground truth values fall within
    the prediction confidence intervals.

    Parameters:
    -----------
    ground_truth_model : str or model object
        Path to or instance of the model to use as ground truth reference
    tested_model : str or model object
        Path to or instance of the model to evaluate with MC dropout
    test_sequence_list : list
        List of sequences to use for the comparison
    mc_dropout_iterations : int, optional (default=100)
        Number of forward passes with dropout for uncertainty estimation
    dropout_prob : float, optional (default=0.1)
        Dropout probability to use during inference
    device : str, optional (default='cpu')
        Device to run the model on ('cpu' or 'cuda')

    Returns:
    --------
    df : pandas DataFrame
        DataFrame containing statistics for each position and amino acid, with columns including:
        - AminoAcid: The amino acid at this position
        - Position: The sequence position
        - mean_log_selection_factor: Mean prediction from MC dropout runs
        - std_log_selection_factor: Standard deviation of predictions
        - lower_bound_log_selection_factor: Lower bound of 95% confidence interval
        - upper_bound_log_selection_factor: Upper bound of 95% confidence interval
        - DASM_log_selection_factor: Ground truth prediction value
        - ground_truth_in_0.95_range: Boolean indicating if ground truth falls in CI
        - test_seq: The sequence being evaluated
    """
    # model with no dropout to use for ground truth comparison
    model = localify(ground_truth_model)
    crepe = load_crepe(model)
    crepe_heavy = heavy_chain_shim(crepe)

    stats_big_df = []
    for seq in tqdm.tqdm(test_sequence_list):
        # ground truth
        [dasm_heavy_original] = crepe_heavy([seq])
        log_dasm_heavy_original = np.log(dasm_heavy_original).T
        log_dasm_heavy_df = df_of_arr(log_dasm_heavy_original)
        assign_wt(log_dasm_heavy_df, seq, np.nan)
        # mc droupout on tested model
        all_data = run_mc_dropout_on_heavy_seq(
            seq,
            model_path=tested_model,
            mc_dropout_iterations=mc_dropout_iterations,
            device=device,
            dropout_prob=dropout_prob,
        )
        stats_df = calc_mc_dropout_stats_per_seq(all_data)
        # merge with ground truth model
        log_dasm_heavy_df_long = log_dasm_heavy_df.stack().reset_index()
        log_dasm_heavy_df_long.columns = [
            "AminoAcid",
            "Position",
            "DASM_log_selection_factor",
        ]
        stats_predict_df = pd.merge(
            stats_df, log_dasm_heavy_df_long, on=["AminoAcid", "Position"], how="outer"
        )
        stats_predict_df["ground_truth_in_0.95_range"] = (
            stats_predict_df["lower_bound_log_selection_factor"]
            <= stats_predict_df["DASM_log_selection_factor"]
        ) & (
            stats_predict_df["upper_bound_log_selection_factor"]
            >= stats_predict_df["DASM_log_selection_factor"]
        )
        stats_predict_df["test_seq"] = seq
        stats_big_df.append(stats_predict_df)

    df = pd.concat(stats_big_df)
    return df


def find_optimal_dropout_rate(
    model_tested,
    test_sequence_list,
    mc_dropout_iterations=100,
    dropout_prob_start=0.1,
    device="cpu",
):
    """Find the optimal dropout probability for Monte Carlo dropout uncertainty
    estimation. Optimal dropout probability is defined as the one where approximately
    95% of the predicted values fall within the 95% confidence intervals generated by
    Monte Carlo dropout for the same model. It uses a simple adaptive search algorithm
    to find this optimal dropout rate.

    Parameters:
    -----------
    model_tested : str or model object
        Path to or instance of the model to calibrate with MC dropout
    test_sequence_list : list
        List of sequences to use for calibration
    mc_dropout_iterations : int, optional (default=100)
        Number of forward passes with dropout for uncertainty estimation
    dropout_prob_start : float, optional (default=0.1)
        Initial dropout probability to try
    device : str, optional (default='cpu')
        Device to run the model on ('cpu' or 'cuda')

    Returns:
    --------
    found_dropout_prob : float
        The calibrated dropout probability that produces well-calibrated
        uncertainty estimates (where ~95% of ground truth values fall within
        the 95% confidence intervals)

    Notes:
    ------
    The function stops when the proportion of ground truth values within the
    95% confidence interval is within 0.01 of the target value (0.95).
    """

    # model with no dropout to use for ground truth comparison
    model = localify(model_tested)
    crepe = load_crepe(model)
    crepe_heavy = heavy_chain_shim(crepe)

    found_dropout_prob = None
    cur_dropout_prob = dropout_prob_start
    while found_dropout_prob is None:
        stats_big_df = []
        for seq in tqdm.tqdm(test_sequence_list):
            # ground truth
            [dasm_heavy_original] = crepe_heavy([seq])
            log_dasm_heavy_original = np.log(dasm_heavy_original).T
            log_dasm_heavy_df = df_of_arr(log_dasm_heavy_original)
            assign_wt(log_dasm_heavy_df, seq, np.nan)
            # mc droupout
            all_data = run_mc_dropout_on_heavy_seq(
                seq,
                model_path=model_tested,
                mc_dropout_iterations=mc_dropout_iterations,
                device=device,
                dropout_prob=cur_dropout_prob,
            )
            stats_df = calc_mc_dropout_stats_per_seq(all_data)
            # merge with ground truth
            log_dasm_heavy_df_long = log_dasm_heavy_df.stack().reset_index()
            log_dasm_heavy_df_long.columns = [
                "AminoAcid",
                "Position",
                "DASM_log_selection_factor",
            ]
            stats_predict_df = pd.merge(
                stats_df,
                log_dasm_heavy_df_long,
                on=["AminoAcid", "Position"],
                how="outer",
            )
            stats_predict_df["ground_truth_in_0.95_range"] = (
                stats_predict_df["lower_bound_log_selection_factor"]
                <= stats_predict_df["DASM_log_selection_factor"]
            ) & (
                stats_predict_df["upper_bound_log_selection_factor"]
                >= stats_predict_df["DASM_log_selection_factor"]
            )
            stats_predict_df["test_seq"] = seq
            stats_big_df.append(stats_predict_df)

        df = pd.concat(stats_big_df)
        # Calculate the proportion of true values in the 0.95 range. it should be around 0.95
        # If not, adjust the dropout probability
        true_proportion = df["ground_truth_in_0.95_range"].value_counts(normalize=True)[
            True
        ]
        loss = true_proportion - 0.95
        print(
            f"Dropout prob: {cur_dropout_prob}, True proportion: {true_proportion}, Loss: {loss}"
        )
        if abs(loss) < 0.01:
            found_dropout_prob = round(cur_dropout_prob, 2)
        elif loss < -0.01:
            cur_dropout_prob -= 0.05
        else:
            cur_dropout_prob += 0.05
    print(f"Found dropout prob: {found_dropout_prob}")
    return found_dropout_prob


def plot_compare_models_w_mc_dropout_prediction_scatter(
    df,
    ax=None,
    title="Ground Truth vs Model trained on simulations",
    x_label="Ground Truth Model \n(log_selection_factor)",
    y_label="Model trained on simulations \n(mean_log_selection_factor)",
):
    """Plot a scatter plot comparing ground truth values vs predictions with MC dropout.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing model comparison data with columns 'DASM_log_selection_factor',
        'mean_log_selection_factor', and 'ground_truth_in_0.95_range'
    ax : matplotlib Axes object, optional
        Axes to plot on, if None, creates a new figure
    title : str, optional
        Title for the plot
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plot
    """
    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1)

    proportion_predctions_in_ci = df["ground_truth_in_0.95_range"].value_counts(
        normalize=True
    )[True]

    # Get the x and y data for regression
    x = df["DASM_log_selection_factor"]  # Ground truth model (x-axis)
    y = df["mean_log_selection_factor"]  # deviation from model (y-axis)

    # Create the scatter plot of deviation vs ground truth
    ax.scatter(x, y, alpha=0.1)

    # Add a diagonal line for x=y (perfect agreement)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "y--", label="Perfect agreement (x=y)"
    )

    # Calculate and plot a trend line for the deviation
    slope_dev, intercept_dev, r_dev, p_dev, std_err_dev = stats.linregress(x, y)
    trend_x = np.array([x.min(), x.max()])
    trend_y = slope_dev * trend_x + intercept_dev
    ax.plot(
        trend_x,
        trend_y,
        "r-",
        label=f"Trend (slope={slope_dev:.3f}, intercept={intercept_dev:.3f})",
    )

    # Add labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, size=10)
    ax.set_title(
        title
        + "\nProportion of predictions in 95% CI: {:.2f}".format(
            proportion_predctions_in_ci
        )
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_compare_models_w_mc_dropout_prediction_seaborn_heatmap(
    df,
    ax=None,
    title="Ground Truth vs Model trained on simulations",
    x_label="Ground Truth Model \n(log_selection_factor)",
    y_label="Model trained on simulations \n(mean_log_selection_factor)",
):
    """Plot a heatmap of ground truth values vs prediction using seaborn with fixed
    bins.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing model comparison data
    ax : matplotlib Axes object, optional
        Axes to plot on, if None, creates a new figure

    Returns:
    --------
    fig : matplotlib Figure
        The figure containing the plot
    """

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    proportion_predictions_in_ci = df["ground_truth_in_0.95_range"].value_counts(
        normalize=True
    )[True]

    # Get the x and y data
    x = df["DASM_log_selection_factor"]  # Ground truth model
    y = df["mean_log_selection_factor"]  # simulated model predictions

    # Create fixed bins for x and y axes
    x_edges = np.arange(-8, 3, 0.2)  # From -8 to 3 with 0.2 step
    y_edges = np.arange(-8, 3, 0.2)  # Same for predictions

    x_bins = len(x_edges) - 1
    y_bins = len(y_edges) - 1

    # Assign each point to a bin
    x_bins_assigned = np.digitize(x, x_edges) - 1
    y_bins_assigned = np.digitize(y, y_edges) - 1

    # Clip to ensure valid indices
    x_bins_assigned = np.clip(x_bins_assigned, 0, x_bins - 1)
    y_bins_assigned = np.clip(y_bins_assigned, 0, y_bins - 1)

    # Create a 2D histogram
    hist2d = np.zeros((y_bins, x_bins))
    for i, j in zip(y_bins_assigned, x_bins_assigned):
        hist2d[i, j] += 1

    # Create bin centers for labels
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create a DataFrame for seaborn
    heatmap_df = pd.DataFrame(hist2d, index=y_centers, columns=x_centers)

    # Plot the heatmap
    sns.heatmap(heatmap_df, cmap="viridis", ax=ax, cbar_kws={"label": "Count"})

    # Add perfect agreement line (y=x)
    # Find points along the y=x line that fall within our bin range
    diag_points_x = np.linspace(
        max(x_edges[0], y_edges[0]), min(x_edges[-1], y_edges[-1]), 100
    )
    diag_points_y = diag_points_x

    # Convert to heatmap indices
    diag_x_idx = np.interp(diag_points_x, x_centers, range(len(x_centers)))
    diag_y_idx = np.interp(diag_points_y, y_centers, range(len(y_centers)))

    # Plot y=x line
    ax.plot(
        diag_x_idx,
        diag_y_idx,
        color="yellow",
        linestyle="--",
        linewidth=2,
        label="Perfect agreement (y=x)",
    )

    # Add axis lines at x=0 and y=0
    # Convert data coordinate 0 to the corresponding heatmap index
    x_zero_idx = np.interp(0, x_centers, range(len(x_centers)))
    y_zero_idx = np.interp(0, y_centers, range(len(y_centers)))

    # Add axis lines
    ax.axhline(y=y_zero_idx, color="grey", linestyle="--", linewidth=1)
    ax.axvline(x=x_zero_idx, color="grey", linestyle="--", linewidth=1)

    # Set custom tick labels for better readability (show more ticks)
    x_tick_indices = range(0, len(x_centers), 10)  # Every 10th bin to avoid crowding
    y_tick_indices = range(0, len(y_centers), 10)  # Every 10th bin to avoid crowding

    x_tick_labels = [f"{x_centers[i]:.1f}" for i in x_tick_indices]
    y_tick_labels = [f"{y_centers[i]:.1f}" for i in y_tick_indices]

    ax.set_xticks(x_tick_indices)
    ax.set_yticks(y_tick_indices)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)

    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(
        title
        + "\nProportion of predictions in 95% CI: {:.2f}".format(
            proportion_predictions_in_ci
        )
    )
    # Add legend
    ax.legend(loc="upper right")

    # Reset the y-axis orientation (negative at bottom, positive at top)
    # First invert the y-axis (seaborn heat maps start at top)
    ax.invert_yaxis()

    # Now invert the y-tick labels to have positive values at top and negative at bottom
    current_labels = ax.get_yticklabels()
    current_positions = ax.get_yticks()

    # Create new y-tick positions that are inverted from the current ones
    # This maintains the position but changes the labels
    new_labels = [float(label.get_text()) for label in current_labels]

    # Sort positions and labels to have negative values at bottom
    sorted_indices = np.argsort(new_labels)
    sorted_positions = current_positions[sorted_indices]
    sorted_labels = [new_labels[i] for i in sorted_indices]

    ax.set_yticks(sorted_positions)
    ax.set_yticklabels([f"{label:.1f}" for label in sorted_labels])

    plt.tight_layout()
    return fig
