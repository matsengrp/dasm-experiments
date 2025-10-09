import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from dnsmex.dxsm_model_summaries import abbreviate_number


def prep_compare_df(
    model_compare_csv_path, data_summary_csv_path, model_parameter_count_csv_path
):
    compare_df = pd.read_csv(model_compare_csv_path)
    compare_df["model"] = compare_df["model_nickname"].apply(lambda x: x.split("-")[0])
    compare_df["train dataset"] = compare_df["model_nickname"].apply(
        lambda x: x.split("-")[1]
    )
    compare_df["train method"] = compare_df["model_nickname"].apply(
        lambda x: x.split("-")[2]
    )
    compare_df = compare_df.drop(columns=["train method"])
    compare_df = compare_df.rename(columns={"data_description": "test dataset"})
    compare_df = compare_df.sort_values(
        by=["train dataset", "model", "test dataset"]
    ).reset_index(drop=True)

    model_df = pd.read_csv(model_parameter_count_csv_path)

    data_summary_df = pd.read_csv(data_summary_csv_path)
    data_summary_df = data_summary_df.rename(columns={"nickname": "train dataset"})
    compare_df = compare_df.merge(data_summary_df, on="train dataset")
    compare_df = compare_df.merge(model_df, on="model")

    return compare_df


def plot_performance_metrics(summary_df, metrics, out_dir, test_datasets=None):
    summary_df = summary_df.sort_values(by="pcps")
    if test_datasets is None:
        test_datasets = sorted(summary_df["test dataset"].unique())

    num_metrics = len(metrics)
    num_datasets = len(test_datasets)

    fig, axes = plt.subplots(
        num_metrics,
        num_datasets,
        figsize=(3 * num_datasets, 3 * num_metrics),
        sharex="col",
        sharey="row",
    )
    fig.tight_layout(pad=3.0)

    # Create a color palette for the models
    models = summary_df["model"].unique()
    palette = sns.color_palette("husl", len(models))
    model_colors = dict(zip(models, palette))

    for i, metric in enumerate(metrics):
        metric_summary_df = summary_df.dropna(axis=0, subset=[metric])

        for j, dataset in enumerate(test_datasets):
            ax = axes[i, j] if num_metrics > 1 and num_datasets > 1 else axes[i or j]
            subset = metric_summary_df[metric_summary_df["test dataset"] == dataset]
            models = subset["model"].unique()
            # model name: (max_pcps, corresponding metric_value)
            max_pcps_and_val_per_model = {
                name: tuple(group.loc[group["pcps"].idxmax()])
                for name, group in subset.groupby("model")[["pcps", metric]]
            }
            max_pcps = max(it[0] for it in max_pcps_and_val_per_model.values())

            # rank model_data[metric].values[-1]
            metric_values = [it[1] for it in max_pcps_and_val_per_model.values()]
            ranks = stats.rankdata(metric_values)
            min_metric = min(metric_values)
            max_metric = max(metric_values)
            # scale the ranks so they go from min_metric to max_metric
            y_values = (ranks - 1) / (len(ranks) - 1) * (
                max_metric - min_metric
            ) + min_metric
            y_value_dict = dict(zip(max_pcps_and_val_per_model.keys(), y_values))

            for model in models:
                model_data = subset[subset["model"] == model]
                param = model_data["parameters"].values[0]
                color = model_colors[model]
                ax.plot(
                    model_data["pcps"],
                    model_data[metric],
                    label=model,
                    alpha=0.7,
                    color=color,
                    marker="o",
                    markersize=3,
                )

                # Add direct labeling with the number of parameters
                text_x = 1.1 * max_pcps
                text_y = y_value_dict[model]
                ax.text(
                    text_x,
                    text_y,
                    abbreviate_number(param),
                    fontsize=10,
                    color=color,
                    verticalalignment="center",
                )
                final_x = model_data["pcps"].values[-1]
                final_y = model_data[metric].values[-1]
                # make a light gray dotted line from the text to the final point, quite thin
                ax.plot(
                    [0.99 * text_x, final_x],
                    [text_y, final_y],
                    color="gray",
                    linestyle="dashed",
                    linewidth=0.5,
                    alpha=0.7,
                )

            if i == 0:
                ax.set_title(dataset.split("_")[-1])
            if i == num_metrics - 1:
                ax.set_xlabel("PCP count")
            if j == 0:
                ax.set_ylabel(metric)

            sns.despine()

    # use abbreviated numbers for the x-axis
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: abbreviate_number(x))
        )

    plt.tight_layout()

    fig.savefig(out_dir)
