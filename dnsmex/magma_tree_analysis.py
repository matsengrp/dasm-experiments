#!/usr/bin/env python3
"""Tree analysis functions for MAGMA-seq mutation tree visualization."""

import pandas as pd
import numpy as np
import altair as alt
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path

from .local import localify


def load_uca_partition_data(uca_group: str) -> pd.DataFrame:
    """Load UCA partition data with existing assignments."""
    uca_file = f"_output/{uca_group}_model_scores.csv"

    if not os.path.exists(uca_file):
        raise FileNotFoundError(f"UCA partition file not found: {uca_file}")

    df = pd.read_csv(uca_file)

    # Add mutations_from_uca column using total_difference_count
    df["mutations_from_uca"] = df["total_difference_count"]

    print(
        f"📊 Raw mutation counts: min={df['mutations_from_uca'].min()}, max={df['mutations_from_uca'].max()}"
    )
    print(
        f"📊 Mutation distribution: {df['mutations_from_uca'].value_counts().sort_index().head(10).to_dict()}"
    )

    # Create sequence_id for compatibility
    df["sequence_id"] = df["VH"] + "|" + df["VL"]

    # Detect dataset type to handle different experimental designs
    dataset_type = df["dataset"].iloc[0] if "dataset" in df.columns else "Unknown"
    print(f"📊 Dataset type detected: {dataset_type}")

    if dataset_type == "Petersen":
        # Petersen data: mutations_from_uca = mutations from MATURE antibody
        # Need to adjust for true UCA distance
        print("🧬 Adjusting Petersen data for true UCA distance...")

        # Known distances from our UCA analysis
        if "222-1C06" in uca_group:
            true_uca_distance = -15  # 222-1C06 is 15 mutations from true UCA
        elif "319-345" in uca_group:
            true_uca_distance = -11  # 319-345 is 11 mutations from true UCA
        else:
            true_uca_distance = -12  # Default estimate for unknown Petersen antibodies

        print(f"📐 True UCA distance for {uca_group}: {true_uca_distance} mutations")

        # Adjust mutation counts: mature (0) → variants (+N) → true UCA (negative)
        # Keep existing mutations_from_uca as-is for experimental data

        # Mark existing UCA sequences (the synthetic 1500 nM ones)
        df["is_uca"] = (df["mutations_from_uca"] == 0) & (df["KD"] == 1500.0)

        # Add true UCA at negative position
        if not df["is_uca"].any():
            print(f"⚠️  Adding true UCA at position {true_uca_distance}")
            uca_vh = df["WT_VH"].iloc[0]
            uca_vl = df["WT_VL"].iloc[0]

            true_uca_row = pd.DataFrame(
                [
                    {
                        "VH": uca_vh,
                        "VL": uca_vl,
                        "KD": 1500.0,  # >1 μM as per Kirby paper
                        "mutations_from_uca": true_uca_distance,  # Show at negative position
                        "sequence_id": f"{uca_vh}|{uca_vl}_TRUE_UCA",
                        "is_uca": True,
                        "dataset": "Petersen",
                        "antibody": uca_group,
                        "WT_VH": uca_vh,
                        "WT_VL": uca_vl,
                    }
                ]
            )

            df = pd.concat([true_uca_row, df], ignore_index=True)
    else:
        # Kirby data: mutations_from_uca = mutations from TRUE UCA (original logic)
        df["is_uca"] = df["mutations_from_uca"] == 0

        # Ensure there's a proper UCA row for Kirby data
        if not df["is_uca"].any():
            print("⚠️  No UCA found in Kirby data, adding synthetic UCA row")
            uca_vh = df["WT_VH"].iloc[0]
            uca_vl = df["WT_VL"].iloc[0]

            uca_row = pd.DataFrame(
                [
                    {
                        "VH": uca_vh,
                        "VL": uca_vl,
                        "KD": df["KD"].median(),
                        "mutations_from_uca": 0,
                        "sequence_id": f"{uca_vh}|{uca_vl}",
                        "is_uca": True,
                        "dataset": "Kirby",
                        "antibody": uca_group,
                        "WT_VH": uca_vh,
                        "WT_VL": uca_vl,
                    }
                ]
            )

            df = pd.concat([uca_row, df], ignore_index=True)

    print(f"📊 Final UCA marking: {df['is_uca'].value_counts().to_dict()}")
    print(
        f"📊 Final mutation range: {df['mutations_from_uca'].min()} to {df['mutations_from_uca'].max()}"
    )
    print(f"✅ Data processing complete, shape: {df.shape}")
    return df


def identify_parent_child_pairs(df: pd.DataFrame, uca_group: str) -> List[Dict]:
    """Identify parent-child pairs differing by one mutation."""
    pairs = []
    sequences = df.to_dict("records")
    n = len(sequences)

    for i in range(n):
        for j in range(i + 1, n):
            seq1, seq2 = sequences[i], sequences[j]

            # Calculate mutations between sequences
            vh_diff = sum(1 for a, b in zip(seq1["VH"], seq2["VH"]) if a != b)
            vl_diff = sum(1 for a, b in zip(seq1["VL"], seq2["VL"]) if a != b)

            total_diff = vh_diff + vl_diff

            if total_diff == 1:
                # Determine parent (closer to UCA) and child
                if seq1["mutations_from_uca"] < seq2["mutations_from_uca"]:
                    parent, child = seq1, seq2
                elif seq1["mutations_from_uca"] > seq2["mutations_from_uca"]:
                    parent, child = seq2, seq1
                else:
                    continue  # Same distance to UCA, skip

                # Find the specific mutation between parent and child
                mutation_str = ""
                for k, (a, b) in enumerate(zip(parent["VH"], child["VH"])):
                    if a != b:
                        mutation_str = f"VH:{a}{k+1}{b}"
                        break
                if not mutation_str:  # Check light chain if no heavy chain mutation
                    for k, (a, b) in enumerate(zip(parent["VL"], child["VL"])):
                        if a != b:
                            mutation_str = f"VL:{a}{k+1}{b}"
                            break

                pairs.append(
                    {
                        "parent_id": parent["sequence_id"],
                        "child_id": child["sequence_id"],
                        "parent_mutations": parent["mutations_from_uca"],
                        "child_mutations": child["mutations_from_uca"],
                        "parent_kd": parent["KD"],
                        "child_kd": child["KD"],
                        "mutation": mutation_str,
                        "delta_log_kd": np.log10(child["KD"]) - np.log10(parent["KD"]),
                    }
                )

    return pairs


def create_mutation_plot(
    tree_data: pd.DataFrame, edges: List[Dict], title: str, dataset_type: str = None
) -> alt.Chart:
    """Create interactive Altair plot for mutation tree."""

    print(f"🎨 Creating plot for {len(tree_data)} sequences, {len(edges)} edges")

    # Create edge dataframe for lines
    edge_df = pd.DataFrame(edges)

    # Merge edge data with node positions
    if len(edge_df) > 0:
        edge_data = []
        for _, edge in edge_df.iterrows():
            parent = tree_data[tree_data["sequence_id"] == edge["parent_id"]].iloc[0]
            child = tree_data[tree_data["sequence_id"] == edge["child_id"]].iloc[0]

            edge_data.append(
                {
                    "x": parent["mutations_from_uca"],
                    "y": parent["KD"],
                    "x2": child["mutations_from_uca"],
                    "y2": child["KD"],
                }
            )

        edge_plot_df = pd.DataFrame(edge_data)
        print(f"📏 Edge data: {len(edge_plot_df)} lines")
    else:
        edge_plot_df = pd.DataFrame()
        print("📏 No edges to plot")

    # Always set dataset based on is_uca (overrides any existing dataset column)
    tree_data = tree_data.copy()
    tree_data["dataset"] = tree_data["is_uca"].apply(
        lambda is_uca: "UCA" if is_uca else "Mutant"
    )

    print(
        f"🔍 Data ranges: KD {tree_data['KD'].min():.1f}-{tree_data['KD'].max():.1f}, Mutations {tree_data['mutations_from_uca'].min()}-{tree_data['mutations_from_uca'].max()}"
    )
    print(f"🔍 Dataset counts: {tree_data['dataset'].value_counts().to_dict()}")
    print(
        f"🔍 Mutation distribution: {tree_data['mutations_from_uca'].value_counts().sort_index().head(10).to_dict()}"
    )

    # Check if all mutations are 0
    non_zero_mutations = tree_data[tree_data["mutations_from_uca"] > 0]
    print(f"🔍 Non-zero mutations: {len(non_zero_mutations)} sequences")
    if len(non_zero_mutations) > 0:
        print(
            f"🔍 Sample non-zero: {non_zero_mutations[['mutations_from_uca', 'KD']].head(3).to_dict('records')}"
        )
    else:
        print("⚠️  All sequences have 0 mutations - data loading issue!")

    # Base chart dimensions (doubled for better visibility)
    width = 1200
    height = 800

    # Debug: Show what data we're plotting
    print(f"🎯 Plotting data sample:")
    sample_data = tree_data[["mutations_from_uca", "KD", "dataset", "is_uca"]].head(10)
    print(sample_data.to_string())

    # Check for data issues
    print(f"🔍 Data validation:")
    print(f"  - Null KD: {tree_data['KD'].isnull().sum()}")
    print(f"  - Null mutations: {tree_data['mutations_from_uca'].isnull().sum()}")
    print(f"  - KD data types: {tree_data['KD'].dtype}")
    print(f"  - Mutations data types: {tree_data['mutations_from_uca'].dtype}")
    print(
        f"  - Non-finite KD: {(~tree_data['KD'].apply(lambda x: x > 0 and x != float('inf'))).sum()}"
    )

    # Clean data for Altair
    tree_data_clean = tree_data.dropna(subset=["KD", "mutations_from_uca"])
    tree_data_clean = tree_data_clean[tree_data_clean["KD"] > 0]
    tree_data_clean = tree_data_clean[tree_data_clean["KD"] != float("inf")]

    print(f"🧹 After cleaning: {len(tree_data_clean)}/{len(tree_data)} sequences")

    # Create node chart (points) with explicit domain
    x_min, x_max = (
        tree_data_clean["mutations_from_uca"].min(),
        tree_data_clean["mutations_from_uca"].max(),
    )
    y_min, y_max = tree_data_clean["KD"].min(), tree_data_clean["KD"].max()

    print(f"🎯 Chart domains: X=[{x_min}, {x_max}], Y=[{y_min:.1f}, {y_max:.1f}]")

    # Ultra-simplified nodes-only chart to avoid layer issues
    print(f"🎯 Creating nodes-only chart (no edges for now)...")

    # Include mutation information for tooltips
    tooltip_columns = ["mutations_from_uca", "KD", "dataset", "is_uca"]

    # Add mutation details if available
    if "heavy_differences" in tree_data.columns:
        tooltip_columns.append("heavy_differences")
    if "light_differences" in tree_data.columns:
        tooltip_columns.append("light_differences")
    if "antibody" in tree_data.columns:
        tooltip_columns.append("antibody")

    simple_data = tree_data[tooltip_columns].copy()

    # Clean up mutation strings for display
    if "heavy_differences" in simple_data.columns:
        simple_data["heavy_mutations"] = simple_data["heavy_differences"].apply(
            lambda x: (
                x.replace("['", "").replace("']", "").replace("'", "")
                if pd.notna(x) and x != "[]"
                else "None"
            )
        )
    if "light_differences" in simple_data.columns:
        simple_data["light_mutations"] = simple_data["light_differences"].apply(
            lambda x: (
                x.replace("['", "").replace("']", "").replace("'", "")
                if pd.notna(x) and x != "[]"
                else "None"
            )
        )

    print(f"🎯 Data with mutations sample:\n{simple_data.head()}")

    # Determine axis label based on dataset type (passed from main script)
    if dataset_type == "Petersen":
        x_axis_title = "Mutations from WT (Wild-Type)"
        tooltip_x_title = "Mutations from WT"
    else:
        x_axis_title = "Number of Mutations from UCA"
        tooltip_x_title = "Total Mutations from UCA"

    print(f"📊 Using X-axis title: '{x_axis_title}' for {dataset_type} data")

    nodes_chart = (
        alt.Chart(simple_data)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X(
                "mutations_from_uca:Q",
                title=x_axis_title,
                axis=alt.Axis(
                    titleFontSize=12, labelFontSize=10, tickMinStep=1, format="d"
                ),
            ),
            y=alt.Y(
                "KD:Q",
                title="Binding Affinity KD (nM)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(titleFontSize=12, labelFontSize=10),
            ),
            color=alt.value("#1f77b4"),  # Remove legend, use single color
            tooltip=[
                alt.Tooltip("mutations_from_uca:Q", title=tooltip_x_title),
                alt.Tooltip("KD:Q", title="Binding Affinity (nM)", format=".3f"),
                alt.Tooltip("dataset:N", title="Sequence Type"),
            ]
            + (
                [alt.Tooltip("heavy_mutations:N", title="Heavy Chain Mutations")]
                if "heavy_mutations" in simple_data.columns
                else []
            )
            + (
                [alt.Tooltip("light_mutations:N", title="Light Chain Mutations")]
                if "light_mutations" in simple_data.columns
                else []
            ),
        )
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(text=title, fontSize=14, fontWeight="bold"),
        )
    )

    # Add edges back now that data embedding works
    if len(edge_plot_df) > 0:
        print(f"🔗 Adding {len(edge_plot_df)} parent-child edges...")

        # Create simple edge chart
        edges_chart = (
            alt.Chart(edge_plot_df)
            .mark_line(opacity=0.3, color="gray", strokeWidth=1)
            .encode(
                x=alt.X(
                    "x:Q", title=x_axis_title, axis=alt.Axis(tickMinStep=1, format="d")
                ),
                y=alt.Y(
                    "y:Q", title="Binding Affinity KD (nM)", scale=alt.Scale(type="log")
                ),
                x2="x2:Q",
                y2="y2:Q",
            )
        )

        # Add invisible points on edges for tooltips
        if "mutation" in edge_df.columns:
            # Merge edge data with original edges info to get mutations
            edge_tooltip_data = []
            for idx, edge in enumerate(edges):
                # Find corresponding edge in edge_plot_df
                parent = tree_data[tree_data["sequence_id"] == edge["parent_id"]].iloc[
                    0
                ]
                child = tree_data[tree_data["sequence_id"] == edge["child_id"]].iloc[0]

                # Calculate midpoint for tooltip placement
                mid_x = (parent["mutations_from_uca"] + child["mutations_from_uca"]) / 2
                mid_y = np.sqrt(
                    parent["KD"] * child["KD"]
                )  # Geometric mean for log scale

                edge_tooltip_data.append(
                    {
                        "x": mid_x,
                        "y": mid_y,
                        "mutation": edge.get("mutation", ""),
                        "delta_log_kd": edge.get("delta_log_kd", 0),
                        "parent_kd": parent["KD"],
                        "child_kd": child["KD"],
                        "direction": (
                            "Improved"
                            if edge.get("delta_log_kd", 0) < 0
                            else "Degraded"
                        ),
                    }
                )

            edge_tooltip_df = pd.DataFrame(edge_tooltip_data)

            # Create small diamond points for edge tooltips
            edge_tooltip_chart = (
                alt.Chart(edge_tooltip_df)
                .mark_point(
                    shape="diamond", size=30, opacity=0.6, color="darkgray"
                )  # Small visible diamonds
                .encode(
                    x=alt.X("x:Q", title=x_axis_title),
                    y=alt.Y(
                        "y:Q",
                        title="Binding Affinity KD (nM)",
                        scale=alt.Scale(type="log"),
                    ),
                    tooltip=[
                        alt.Tooltip("mutation:N", title="Mutation"),
                        alt.Tooltip(
                            "parent_kd:Q", title="Parent KD (nM)", format=".2f"
                        ),
                        alt.Tooltip("child_kd:Q", title="Child KD (nM)", format=".2f"),
                        alt.Tooltip("delta_log_kd:Q", title="ΔlogKD", format=".3f"),
                        alt.Tooltip("direction:N", title="Effect"),
                    ],
                )
            )

            # Layer charts: edges behind, invisible tooltip points in middle, nodes on top
            chart = edges_chart + edge_tooltip_chart + nodes_chart
            print("✅ Created chart with edges, tooltips, and nodes")
        else:
            # Layer charts: edges behind, nodes on top
            chart = edges_chart + nodes_chart
            print("✅ Created chart with edges and nodes")
    else:
        chart = nodes_chart
        print("✅ Created nodes-only chart (no edges found)")

    return chart


def analyze_tree_metrics(tree_data: pd.DataFrame, edges: List[Dict]) -> Dict:
    """Calculate metrics for the mutation tree."""
    metrics = {
        "n_sequences": len(tree_data) - len(tree_data[tree_data["is_uca"]]),
        "n_edges": len(edges),
        "max_mutations": tree_data[~tree_data["is_uca"]]["mutations_from_uca"].max(),
        "mean_kd": tree_data[~tree_data["is_uca"]]["KD"].mean(),
        "median_kd": tree_data[~tree_data["is_uca"]]["KD"].median(),
        "kd_range": (
            tree_data[~tree_data["is_uca"]]["KD"].min(),
            tree_data[~tree_data["is_uca"]]["KD"].max(),
        ),
    }

    # Calculate improvement/degradation from mutations
    if len(edges) > 0:
        edge_df = pd.DataFrame(edges)
        improvements = edge_df[edge_df["child_kd"] < edge_df["parent_kd"]]
        degradations = edge_df[edge_df["child_kd"] > edge_df["parent_kd"]]

        metrics["n_improvements"] = len(improvements)
        metrics["n_degradations"] = len(degradations)
        metrics["improvement_rate"] = (
            len(improvements) / len(edges) if len(edges) > 0 else 0
        )

    return metrics
