"""Simulation framework for clonal family evolution based on DXSM models."""

import numpy as np
from pathlib import Path
import torch
import yaml
from ete3 import Tree
from tqdm.auto import tqdm
import pandas as pd
from typing import Tuple

from netam.common import chunked
from netam.sequences import (
    contains_stop_codon,
    HL_SEPARATOR_TOKEN_IDX,
)
from netam.data_format import _all_pcp_df_columns
from netam.framework import (
    codon_probs_of_parent_seq,
    sample_sequence_from_codon_probs,
    load_pcp_df,
    add_shm_model_outputs_to_pcp_df,
)
from dnsmex.dxsm_data import filter_valid_pcps
from netam.framework import load_crepe
from netam import pretrained

from netam.dasm import (
    DASMBurrito,
    DASMDataset,
)
from netam.dnsm import (
    DNSMBurrito,
    DNSMDataset,
)

import dnsmex.dasm_oe
import dnsmex.dnsm_oe

_tree_file_columns = _all_pcp_df_columns | {
    "naive_sequence_heavy": str,
    "naive_sequence_light": str,
    "rate_scale_heavy": float,
    "rate_scale_light": float,
    "newick": str,
}

_dxsm_classes_of_name = {
    "dasm": (DASMDataset, DASMBurrito),
    "dnsm": (DNSMDataset, DNSMBurrito),
}

_dxsm_oe_of_name = {
    "dasm": dnsmex.dasm_oe.OEPlotter,
    "dnsm": dnsmex.dnsm_oe.OEPlotter,
}


def read_tree_file(tree_file_path: str):
    """Load a raw or prepared tree file containing clonal families."""
    trees_df = (
        pd.read_csv(tree_file_path, keep_default_na=False, dtype=_tree_file_columns)
        .drop(columns=["Unnamed: 0"], errors="ignore")
        .reset_index(drop=True)
    )
    return trees_df


def _remove_extensions(filepath: Path) -> Path:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    while filepath.suffix:
        filepath = filepath.with_suffix("")
    return filepath


def simulate_clonal_family(
    tree: Tree,
    naive_sequence: Tuple[str, str],
    selection_crepe,
    neutral_crepe=None,
    multihit_model=None,
):
    """Simulate mutations along a phylogenetic tree starting from a naive sequence.

    For reproducibility, use `torch.manual_seed` before calling this function.

    Args:
        tree: An ete3 Tree object with branch lengths
        naive_sequence: The naive (root) nucleotide sequence as a heavy-light pair
        selection_crepe: A DXSM model for computing selection factors
        neutral_crepe: A Netam neutral model. If not provided, this will be determined from selection_crepe metadata.
        multihit_model: An optional multihit model for adjusting neutral codon probabilities. If not provided, this will be determined from selection_crepe metadata.

    Returns:
        The provided tree with simulated sequences stored as node ``sequence`` attributes
    """
    tree.add_feature("sequence", naive_sequence)
    if neutral_crepe is None:
        neutral_crepe = pretrained.load(selection_crepe.model.neutral_model_name)
    if multihit_model is None:
        multihit_model = pretrained.load_multihit(
            selection_crepe.model.multihit_model_name
        )

    for node in tree.iter_descendants(strategy="preorder"):
        if node.dist == 0.0:
            # Numerical instability surprisingly samples 0-probability codons
            # fairly often if we do the standard thing
            new_sequence = node.up.sequence
        else:
            new_sequence = tuple(
                sample_sequence_from_codon_probs(codon_probs)
                for codon_probs in codon_probs_of_parent_seq(
                    selection_crepe,
                    node.up.sequence,
                    node.dist,
                    neutral_crepe,
                    multihit_model=multihit_model,
                )
            )
        node.add_feature("sequence", new_sequence)
    return tree


def simulate_dataset(
    clonal_family_dataframe: pd.DataFrame,
    selection_crepe,
):
    """Create a pcp_df from newick strings containing branch lengths, and associated
    naive heavy-light nt sequence pairs.

    Input can be prepared with `add_branch_lengths_to_trees`.

    For reproducibility, use `torch.manual_seed` before calling this function.

    Consider running `dxsm_data.filter_valid_pcps` and adding shm outputs with `netam.framework.add_shm_model_outputs_to_pcp_df` on the output before using it for DXSM training.
    Args:
        clonal_family_dataframe: A DataFrame with columns `newick` containing newick strings
            with branch lengths, `naive_sequence_heavy` and `naive_sequence_light` containing heavy and
            light chain naive sequences (both are required, but one may be empty), and any other columns desired (at least `v_gene` is
            required to load the resulting pcp_df as a Dataset, or `v_gene_heavy` and `v_gene_light` if both heavy and light chain sequences are provided).
        selection_crepe: A DXSM model for computing selection factors
    Returns:
        A DataFrame with simulated PCPs from the clonal families provided.
            All columns from the input dataframe other than `newick` and
            `naive_sequence_*` will be copied to the new dataframe,
            with data for each PCP duplicated from the corresponding clonal family.
            Additional columns `parent_name`, `child_name`, `branch_length`, `depth`, `distance`, `parent_is_naive`, and `child_is_leaf` will also be added.
    """
    neutral_crepe = pretrained.load(selection_crepe.model.neutral_model_name)
    multihit_model = pretrained.load_multihit(selection_crepe.model.multihit_model_name)
    pass_through_columns = [
        col
        for col in clonal_family_dataframe.columns
        if col not in ["newick", "naive_sequence_heavy", "naive_sequence_light"]
    ]
    trees = []
    pcps = {col: [] for col in pass_through_columns}
    pcps.update(
        {"parent_heavy": [], "parent_light": [], "child_heavy": [], "child_light": []}
    )
    pcps.update(
        {
            "parent_name": [],
            "child_name": [],
            "branch_length": [],
            "depth": [],
            "distance": [],
            "parent_is_naive": [],
            "child_is_leaf": [],
        }
    )

    pbar = tqdm(total=None, desc="Simulating PCPs", unit="pcp", position=1)
    pbar.update(0)
    for row in tqdm(
        clonal_family_dataframe.itertuples(),
        total=len(clonal_family_dataframe),
        desc="Simulating PCPs",
        unit="clonal families",
        position=0,
        smoothing=0,
    ):
        tree = Tree(row.newick, format=1)
        simulate_clonal_family(
            tree,
            (row.naive_sequence_heavy, row.naive_sequence_light),
            selection_crepe,
            neutral_crepe=neutral_crepe,
            multihit_model=multihit_model,
        )
        trees.append(tree)
        tree_pcps = 0
        for node in tree.iter_descendants():
            pcps["parent_heavy"].append(node.up.sequence[0])
            pcps["parent_light"].append(node.up.sequence[1])
            pcps["child_heavy"].append(node.sequence[0])
            pcps["child_light"].append(node.sequence[1])
            for col in pass_through_columns:
                pcps[col].append(getattr(row, col))
            pcps["parent_name"].append(node.up.name)
            pcps["child_name"].append(node.name)
            pcps["branch_length"].append(node.dist)
            pcps["distance"].append(node.get_distance(tree))
            pcps["depth"].append(node.get_distance(tree, topology_only=True))
            pcps["parent_is_naive"].append(node.up.is_root())
            pcps["child_is_leaf"].append(node.is_leaf())
            # Update progress
            tree_pcps += 1
        # Update progress bar with PCPs from this tree
        pbar.update(tree_pcps)
    pbar.close()
    pcp_df = pd.DataFrame(pcps)
    if pcp_df["parent_heavy"].str.len().max() == 0:
        # Make this a bulk light dataset
        pcp_df.drop(
            columns=[col for col in pcp_df.columns if col.endswith("_heavy")],
            inplace=True,
        )
    if pcp_df["parent_light"].str.len().max() == 0:
        # Make this a bulk heavy dataset
        pcp_df.drop(
            columns=[col for col in pcp_df.columns if col.endswith("_light")],
            inplace=True,
        )
    return pcp_df


_pcp_df_columns = [
    "sample_id",
    "family",
    "v_gene_heavy",
    "j_gene_heavy",
    "cdr1_codon_start_heavy",
    "cdr1_codon_end_heavy",
    "cdr2_codon_start_heavy",
    "cdr2_codon_end_heavy",
    "cdr3_codon_start_heavy",
    "cdr3_codon_end_heavy",
    "v_gene_light",
    "j_gene_light",
    "cdr1_codon_start_light",
    "cdr1_codon_end_light",
    "cdr2_codon_start_light",
    "cdr2_codon_end_light",
    "cdr3_codon_start_light",
    "cdr3_codon_end_light",
    "light_chain_type",
]


def _check_paired_model(model, pcp_df):
    if (pcp_df["parent_heavy"].str.len() > 0).any() and (
        pcp_df["parent_light"].str.len() > 0
    ).any():
        # make sure model is paired model:
        if model.hyperparameters["known_token_count"] <= HL_SEPARATOR_TOKEN_IDX:
            raise ValueError(
                "Model is not a paired model, but pcp_df contains both "
                "heavy and light chains. If this fails, provide a bulk "
                "pcp file with sequences only for the chain of interest."
            )


def apply_func_to_chunked_df(df, func, batch_size, concat_func=torch.concat):
    if batch_size is None:
        chunk_size = len(df)
    else:
        n_chunks = int(len(df) / batch_size + 1)
        chunk_size = (len(df) // n_chunks) + 1
    position_chunks = list(chunked(range(len(df)), chunk_size))
    n_chunks = len(position_chunks)

    df_chunks = (df.iloc[position_chunk] for position_chunk in position_chunks)

    results = []
    for idx, df_chunk in enumerate(df_chunks):
        if n_chunks > 1:
            print(f"Processing batch {idx + 1} of {n_chunks}")
        results.append(func(df_chunk))
        del df_chunk
    if concat_func is not None:
        return concat_func(results)
    else:
        return results


def fit_branch_lengths(selection_crepe, pcp_df, batch_size=500_000):
    """Fit branch lengths to a pcp_df using the provided selection crepe.

    Returns a list of branch lengths, corresponding to the rows in pcp_df.

    batch_size can be None to use a single batch, or an integer to specify the batch
    size. Returns a list of branch lengths, corresponding to the rows in pcp_df.
    """
    model_type = selection_crepe.model.hyperparameters["model_type"]
    print("model type: ", model_type)
    dataset_cls, burrito_cls = _dxsm_classes_of_name[model_type]
    known_token_count = selection_crepe.model.hyperparameters["known_token_count"]
    neutral_crepe = pretrained.load(selection_crepe.model.neutral_model_name)
    multihit_model = pretrained.load_multihit(selection_crepe.model.multihit_model_name)
    # Make val dataset from pcp_df:
    pcp_df = add_shm_model_outputs_to_pcp_df(pcp_df, neutral_crepe)
    pcp_df["in_train"] = False

    def _branch_length_fit_helper(df_chunk):
        _, val_dataset = dataset_cls.train_val_datasets_of_pcp_df(
            df_chunk, known_token_count, multihit_model=multihit_model
        )

        burrito = burrito_cls(
            None,
            val_dataset,
            selection_crepe.model,
        )

        burrito.standardize_and_optimize_branch_lengths()
        result = burrito.val_dataset.branch_lengths
        del burrito, val_dataset
        return result

    return apply_func_to_chunked_df(pcp_df, _branch_length_fit_helper, batch_size)


def add_branch_lengths_to_trees(
    pcp_path,
    trees_path,
    selection_crepe,
    remove_naive_branch=False,
    transfer_columns=_pcp_df_columns,
    filter_clonal_families_with_ambiguities=True,
):
    """Create a dataframe of clonal family trees with naive sequences and branch lengths
    fit using the provided selection crepe.

    Args:
        pcp_path: Path to a file containing a pcp_df with heavy and light sequences.
        trees_path: Path to a file containing clonal family trees in newick format.
        selection_crepe: A DXSM model for computing selection factors.
        remove_naive_branch: If True, the naive branch will be removed from the tree, and the 'naive' sequence will be its (only) child sequence.
        transfer_columns: Columns to transfer from the pcp_df to the trees_df.
        filter_clonal_families_with_ambiguities: If True, clonal families whose naive sequences contain ambiguities will be filtered out.

    Returns:
        A DataFrame containing the clonal family trees with fitted branch lengths, columns for naive heavy and light chain sequences, and any other columns from the input pcp dataframe as specified by `transfer_columns`.
    """
    pcp_df = load_pcp_df(pcp_path)
    print(f"loaded pcps from {pcp_path}")
    trees_df = read_tree_file(trees_path)
    print(f"loaded trees from {trees_path}")

    # Filter pcp_df to only include rows with sample_id and family values in trees_df.
    sample_ids = set(trees_df["sample_id"].unique())
    families = set(trees_df["family"].unique())
    pcp_df = pcp_df[
        pcp_df["sample_id"].isin(sample_ids) & pcp_df["family"].isin(families)
    ]
    _check_paired_model(selection_crepe.model, pcp_df)

    # Pre-compute stop codon mask to avoid repeated function calls.
    has_stop = pcp_df["child_heavy"].apply(contains_stop_codon) | pcp_df[
        "child_light"
    ].apply(contains_stop_codon)
    nodes_with_stop_codon = set(
        (row.sample_id, row.family, row.child_name)
        for row in pcp_df[has_stop].itertuples(index=False)
    )

    # Keep complete_pcp_df only with necessary columns for memory efficiency.
    required_cols = set(transfer_columns) | {
        "sample_id",
        "family",
        "parent_name",
        "parent_heavy",
        "parent_light",
    }
    complete_pcp_df = pcp_df[
        [col for col in required_cols if col in pcp_df.columns]
    ].copy()

    # Filter out stop codons and invalid PCPs.
    pcp_df = pcp_df[~has_stop]
    # All remaining pcps are assumed to have zero length if they're not valid
    # for dataset construction
    # Filter valid pcps:
    pcp_df = filter_valid_pcps(pcp_df)
    assert len(pcp_df) > 0, "No valid pcps found after filtering!"
    pcp_df = pcp_df.reset_index(drop=True)

    pcp_df["branch_lengths"] = fit_branch_lengths(selection_crepe, pcp_df)

    print("Adding branch lengths to clonal family trees")

    # Build mapping from sample_id, family, child_name triples to branch length.
    pcp_branch_mapping = {
        (row.sample_id, row.family, row.child_name): row.branch_length
        for row in pcp_df.itertuples(index=False)
    }

    # Pre-build root data mapping for faster lookup.
    root_data_mapping = {}
    for row in complete_pcp_df.itertuples(index=False):
        key = (row.sample_id, row.family, row.parent_name)
        root_data_mapping[key] = row

    # Process trees.
    new_newicks = []
    new_column_data = {
        col: [] for col in transfer_columns if col in complete_pcp_df.columns
    }
    new_column_data.update({"naive_sequence_heavy": [], "naive_sequence_light": []})
    failed_tree_indices = []

    for tree_index, row in tqdm(
        enumerate(trees_df.itertuples(index=False)),
        desc="Processing trees",
        unit="trees",
        total=len(trees_df),
    ):
        tree = Tree(row.newick, format=1)
        naive_node = tree & "naive"
        assert naive_node.is_leaf()

        if remove_naive_branch:
            naive_node.detach()
        else:
            # Invert naive branch so that naive node is at the root.
            naive_node.detach()
            tree.dist = naive_node.dist
            naive_node.dist = 0
            naive_node.add_child(tree)
            tree = naive_node

        # Process all nodes in a single pass.
        nodes_to_detach = []
        for node in tree.iter_descendants(strategy="preorder"):
            branch_key = (row.sample_id, row.family, node.name)

            if branch_key in pcp_branch_mapping:
                node.dist = pcp_branch_mapping[branch_key]
            elif branch_key in nodes_with_stop_codon:
                nodes_to_detach.append(node)
            else:
                # Assume edge had no mutations.
                node.dist = 0.0

        # Remove nodes with stop codons.
        if len(nodes_to_detach) > 0:
            n_count = sum(1 for _ in tree.iter_descendants())
            for node in nodes_to_detach:
                node.detach()
            new_count = sum(1 for _ in tree.iter_descendants())
            print(f"removed {n_count - new_count} nodes due to stop codons")

        # Get tree data for root node using pre-built mapping.
        root_key = (row.sample_id, row.family, tree.name)
        root_row = root_data_mapping.get(root_key)

        if root_row is None:
            print("No data found for root node of tree... skipping!")
            failed_tree_indices.append(tree_index)
            continue

        if filter_clonal_families_with_ambiguities and (
            "N" in root_row.parent_heavy or "N" in root_row.parent_light
        ):
            failed_tree_indices.append(tree_index)
            continue

        new_newicks.append(tree.write(format=1, format_root_node=True))
        for key, val_list in new_column_data.items():
            if key not in ["naive_sequence_heavy", "naive_sequence_light"]:
                val_list.append(getattr(root_row, key))
        new_column_data["naive_sequence_heavy"].append(root_row.parent_heavy)
        new_column_data["naive_sequence_light"].append(root_row.parent_light)

    # Update trees_df efficiently.
    trees_df = trees_df.drop(index=failed_tree_indices).reset_index(drop=True)
    trees_df["newick"] = new_newicks
    for key, val_list in new_column_data.items():
        trees_df[key] = val_list

    return trees_df


def read_simulation_yaml(yaml_path: str) -> dict:
    """Read a YAML file containing simulation parameters.

    The yaml file must contain keys:
        - tree_file: path to file containing the clonal family trees
        - pcp_df: path to file containing all pcps for the provided clonal
          family trees
        - selection_crepe_path: path to the selection crepe
        - seed: seed for random number generation

    The yaml file may also contain keys:
        - replicates: Number of replicates to simulate (default 1). If greater
          than 1, the seed will be incremented for each replicate, and the results
          will be concatenated into the output dataframe.
        - anarci_paths: paths to ANARCI files, required for simulation
          validation OE plotting. Use sub-keys "heavy" and "light" for heavy and light chain files.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        A dictionary containing the simulation parameters.
    """
    # Read yaml file
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as file:
        sim_data = yaml.safe_load(file)

    for key in [
        "tree_file",
        "pcp_df",
        "selection_crepe_path",
        "seed",
    ]:
        if key not in sim_data:
            raise ValueError(f"Missing required key '{key}' in YAML file.")
    defaults = {"replicates": 1}
    defaults.update(sim_data)
    return defaults


def _prepared_tree_path_of_inputs(yaml_path, output_dir):
    yaml_path = Path(yaml_path)
    return Path(output_dir) / (yaml_path.stem + ".prepared_clonal_families.csv.gz")


def fit_tree_branch_lengths_from_yaml(
    yaml_path: str,
    output_dir: str = "./",
    output_file: str = None,
    filter_clonal_families_with_ambiguities=True,
):
    """This function fits branch lengths to clonal family trees from a YAML file.

    Keys expected in the yaml file are described in the docstring for the `read_simulation_yaml` function.

    Args:
        yaml_path: Path to the YAML file.
        output_dir: Directory to save the output files.
        output_file: Path for the output prepared tree file (recommend .csv.gz).

    Saves the prepared clonal family trees to the output file.
    Also checkpoints the clonal family trees with branch lengths added to the output directory.
    """
    sim_data = read_simulation_yaml(yaml_path)

    output_dir = Path(output_dir)
    if output_file is None:
        out_path = _prepared_tree_path_of_inputs(yaml_path, output_dir)
    else:
        out_path = output_dir / output_file
    if out_path.exists():
        print(f"File {out_path} already exists, skipping.")
        return

    torch.manual_seed(sim_data["seed"])
    selection_crepe = load_crepe(sim_data["selection_crepe_path"])
    # Use the crepes to be used in simulation for fitting branch lengths on the
    # trees provided
    trees_df = add_branch_lengths_to_trees(
        sim_data["pcp_df"],
        sim_data["tree_file"],
        selection_crepe,
        filter_clonal_families_with_ambiguities=filter_clonal_families_with_ambiguities,
    )
    trees_df.to_csv(out_path, index=False)
    print(f"Saved prepared trees to {out_path}")


def _dataset_path_of_inputs(output_path, output_file, yaml_path, provided_seed):
    out_dir = Path(output_path)
    if provided_seed is None:
        seed_string = ""
    else:
        seed_string = "_seed" + str(provided_seed)

    if output_file is None:
        output_file = out_dir / (Path(yaml_path).stem + seed_string + ".csv.gz")
    else:
        output_file = out_dir / output_file

    return Path(output_file)


def simulate_dataset_from_yaml(
    yaml_path: str,
    output_path: str = "./",
    prepared_trees_file: str = None,
    output_file: str = None,
    seed: int = None,
):
    """This function simulates a dataset from a YAML file.

    Keys expected in the yaml file are described in the docstring for the `read_simulation_yaml` function.

    Args:
        yaml_path: Path to the YAML file.
        output_path: Directory to save the output files.
        prepared_trees_file: Path to the prepared clonal family trees file.
        output_file: Alternative name for the output file (recommend .csv.gz extension).
        seed: Alternative seed for choosing mutations. If provided, overrides the
            seed in the provided yaml file, and is reflected in the default output filename.

    Saves the simulated dataset to the outut directory, with the same name as the provided yaml file.
    """
    sim_data = read_simulation_yaml(yaml_path)

    if prepared_trees_file is None:
        clonal_family_file = _prepared_tree_path_of_inputs(yaml_path, output_path)
    else:
        clonal_family_file = Path(prepared_trees_file)

    selection_crepe = load_crepe(sim_data["selection_crepe_path"])
    if not clonal_family_file.exists():
        raise FileNotFoundError(
            f"File {clonal_family_file} does not exist! Please run the branch length optimization first."
        )
    trees_df = read_tree_file(clonal_family_file)

    out_path = _dataset_path_of_inputs(output_path, output_file, yaml_path, seed)
    if out_path.exists():
        print(f"File {out_path} already exists, skipping.")
        return

    result_dfs = []
    for idx in range(sim_data["replicates"]):
        if seed is None:
            seed = sim_data["seed"]
        torch.manual_seed(seed)
        print(
            f"Simulating replicate {idx + 1} of {sim_data['replicates']} with seed {seed}"
        )
        # use the prepared trees to simulate a dataset
        pcp_df = simulate_dataset(trees_df, selection_crepe)
        seed += 1
        result_dfs.append(pcp_df)

    # Concatenate the results into a single DataFrame
    combined_df = pd.concat(result_dfs, ignore_index=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Check again, since this function may take awhile
    if out_path.exists():
        print(f"File {out_path} already exists, skipping.")
        return
    combined_df.to_csv(out_path)
    print(f"Saved simulated dataset to {out_path}")


def validate_simulation_from_yaml(
    yaml_path: str,
    dataset_path: str,
    refit_branch_lengths=False,
    filter_valid=False,
    max_validation_samples: int = 200_000,
):
    """Validate the simulation from a YAML file.

    Keys expected in the yaml file are described in the docstring for the `read_simulation_yaml` function.
    This function produces OE plots written to `<yaml_filename>_oe_plots.pdf`.

    Args:
        yaml_path: Path to the YAML file.
        dataset_path: Path to the dataset file.
        refit_branch_lengths: If True, refit the branch lengths using the simulated
            PCPs. Otherwise, uses the branch lengths that were computed for clonal family trees.
        filter_valid: If True, filter simulated PCPs to only include those which are non-identical.
            If False, this condition must be met by the provided dataset.
        max_validation_samples: Maximum number of samples to use for validation. If provided dataset exceeds this size, a stratified downsample of this size will be used for validation.
    """
    sim_data = read_simulation_yaml(yaml_path)

    selection_crepe = load_crepe(sim_data["selection_crepe_path"])
    model_type = selection_crepe.model.hyperparameters["model_type"]
    plotter_class = _dxsm_oe_of_name[model_type]
    pcp_df = load_pcp_df(dataset_path).reset_index(drop=True)
    if max_validation_samples < len(pcp_df):
        pcp_df = pcp_df.iloc[
            np.linspace(0, len(pcp_df) - 1, max_validation_samples, dtype=int)
        ]
    if filter_valid:
        pcp_df = filter_valid_pcps(pcp_df)

    valid_index = pcp_df.index
    pcp_df = pcp_df.reset_index(drop=True)
    branch_lengths_path = Path("./") / (
        _remove_extensions(yaml_path).stem
        + "_FOR_"
        + _remove_extensions(dataset_path).stem
        + "_test_branch_lengths.csv"
    )
    if refit_branch_lengths:
        pd.DataFrame(
            {"branch_length": fit_branch_lengths(selection_crepe, pcp_df)}
        ).to_csv(branch_lengths_path, index=False)
    else:
        # Yes, this is the same as pcp_df above, but load_pcp_df messes it up
        sim_df = pd.read_csv(dataset_path).reset_index(drop=True).loc[valid_index, :]
        sim_df["branch_length"].to_csv(branch_lengths_path, index=False)

    plotters = plotter_class.heavy_light_plotter_pair_of_pcp_df(
        sim_data["selection_crepe_path"],
        pcp_df,
        str(dataset_path),
        branch_lengths_path,
        sim_data["anarci_paths"],
        1e-9,
    )

    fig, results_df = plotter_class.sites_oe_plots_of_plotter_dict(plotters)
    fig.savefig(Path("./") / (_remove_extensions(yaml_path).stem + "_oe_plots.pdf"))


def process_yaml(
    yaml_path: str, validate: bool = False, use_original_branch_lengths=False
):
    """Process a YAML file containing simulation parameters. Puts files at default
    locations.

    Keys expected in the yaml file are described in the docstring for the `read_simulation_yaml` function.

    Args:
        yaml_path: Path to the YAML file.
    """
    print("Fitting branch lengths")
    fit_tree_branch_lengths_from_yaml(
        yaml_path,
    )
    print("Simulating dataset")
    simulate_dataset_from_yaml(
        yaml_path,
    )
    if validate:
        print("Validating simulation")
        data_path = _dataset_path_of_inputs("./", None, yaml_path, None)
        print(data_path)
        validate_simulation_from_yaml(
            yaml_path,
            data_path,
            refit_branch_lengths=(not use_original_branch_lengths),
            filter_valid=True,
        )
