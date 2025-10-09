import fire
from dnsmex.simulation import (
    fit_tree_branch_lengths_from_yaml,
    simulate_dataset_from_yaml,
    validate_simulation_from_yaml,
    process_yaml,
)


def main():
    fire.Fire(
        {
            "prepare_clonal_families": fit_tree_branch_lengths_from_yaml,
            "simulate": simulate_dataset_from_yaml,
            "validate": validate_simulation_from_yaml,
            "process_simulation": process_yaml,
        }
    )
