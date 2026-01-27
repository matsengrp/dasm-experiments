"""Compute true pseudo-perplexity for Koenig variants.

This script computes pseudo-perplexity by masking each position within each
variant's own sequence context, rather than using the wildtype context for all
variants (the masked-marginals approximation).

Usage:
    python scripts/koenig_true_ppl.py
    python scripts/koenig_true_ppl.py --validate  # Include validation comparison
    python scripts/koenig_true_ppl.py --dry-run   # Test with 5 variants
"""

import argparse
import os

import pandas as pd

from dnsmex.koenig_helper import KoenigDataset
from dnsmex.local import localify
from dnsmex.cached_model_wrappers import (
    get_cached_esm_wrapper,
    get_cached_ablang_wrapper,
)


def prepare_chain_data(chain_df, chain_name):
    """Extract variant pairs and metadata for a single chain."""
    variants = list(zip(chain_df["heavy"], chain_df["light"]))
    metadata = chain_df[["site", "aa", "difference"]].copy()
    metadata["chain"] = chain_name
    metadata["wt_aa"] = metadata["difference"].str[0]
    return variants, metadata


def score_variants(variants, chain_name, esm_wrapper, ablang_wrapper):
    """Score variants with both ESM and AbLang models."""
    group = f"koenig_{chain_name}_true_ppl"
    esm_scores = esm_wrapper.evaluate_antibodies(variants, antibody_group=group)
    ablang_scores = ablang_wrapper.evaluate_antibodies(variants, antibody_group=group)
    return esm_scores, ablang_scores


def validate_approach(dataset, n_samples=10):
    """Compare true PPL vs masked-marginals for sanity check."""
    from dnsmex.esm_wrapper import esm2_wrapper_of_size
    from dnsmex import perplexity

    print("\n=== Validation: Comparing true PPL vs masked-marginals ===\n")

    esm = esm2_wrapper_of_size("650M")

    # Get WT masked logits (current approach)
    wt_heavy_logits = esm.masked_logits(dataset.heavy_consensus)

    # Sample a few variants
    sample_df = dataset.heavy_df.sample(
        n=min(n_samples, len(dataset.heavy_df)), random_state=42
    )

    print(f"{'Mutation':<12} {'Masked-Marginals':>18} {'True PPL':>12} {'Diff':>10}")
    print("-" * 55)

    for _, row in sample_df.iterrows():
        variant_seq = row["heavy"]

        # Masked-marginals PPL (current approach)
        mm_ppl = perplexity.sequence_pseudo_perplexity(wt_heavy_logits, variant_seq)

        # True PPL (new approach)
        true_ppl = esm.pseudo_perplexity(variant_seq)

        diff = abs(mm_ppl - true_ppl)
        print(
            f"{row['difference']:<12} {mm_ppl:>18.4f} {true_ppl:>12.4f} {diff:>10.4f}"
        )

    print("\nExpected: Small but measurable differences (~0.01-0.1)")
    print("At mutated position, logits are identical (both methods mask that position)")
    print(
        "At other positions, true PPL uses variant context while MM uses WT context\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute true pseudo-perplexity for Koenig variants"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation comparing true PPL vs masked-marginals",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with only 5 variants per chain",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local GPU processing instead of remote",
    )
    args = parser.parse_args()

    # Load dataset
    print("Loading Koenig dataset...")
    dataset = KoenigDataset(
        localify("DATA_DIR/FLAb/data/expression/Koenig2017_g6_er.csv"),
        fitness_column="fitness",
        fitness_label="expression",
        log_transform=True,
    )

    # Prepare variant pairs
    heavy_variants, heavy_metadata = prepare_chain_data(dataset.heavy_df, "heavy")
    light_variants, light_metadata = prepare_chain_data(dataset.light_df, "light")

    print(f"Found {len(heavy_variants)} heavy chain variants")
    print(f"Found {len(light_variants)} light chain variants")

    # Dry-run mode: use only 5 variants per chain
    if args.dry_run:
        heavy_variants = heavy_variants[:5]
        light_variants = light_variants[:5]
        heavy_metadata = heavy_metadata.head(5)
        light_metadata = light_metadata.head(5)
        print(
            f"\nDRY RUN: Testing with {len(heavy_variants)} heavy + {len(light_variants)} light variants"
        )

    # Run validation if requested
    if args.validate:
        validate_approach(dataset)

    # Get cached model wrappers
    use_remote = not args.local
    esm_wrapper = get_cached_esm_wrapper("650M", use_remote=use_remote)
    ablang_wrapper = get_cached_ablang_wrapper(use_remote=use_remote)

    # Score variants
    print("\n=== Scoring heavy chain variants ===")
    heavy_esm_scores, heavy_ablang_scores = score_variants(
        heavy_variants, "heavy", esm_wrapper, ablang_wrapper
    )

    print("\n=== Scoring light chain variants ===")
    light_esm_scores, light_ablang_scores = score_variants(
        light_variants, "light", esm_wrapper, ablang_wrapper
    )

    # Combine results
    heavy_metadata = heavy_metadata.copy()
    heavy_metadata["esm_true_ppl"] = heavy_esm_scores
    heavy_metadata["ablang_true_ppl"] = heavy_ablang_scores

    light_metadata = light_metadata.copy()
    light_metadata["esm_true_ppl"] = light_esm_scores
    light_metadata["ablang_true_ppl"] = light_ablang_scores

    results = pd.concat([heavy_metadata, light_metadata], ignore_index=True)

    # Save results
    output_dir = "notebooks/dasm_paper/_ignore"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "koenig_true_ppl.csv")
    results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print(f"Total rows: {len(results)}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Heavy chain: {len(heavy_metadata)} variants")
    print(f"Light chain: {len(light_metadata)} variants")
    print(f"\nColumns: {list(results.columns)}")
    print(f"\nSample output:")
    print(results.head(10).to_string())


if __name__ == "__main__":
    main()
