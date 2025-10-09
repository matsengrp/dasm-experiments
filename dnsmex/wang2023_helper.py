from pathlib import Path
import pandas as pd
from Bio import SeqIO


def fasta_to_dict(fasta_path):
    """Convert a FASTA file to a dictionary mapping sequence ID to sequence.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        Dictionary mapping sequence ID to sequence string.
    """
    sequences = {}
    with open(fasta_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Convert sequence ID to integer since files use integer IDs
            seq_id = int(record.id)
            sequences[seq_id] = str(record.seq)
    return sequences


def process_wang_specificity(data_dir):
    """Process Wang specificity data to create a combined dataframe.

    Args:
        data_dir: Path to the data directory containing Annotations and FASTA subdirectories.

    Returns:
        DataFrame containing annotation data with sequences added.
    """
    data_dir = Path(data_dir)

    # Load sequence dictionaries
    heavy_seqs = fasta_to_dict(data_dir / "FASTA" / "combined_distinct_heavy.fa")
    light_seqs = fasta_to_dict(data_dir / "FASTA" / "combined_distinct_light.fa")

    # Load annotations
    anno_df = pd.read_csv(
        data_dir / "Annotations" / "specificity.anno",
        sep="\t",
        dtype={"heavy_id": int, "light_id": int},
    )

    anno_df["heavy"] = anno_df["heavy_id"].map(heavy_seqs)
    anno_df["light"] = anno_df["light_id"].map(light_seqs)
    anno_df["binds"] = anno_df["label"] == "S+"
    anno_df.drop(columns=["heavy_id", "light_id"], inplace=True)
    anno_df = anno_df[
        ["heavy", "light"] + list(anno_df.columns.difference(["heavy", "light"]))
    ]

    return anno_df


def filtered_wang_specificity(data_dir):
    """Process Wang specificity data and filter based on sequence length.

    Returns the filtered DataFrame and the maximum sequence length.
    """
    wang_df = process_wang_specificity(data_dir)
    wang_df["heavy_len"] = wang_df["heavy"].apply(len)
    wang_df["light_len"] = wang_df["light"].apply(len)
    wang_df["total_len"] = wang_df["heavy_len"] + wang_df["light_len"]
    print(
        f"Discarding rows with total_len > 300. There is {wang_df[wang_df['total_len'] > 300].shape[0]}"
    )
    wang_df = wang_df[wang_df["total_len"] <= 300]
    max_seq_len = wang_df["total_len"].max()
    return wang_df, max_seq_len
