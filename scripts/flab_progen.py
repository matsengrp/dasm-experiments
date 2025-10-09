"""
This hacky script is just here to get per-antibody perplexities under the
ProGen2 model on various datasets in the FLAb project.

The important bits are copied whole-cloth from the `progen_score` function in
https://github.com/Graylab/FLAb/blob/main/scripts/models.py
so thank you to them for writing and sharing that code.

I haven't integrated this work in an elegant way with the rest of the notebooks,
etc, because in any case this needs to be run in a separate environment, as 
the Python and torch versions are incompatible with the rest of the project.
I also could not run FLAb directly because the conda environment for that
requires channels that are no longer allowed at Fred Hutch.

So the steps are:
1. Get ProGen2 set up. I had to patch it using this fork:
    https://github.com/matsen/progen2/tree/patch-1
2. Build a venv according to their instructions.
3. Run this script in that venv in the root directory of the ProGen2 repo.
"""

import math
import os

import fire
import pandas as pd
import torch

from tqdm import tqdm

torch.set_num_threads(1)

from likelihood import create_model, create_tokenizer_custom, log_likelihood, print_time


def progen_score(df, model_version, device):
    device = torch.device(device)
    ckpt = f"checkpoints/{model_version}"

    with print_time("loading parameters"):
        model = create_model(ckpt=ckpt, fp16=True).to(device)

    with print_time("loading tokenizer"):
        tokenizer = create_tokenizer_custom(file="tokenizer.json")

    def ll(tokens, f=log_likelihood, reduction="mean"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                logits = model(target, labels=target).logits

                # shift
                logits = logits[:-1, ...]
                target = target[1:]

                # remove terminals
                bos_token, eos_token = 3, 4
                if target[-1] in [bos_token, eos_token]:
                    logits = logits[:-1, ...]
                    target = target[:-1]

                assert (target == bos_token).sum() == 0
                assert (target == eos_token).sum() == 0

                # remove unused logits
                first_token, last_token = 5, 29
                logits = logits[:, first_token : (last_token + 1)]
                target = target - first_token

                assert logits.shape[1] == (last_token - first_token + 1)

                return f(logits=logits, target=target, reduction=reduction).item()

    # score heavy sequences using progen
    perplexity_mean_list_h = []

    for seq in tqdm(df["heavy"], desc="heavy"):
        context = seq

        reverse = lambda s: s[::-1]

        ll_lr_mean = ll(tokens=context, reduction="mean")
        ll_rl_mean = ll(tokens=reverse(context), reduction="mean")

        ll_mean = 0.5 * (ll_lr_mean + ll_rl_mean)

        perplexity = math.exp(-ll_mean)

        perplexity_mean_list_h.append(perplexity)

    df["heavy_perplexity"] = perplexity_mean_list_h

    # score light sequences using progen
    perplexity_mean_list_l = []

    for seq in tqdm(df["light"], desc="light"):
        context = seq

        reverse = lambda s: s[::-1]

        ll_lr_mean = ll(tokens=context, reduction="mean")
        ll_rl_mean = ll(tokens=reverse(context), reduction="mean")

        ll_mean = 0.5 * (ll_lr_mean + ll_rl_mean)

        perplexity = math.exp(-ll_mean)

        perplexity_mean_list_l.append(perplexity)

    df["light_perplexity"] = perplexity_mean_list_l

    df["average_perplexity"] = (df["heavy_perplexity"] + df["light_perplexity"]) / 2

    return df


def progen_score_cli(
    input_file: str,
    model: str = "progen2-small",
    device: str = "cuda:0",
    dry_run: bool = False,
    output_dir: str = "output",
):
    """Score sequences using a Progen model.

    Args:
        input_file: Path to input CSV file.
        model: Model name to use for scoring. Use "all" to run all available models.
        device: Device to run model on (e.g. "cuda:0", "cpu").
        dry_run: If True, only process first row of data.
        output_dir: Directory to save output files.
    """
    df = pd.read_csv(input_file)

    if dry_run:
        df = df.head(1)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename without path or extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    if model == "all":
        models = ["progen2-small", "progen2-medium", "progen2-large", "progen2-xlarge"]
        for model_name in models:
            scored_df = progen_score(df, model_name, device)
            output_file = os.path.join(output_dir, f"{base_name}.{model_name}.csv")
            scored_df.to_csv(output_file, index=False)
            print(f"Saved scores to {output_file}")
    else:
        scored_df = progen_score(df, model, device)
        output_file = os.path.join(output_dir, f"{base_name}.{model}.csv")
        scored_df.to_csv(output_file, index=False)
        print(f"Saved scores to {output_file}")


if __name__ == "__main__":
    fire.Fire(progen_score_cli)
