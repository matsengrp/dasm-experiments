import time
from typing import List, Optional

import numpy as np
import torch

import ablang2

from tqdm import tqdm

from netam.sequences import AA_STR_SORTED
from dnsmex import perplexity


def time_and_print(func, name):
    start = time.perf_counter()
    result = func()
    print(f"{name} took {time.perf_counter() - start:.3f}s")
    return result


class AbLangWrapper:
    def __init__(
        self, model_name="ablang2-paired", aa_order=list(AA_STR_SORTED), device="cpu"
    ):
        """A wrapper for the Ablang2 model to handle specific amino acid order and
        likelihood calculation.

        Args:
            model_name (str): The pretrained Ablang2 model to use.
            aa_order (list): The desired amino acid order for indexing logits.
        """

        self.ablang_model = ablang2.pretrained(
            model_to_use=model_name, random_init=False, device=device
        )

        vocab_dict = self.ablang_model.tokenizer.aa_to_token
        self.aa_order_indices = [vocab_dict[aa] for aa in aa_order]

        assert aa_order == [
            key
            for value in self.aa_order_indices
            for key, v in vocab_dict.items()
            if v == value
        ], "The provided amino acid order does not match the tokenization."

        self.aa_order_pos_of_aa = {aa: i for i, aa in enumerate(aa_order)}

    def masked_logits(self, heavy_seq, light_seq, split=False, stepwise_masking=True):
        """Computes the stepwise masked logits for given heavy and light chain
        sequences.

        Args:
            heavy_seq (str): The heavy chain sequence.
            light_seq (str): The light chain sequence.
            split (bool): Whether to return the logits for the heavy and light chains separately.

        Returns:
            torch.Tensor: A 2D tensor of logits with rows corresponding to amino acids
                          and columns corresponding to sequence positions.
        """
        [logits] = self.ablang_model(
            [heavy_seq, light_seq], mode="likelihood", stepwise_masking=stepwise_masking
        )

        # Transpose so rows are amino acids and columns are sequence positions
        logits = logits.T

        # Remove start and end tokens
        logits = logits[:, 1:-2]

        # Subset and reorder rows according to the amino acid order
        logits = logits[self.aa_order_indices, :]

        if split:
            heavy_logits = logits[:, : len(heavy_seq)]
            light_logits = logits[:, len(heavy_seq) :]

            return torch.Tensor(heavy_logits), torch.Tensor(light_logits)

        # else:
        return torch.Tensor(logits)

    def pseudo_perplexity(self, heavy_light_pairs):
        """Calculate pseudo-perplexity for a list of heavy-light pairs."""

        # https://github.com/oxpig/AbLang2/blob/main/notebooks/pretrained_module.ipynb
        results = time_and_print(
            lambda: self.ablang_model(heavy_light_pairs, mode="pseudo_log_likelihood"),
            "AbLang2 pseudo-perplexity",
        )
        return np.exp(-results)

    def per_variant_pseudo_perplexity(self, heavy_seq, light_seq):
        """Computes the pseudo-perplexity for each possible single amino acid variant.

        Args:
            heavy_seq (str): The heavy chain sequence.
            light_seq (str): The light chain sequence.

        Returns:
            tuple: Two arrays of shape (n_aa, seq_len) containing pseudo-perplexity values
                for heavy and light chain variants respectively.
                In each array, entry [i,j] is the pseudo-perplexity of the sequence with
                amino acid i at position j and wildtype amino acids at all other positions.
        """
        heavy_logits, light_logits = self.masked_logits(
            heavy_seq, light_seq, split=True
        )

        return (
            time_and_print(
                lambda: perplexity.per_variant_pseudo_perplexity(
                    heavy_logits, heavy_seq
                ),
                "AbLang2 Heavy chain perplexity",
            ),
            time_and_print(
                lambda: perplexity.per_variant_pseudo_perplexity(
                    light_logits, light_seq
                ),
                "AbLang2 Light chain perplexity",
            ),
        )

    def csp_perplexity_single(self, parent_aa_seq, child_aa_seq, logits):
        """Calculate the CSP perplexity for a single parent and child sequence.

        CSP means "conditional substution probability" and means that we only compute
        the perplexity for the sites where the parent and child sequences differ, and we
        remove the wildtype amino acid from consideration.
        """
        subs_prob_list = []

        for parent_aa, child_aa, site_logits in zip(
            parent_aa_seq, child_aa_seq, logits
        ):
            if parent_aa != child_aa:
                parent_aa_idx = self.aa_order_pos_of_aa[parent_aa]
                child_aa_idx = self.aa_order_pos_of_aa[child_aa]
                site_probs = np.exp(site_logits)
                site_probs[parent_aa_idx] = 0.0
                site_probs /= site_probs.sum()
                subs_prob_list.append(site_probs[child_aa_idx])

        return np.exp(-np.log(subs_prob_list).mean())

    def csp_perplexity_heavy(
        self, heavy_parent_aa_seqs, heavy_child_aa_seqs, stepwise_masking=True
    ):
        """Calculate the CSP perplexity for a list of heavy sequences."""

        heavy_light_pairs = [(heavy, "") for heavy in heavy_parent_aa_seqs]

        logitss = self.ablang_model(
            heavy_light_pairs, mode="likelihood", stepwise_masking=stepwise_masking
        )

        # Trim start and end tokens and subset/reorder amino acids.
        logitss = [logits[1:-2, self.aa_order_indices] for logits in logitss]

        perplexities = []

        for logits, parent_aa_seq, child_aa_seq in zip(
            logitss, heavy_parent_aa_seqs, heavy_child_aa_seqs
        ):
            perplexities.append(
                self.csp_perplexity_single(parent_aa_seq, child_aa_seq, logits)
            )

        return np.array(perplexities)

    def seqcoding(
        self, paired_seq_list: List[List[str]], batch_size: int = 100
    ) -> torch.Tensor:
        """Run seqcoding on paired heavy/light chain sequences and aggregate results
        into a tensor.

        Args:
            paired_seq_list: List of [heavy, light] sequence pairs.
            batch_size: Number of sequences to process in each batch.

        Returns:
            Tensor of shape (batch_size, seq_len, model_dim) containing the embeddings.
        """

        seq_coding_batches = []

        for i in tqdm(
            range(0, len(paired_seq_list), batch_size), desc="Running seqcoding"
        ):
            batch = paired_seq_list[i : i + batch_size]
            seq_coding = self.ablang_model(batch, mode="seqcoding")
            seq_coding = [torch.tensor(arr) for arr in seq_coding]
            seq_coding_batches.append(torch.stack(seq_coding))

        return torch.cat(seq_coding_batches)  # Concatenate all batches

    def rescoding(
        self,
        paired_seq_list: List[List[str]],
        max_seq_len: Optional[int] = None,
        stepwise_masking: bool = False,
    ) -> torch.Tensor:
        """Run rescoding on paired heavy/light chain sequences and aggregate results
        into a tensor.

        Args:
            paired_seq_list: List of [heavy, light] sequence pairs.
            max_seq_len: Optional desired output sequence length. If None, uses maximum
                combined sequence length.
            stepwise_masking: Whether to use stepwise masking in ablang model.

        Returns:
            Tensor of shape (batch_size, seq_len, model_dim) containing the embeddings.

        Raises:
            ValueError: If seq_len is too small for the sequences.
        """
        # Run model on all sequence pairs
        embeddings_list = [
            self.ablang_model(pair, mode="rescoding", stepwise_masking=stepwise_masking)
            for pair in paired_seq_list
        ]

        # Trim embeddings by removing token columns (<, >, |)
        trimmed_embeddings = []
        for emb, (heavy, light) in zip(embeddings_list, paired_seq_list):
            # Convert to numpy and squeeze out batch dimension since we're processing one at a time
            emb = np.array(emb).squeeze(0)

            # Skip the start token, take heavy sequence, skip end and separator tokens,
            # take light sequence, skip end token
            heavy_seq = emb[1 : len(heavy) + 1]
            light_seq = emb[len(heavy) + 3 : len(heavy) + len(light) + 3]

            # Concatenate heavy and light sequences
            trimmed_emb = np.concatenate([heavy_seq, light_seq])
            trimmed_embeddings.append(trimmed_emb)

        # Validate trimmed lengths match original sequence lengths
        for emb, (heavy, light) in zip(trimmed_embeddings, paired_seq_list):
            expected_len = len(heavy) + len(light)
            assert emb.shape[0] == expected_len, (
                f"Trimmed embedding length {emb.shape[0]} does not match "
                f"combined sequence length {expected_len}"
            )

        # Get maximum sequence length if not specified
        max_seq_len = max(emb.shape[0] for emb in trimmed_embeddings)
        if max_seq_len is None:
            max_seq_len = max_seq_len
        elif max_seq_len < max_seq_len:
            raise ValueError(
                f"Specified seq_len {max_seq_len} is too small for maximum "
                f"sequence length {max_seq_len}"
            )

        # Pad sequences to desired length
        padded_embeddings = []
        model_dim = trimmed_embeddings[0].shape[1]

        for emb in trimmed_embeddings:
            padding_len = max_seq_len - emb.shape[0]
            padded_emb = np.pad(
                emb,
                pad_width=((0, padding_len), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            padded_embeddings.append(padded_emb)

        # Stack into final tensor
        return torch.tensor(np.stack(padded_embeddings))
