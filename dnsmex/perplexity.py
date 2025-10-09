import torch

from netam.sequences import AA_STR_SORTED


def sequence_pseudo_perplexity(logits, seq):
    """Calculate pseudo-perplexity for a sequence from masked logits.

    Args:
        logits (torch.Tensor): Tensor of shape (n_aa, seq_len) containing masked logits
        seq (str): The sequence to calculate pseudo-perplexity for

    Returns:
        float: Pseudo-perplexity value for the sequence
    """
    log_probs = []
    for i, aa in enumerate(seq):
        if aa in AA_STR_SORTED:
            aa_idx = AA_STR_SORTED.index(aa)
            log_probs.append(
                torch.nn.functional.log_softmax(logits[:, i], dim=0)[aa_idx]
            )
    return torch.exp(-torch.mean(torch.stack(log_probs))).item()


def per_variant_pseudo_perplexity(logits, seq):
    """Calculate pseudo-perplexity for each possible single amino acid variant.

    Args:
        logits (torch.Tensor): Tensor of shape (n_aa, seq_len) containing masked logits
        seq (str): The wildtype sequence

    Returns:
        numpy.ndarray: Array of shape (n_aa, seq_len) containing pseudo-perplexity values
            where entry [i,j] is the pseudo-perplexity of the sequence with amino acid i
            at position j and wildtype amino acids at all other positions
    """
    perplexities = torch.zeros_like(logits)
    for var_aa_idx in range(logits.shape[0]):
        for pos in range(len(seq)):
            var_seq = list(seq)
            var_seq[pos] = AA_STR_SORTED[var_aa_idx]
            perplexities[var_aa_idx, pos] = sequence_pseudo_perplexity(logits, var_seq)
    return perplexities.numpy()
