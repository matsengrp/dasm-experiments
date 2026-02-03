import torch

from netam.sequences import AA_STR_SORTED


def sequence_pseudo_perplexity(logits, seq):
    """Calculate pseudo-perplexity for a sequence from pre-computed masked logits.

    Computes: exp(-mean_i(log p(seq_i | context_i)))

    IMPORTANT: The logits determine what "context" means:
    - If logits were computed by masking positions in `seq` itself, this gives
      true pseudo-perplexity.
    - If logits were computed by masking positions in a *different* sequence
      (e.g., wild-type), this gives masked-marginals pseudo-perplexity.

    Args:
        logits (torch.Tensor): Tensor of shape (n_aa, seq_len) containing masked logits.
            These logits define p(a | context) for each position.
        seq (str): The sequence to calculate pseudo-perplexity for.

    Returns:
        float: Pseudo-perplexity value for the sequence.
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

    For masked-marginals pseudo-perplexity (used in Figure 1c):
        1. Call masked_logits(wt_seq) to get logits from WT context
        2. Call this function with those logits and wt_seq
        3. Result: each variant scored using WT-context probabilities

    For true pseudo-perplexity (used in Table 1):
        Call pseudo_perplexity(variant_seq) for each variant separately,
        which recomputes logits in each variant's own context.

    Args:
        logits (torch.Tensor): Tensor of shape (n_aa, seq_len) containing masked logits.
            For masked-marginals, these should be computed from the WT sequence.
        seq (str): The wild-type sequence.

    Returns:
        numpy.ndarray: Array of shape (n_aa, seq_len) containing pseudo-perplexity values
            where entry [i,j] is the pseudo-perplexity of the sequence with amino acid i
            at position j and wildtype amino acids at all other positions.
    """
    perplexities = torch.zeros_like(logits)
    for var_aa_idx in range(logits.shape[0]):
        for pos in range(len(seq)):
            var_seq = list(seq)
            var_seq[pos] = AA_STR_SORTED[var_aa_idx]
            perplexities[var_aa_idx, pos] = sequence_pseudo_perplexity(logits, var_seq)
    return perplexities.numpy()
