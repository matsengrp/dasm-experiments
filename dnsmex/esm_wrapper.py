from netam.sequences import AA_STR_SORTED
import torch
import esm

from dnsmex import perplexity


class ESMWrapper:
    def __init__(
        self, model_name="esm2_t48_15B_UR50D", aa_order=list(AA_STR_SORTED), device=None
    ):
        """A wrapper for ESM2 models to handle specific amino acid order and likelihood
        calculation.

        Args:
            model_name (str): The pretrained ESM2 model to use.
            aa_order (list): The desired amino acid order for indexing logits.
        """
        # Get the appropriate model function from esm.pretrained
        model_fn = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = model_fn()
        self.model.eval()

        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        self.aa_order_indices = [self.alphabet.get_idx(aa) for aa in aa_order]
        self.batch_converter = self.alphabet.get_batch_converter()

    def logits(self, seq):
        """Computes logits for a given sequence.

        Args:
            seq (str): The amino acid sequence to analyze.

        Returns:
            torch.Tensor: A 2D tensor of logits with rows corresponding to amino acids
                         and columns corresponding to sequence positions.
        """
        _, _, batch_tokens = self.batch_converter([("protein1", seq)])
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens)
            token_logits = results["logits"].to("cpu")

            # Extract standard AA logits and remove special tokens
            logits = token_logits[0, 1 : len(seq) + 1, self.aa_order_indices]
            logits = logits.T.numpy()

        return logits

    def masked_logits(self, seq):
        """Gets masked logits by masking each position one at a time.

        This is the first step of masked-marginals pseudo-perplexity:
        compute p(a | seq_{\\i}) for all amino acids a at each position i.

        When called with a wild-type sequence, these logits can be reused
        to score any variant using the WT context (masked-marginals approach),
        rather than recomputing logits for each variant in its own context
        (true pseudo-perplexity).
        """
        _, _, batch_tokens = self.batch_converter([("protein1", seq)])
        batch_tokens = batch_tokens.to(self.device)

        all_token_logits = []
        for site in range(batch_tokens.size(1)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, site] = self.alphabet.mask_idx

            with torch.no_grad():
                batch_tokens_masked = batch_tokens_masked.to(self.device)
                logits = self.model(batch_tokens_masked)["logits"]
                all_token_logits.append(logits[:, site])

        token_logits = torch.cat(all_token_logits, dim=0)
        aa_logits = token_logits[:, self.aa_order_indices]

        # Remove special tokens and transpose
        aa_logits = aa_logits[1 : len(seq) + 1].T

        return aa_logits.cpu()

    def pseudo_perplexity(self, seq):
        """Calculate pseudo-perplexity for a sequence."""
        logits = self.masked_logits(seq)
        return perplexity.sequence_pseudo_perplexity(logits, seq)

    def per_variant_pseudo_perplexity(self, seq):
        """Calculate pseudo-perplexity for all point mutants of a sequence."""
        logits = self.masked_logits(seq)
        return perplexity.per_variant_pseudo_perplexity(logits, seq)


def esm2_wrapper_of_size(size, device=None):
    name_of_size = {
        "650M": "esm2_t33_650M_UR50D",
        "3B": "esm2_t36_3B_UR50D",
        "15B": "esm2_t48_15B_UR50D",
    }

    return ESMWrapper(name_of_size[size], device=device)
