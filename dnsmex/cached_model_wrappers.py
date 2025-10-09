"""Cached model wrappers for antibody sequence evaluation.

Implements the FLAb paper approach: separate VH/VL evaluation + averaging.
Results are cached to disk in _ignore directory to persist between notebook restarts.
"""

from typing import List, Tuple
from tqdm import tqdm

from dnsmex.esm_wrapper import esm2_wrapper_of_size
from dnsmex.ablang_wrapper import AbLangWrapper
from dnsmex.persistent_cache import cached_model_wrapper, get_global_cache


class ESMSequenceEvaluator:
    """ESM model evaluator with VH+VL averaging following FLAb paper approach."""

    def __init__(self, model_size: str = "650M"):
        self.model_size = model_size
        self._esm_wrapper = None

    def _get_esm_wrapper(self):
        """Lazy initialization of ESM wrapper."""
        if self._esm_wrapper is None:
            self._esm_wrapper = esm2_wrapper_of_size(self.model_size)
        return self._esm_wrapper

    def evaluate_antibodies(self, sequences: List[Tuple[str, str]]) -> List[float]:
        """Evaluate sequences using ESM model with VH+VL averaging.

        Following FLAb paper methodology (Chungyoun et al. 2024): "For each protein
        language model, we separately input the heavy and light sequence to return two
        perplexity scores, and we tabulate the average perplexity between the two
        sequences."
        """
        esm_wrapper = self._get_esm_wrapper()
        results = []

        for heavy, light in tqdm(sequences, desc=f"Computing ESM-{self.model_size}"):
            # Evaluate heavy and light chains separately
            heavy_score = esm_wrapper.pseudo_perplexity(heavy)
            light_score = esm_wrapper.pseudo_perplexity(light)

            # Average the perplexities (FLAb paper approach)
            avg_score = (heavy_score + light_score) / 2
            results.append(avg_score)

        return results


class AbLangSequenceEvaluator:
    """AbLang model evaluator with proper VH+VL pair handling."""

    def __init__(self):
        self._ablang_wrapper = None

    def _get_ablang_wrapper(self):
        """Lazy initialization of AbLang wrapper."""
        if self._ablang_wrapper is None:
            import torch

            # Set single thread for PyTorch to avoid slow parallel overhead
            torch.set_num_threads(1)
            print("Set PyTorch to single thread for AbLang")

            # AbLang has issues with MPS, stick to CPU for now
            self._ablang_wrapper = AbLangWrapper(device="cpu")
        return self._ablang_wrapper

    def evaluate_antibodies(self, sequences: List[Tuple[str, str]]) -> List[float]:
        """Evaluate sequences using AbLang model with smaller batches."""
        ablang_wrapper = self._get_ablang_wrapper()

        # AbLang expects list of [heavy, light] pairs
        heavy_light_pairs = [[heavy, light] for heavy, light in sequences]

        print(f"Computing AbLang2 for {len(sequences)} sequences...")

        # Process one by one to verify it's actually working and show real progress
        batch_size = 1  # Individual sequences
        all_results = []

        for i in range(0, len(heavy_light_pairs), batch_size):
            batch = heavy_light_pairs[i : i + batch_size]
            print(f"  Sequence {i+1}/{len(heavy_light_pairs)}: Computing...")

            batch_results = ablang_wrapper.pseudo_perplexity(batch)
            all_results.extend(batch_results)

            print(
                f"  Sequence {i+1}/{len(heavy_light_pairs)}: Done (score: {batch_results[0]:.3f})"
            )

        return all_results


# Create cached versions of the model wrappers
def get_cached_esm_wrapper(model_size: str = "650M", use_remote: bool = True):
    """Get cached ESM wrapper with specified model size.

    Args:
        model_size: ESM model size (e.g., "650M")
        use_remote: Whether to use remote GPU processing for faster inference
    """
    if use_remote:
        try:
            from dnsmex.remote_esm import RemoteESMEvaluator

            print(f"🚀 Using remote ESM-{model_size} processing on GPU")
            CachedESMWrapper = cached_model_wrapper(
                RemoteESMEvaluator, f"ESM-{model_size}-Remote", get_global_cache()
            )
            return CachedESMWrapper(model_size)
        except Exception as e:
            print(f"⚠️  Remote ESM processing failed ({e}), falling back to local")

    print(f"🖥️  Using local ESM-{model_size} processing")
    CachedESMWrapper = cached_model_wrapper(
        ESMSequenceEvaluator, f"ESM-{model_size}", get_global_cache()
    )
    return CachedESMWrapper(model_size)


def get_cached_ablang_wrapper(use_remote: bool = True):
    """Get cached AbLang wrapper.

    Args:
        use_remote: Whether to use remote GPU processing for faster inference
    """
    if use_remote:
        try:
            from dnsmex.remote_ablang import RemoteAbLangEvaluator

            print(f"🚀 Using remote AbLang2 processing on GPU")
            CachedAbLangWrapper = cached_model_wrapper(
                RemoteAbLangEvaluator, "AbLang2-Remote", get_global_cache()
            )
            return CachedAbLangWrapper()
        except Exception as e:
            print(f"⚠️  Failed to use remote AbLang: {e}")
            print("   Falling back to local processing...")

    # Local processing
    print("🏠 Using local AbLang2 processing (slower)")
    CachedAbLangWrapper = cached_model_wrapper(
        AbLangSequenceEvaluator, "AbLang2", get_global_cache()
    )
    return CachedAbLangWrapper()


def get_cached_progen_wrapper(model_version: str = "progen2-small"):
    """Get cached ProGen2 wrapper with specified model version.

    Args:
        model_version: ProGen2 model version (small, medium, large, xlarge)
    """
    try:
        from dnsmex.remote_progen import (
            get_cached_progen_wrapper as _get_cached_progen_wrapper,
        )

        return _get_cached_progen_wrapper(model_version)
    except Exception as e:
        print(f"⚠️  Failed to use remote ProGen2: {e}")
        print("   ProGen2 requires remote processing with separate environment")
        raise RuntimeError("ProGen2 processing requires remote environment setup")
