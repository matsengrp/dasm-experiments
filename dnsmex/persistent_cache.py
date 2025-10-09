"""Persistent caching decorator for expensive model computations.

Saves results to disk in _ignore directory to persist between notebook restarts.
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple, Optional, Dict
import pandas as pd
import time


class PersistentCache:
    """Persistent file-based cache for model results."""

    def __init__(self, cache_dir: str = "_ignore/persistent_cache"):
        # Find project root by looking for setup.py or pyproject.toml
        current_dir = Path.cwd()
        project_root = current_dir

        # Walk up the directory tree to find project root
        while project_root != project_root.parent:
            if (project_root / "setup.py").exists() or (
                project_root / "pyproject.toml"
            ).exists():
                break
            project_root = project_root.parent

        # Use absolute path from project root
        if cache_dir.startswith("_ignore/"):
            self.cache_dir = project_root / cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self, antibody_group: str, model_name: str, sequences: List[Tuple[str, str]]
    ) -> str:
        """Generate cache key from antibody group, model name, and sequences."""
        # Hash the sequences to create a unique identifier
        seq_str = "".join(f"{h}|{l}" for h, l in sequences)
        combined = f"{antibody_group}_{model_name}_{seq_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self, antibody_group: str, model_name: str, sequences: List[Tuple[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Get cached results if available.

        Returns dict with 'results' and 'metadata'.
        """
        cache_key = self._get_cache_key(antibody_group, model_name, sequences)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                # Handle old cache format (just results list)
                if isinstance(cache_data, list):
                    cache_data = {"results": cache_data, "metadata": {}}

                print(
                    f"✓ Loaded {len(cache_data['results'])} cached {model_name} results for {antibody_group}"
                )
                return cache_data
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
                return None
        return None

    def set(
        self,
        antibody_group: str,
        model_name: str,
        sequences: List[Tuple[str, str]],
        results: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save results to cache with optional metadata (e.g., timing info)."""
        cache_key = self._get_cache_key(antibody_group, model_name, sequences)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {"results": results, "metadata": metadata or {}}

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cached {len(results)} {model_name} results for {antibody_group}")
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"✓ Cleared cache directory: {self.cache_dir}")

    def info(self):
        """Print cache information."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Files: {len(cache_files)}")
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB")


def cached_model_wrapper(model_class, model_name: str, cache: PersistentCache = None):
    """Decorator to add persistent caching to any model class.

    Args:
        model_class: The model class to wrap
        model_name: Name for cache identification
        cache: Optional cache instance (creates default if None)

    Returns:
        Wrapped class with caching capabilities
    """
    if cache is None:
        cache = PersistentCache()

    class CachedModelWrapper:
        def __init__(self, *args, **kwargs):
            self.model = model_class(*args, **kwargs)
            self.model_name = model_name
            self.cache = cache

        def evaluate_antibodies(
            self, sequences: List[Tuple[str, str]], antibody_group: str = "default"
        ) -> List[float]:
            """Evaluate antibody sequences with persistent caching."""
            # Try to get from cache
            cached_data = self.cache.get(antibody_group, self.model_name, sequences)
            if cached_data is not None:
                # Extract results from cache data structure
                return (
                    cached_data.get("results", cached_data)
                    if isinstance(cached_data, dict)
                    else cached_data
                )

            # Compute results using the wrapped model
            print(
                f"⏳ Computing {self.model_name} for {len(sequences)} sequences in {antibody_group}..."
            )

            # Time the computation
            start_time = time.time()

            # Call the appropriate evaluation method on the wrapped model
            if hasattr(self.model, "evaluate_antibodies"):
                results = self.model.evaluate_antibodies(sequences)
            elif hasattr(self.model, "evaluate_sequences"):
                results = self.model.evaluate_sequences(sequences)
            else:
                raise AttributeError(
                    f"Model {model_class} must have evaluate_antibodies or evaluate_sequences method"
                )

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            print(f"   ✅ Computed in {elapsed_time:.1f}s")

            # Cache results with timing metadata
            metadata = {
                "compute_time": elapsed_time,
                "num_sequences": len(sequences),
                "timestamp": time.time(),
            }
            self.cache.set(
                antibody_group, self.model_name, sequences, results, metadata
            )

            return results

        def __getattr__(self, name):
            """Delegate other attributes to the wrapped model."""
            return getattr(self.model, name)

    return CachedModelWrapper


# Global cache instance
_global_cache = PersistentCache()


def get_global_cache() -> PersistentCache:
    """Get the global cache instance."""
    return _global_cache
