"""Remote ProGen2 processing for antibody sequences.

ProGen2 requires a separate environment and needs to be run in the ProGen2 repo
directory. This module handles remote execution on ermine with proper environment
isolation.
"""

import os
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .local import get_remote_config


class RemoteProGenWrapper:
    """Wrapper for remote ProGen2 model evaluation."""

    def __init__(self, model_version="progen2-small"):
        """Initialize ProGen wrapper.

        Args:
            model_version: ProGen2 model version (small, medium, large, xlarge)
        """
        self.model_version = model_version
        self.remote_config = get_remote_config()
        self.remote_host = self.remote_config["host"]
        self.remote_progen_dir = (
            "/home/matsen/progen2"  # ProGen2 repo on ermine (created by setup script)
        )
        self.remote_venv = "/home/matsen/progen2/.venv/bin/activate"  # ProGen2 venv (created by setup script)

    def evaluate_antibodies(self, sequences, antibody_group=None):
        """Evaluate antibody sequences using ProGen2.

        Args:
            sequences: List of (heavy, light) sequence tuples
            antibody_group: Optional group name for caching

        Returns:
            List of average perplexity scores (negated for consistency)
        """
        if not sequences:
            return []

        # Create temporary input file
        input_data = []
        for i, (heavy, light) in enumerate(sequences):
            input_data.append(
                {"sequence_id": f"seq_{i}", "heavy": heavy, "light": light}
            )

        input_df = pd.DataFrame(input_data)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            input_df.to_csv(f.name, index=False)
            temp_input_path = f.name

        try:
            # Copy input file to remote
            remote_input_path = f"{self.remote_progen_dir}/temp_input_{os.getpid()}.csv"
            subprocess.run(
                ["scp", temp_input_path, f"{self.remote_host}:{remote_input_path}"],
                check=True,
                capture_output=True,
            )

            # Run ProGen2 scoring on remote with smart device selection
            remote_output_dir = f"{self.remote_progen_dir}/temp_output_{os.getpid()}"
            remote_command = (
                f"cd {self.remote_progen_dir} && "
                f"source {self.remote_venv} && "
                f"DEVICE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader | "
                f"awk 'NR==1{{min=$1; idx=0}} $1<min{{min=$1; idx=NR-1}} END{{print \"cuda:\" idx}}') && "
                f'echo "Using device: $DEVICE" && '
                f"python flab_progen.py "
                f"--input_file {remote_input_path} "
                f"--model {self.model_version} "
                f"--device $DEVICE "
                f"--output_dir {remote_output_dir}"
            )

            try:
                result = subprocess.run(
                    ["ssh", self.remote_host, remote_command],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Log output for debugging
                if result.stdout:
                    print(f"ProGen2 stdout: {result.stdout}")
                if result.stderr:
                    print(f"ProGen2 stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"ProGen2 command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                raise

            # Copy results back
            base_name = os.path.splitext(os.path.basename(remote_input_path))[0]
            remote_output_file = (
                f"{remote_output_dir}/{base_name}.{self.model_version}.csv"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                temp_output_path = f.name

            subprocess.run(
                ["scp", f"{self.remote_host}:{remote_output_file}", temp_output_path],
                check=True,
                capture_output=True,
            )

            # Read results
            output_df = pd.read_csv(temp_output_path)

            # Extract average perplexity scores (negate for consistency with other models)
            scores = [-score for score in output_df["average_perplexity"].tolist()]

            # Cleanup remote files
            cleanup_command = f"rm -f {remote_input_path} && rm -rf {remote_output_dir}"
            subprocess.run(
                ["ssh", self.remote_host, cleanup_command],
                check=True,
                capture_output=True,
            )

            return scores

        finally:
            # Cleanup local temporary files
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if "temp_output_path" in locals() and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    def evaluate_antibodies_with_timing(
        self, sequences: List[Tuple[str, str]], antibody_group: str = "timing_benchmark"
    ) -> Dict[str, Any]:
        """Evaluate sequences with server-side timing - bypasses caching.

        Args:
            sequences: List of (heavy, light) sequence pairs
            antibody_group: Antibody group identifier (for timing isolation)

        Returns:
            Dictionary with {"scores": [...], "wall_time_seconds": X.X}
        """
        print(f"🚀 Running ProGen2-{self.model_version} with timing on remote GPU...")

        if not sequences:
            return {"scores": [], "wall_time_seconds": 0.0}

        # Create temporary input file
        input_data = []
        for i, (heavy, light) in enumerate(sequences):
            input_data.append(
                {"sequence_id": f"seq_{i}", "heavy": heavy, "light": light}
            )

        input_df = pd.DataFrame(input_data)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            input_df.to_csv(f.name, index=False)
            temp_input_path = f.name

        try:
            # Copy input file to remote
            remote_input_path = f"{self.remote_progen_dir}/temp_input_{os.getpid()}.csv"
            subprocess.run(
                ["scp", temp_input_path, f"{self.remote_host}:{remote_input_path}"],
                check=True,
                capture_output=True,
            )

            # Run ProGen2 scoring on remote with timing
            remote_output_dir = f"{self.remote_progen_dir}/temp_output_{os.getpid()}"
            remote_command = (
                f"cd {self.remote_progen_dir} && "
                f"source {self.remote_venv} && "
                f"DEVICE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader | "
                f"awk 'NR==1{{min=$1; idx=0}} $1<min{{min=$1; idx=NR-1}} END{{print \"cuda:\" idx}}') && "
                f'echo "Using device: $DEVICE" && '
                f'echo "Starting timed ProGen2 evaluation..." && '
                f"start_time=$(date +%s.%N) && "
                f"python flab_progen.py "
                f"--input_file {remote_input_path} "
                f"--model {self.model_version} "
                f"--device $DEVICE "
                f"--output_dir {remote_output_dir} && "
                f"end_time=$(date +%s.%N) && "
                f'elapsed=$(echo "$end_time - $start_time" | bc) && '
                f'echo "ProGen2 timing: $elapsed seconds"'
            )

            try:
                result = subprocess.run(
                    ["ssh", self.remote_host, remote_command],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Extract timing from output
                elapsed_time = 0.0
                if result.stdout:
                    print(f"ProGen2 stdout: {result.stdout}")
                    # Look for timing line
                    for line in result.stdout.split("\n"):
                        if "ProGen2 timing:" in line:
                            try:
                                elapsed_time = float(
                                    line.split("ProGen2 timing:")[1]
                                    .split("seconds")[0]
                                    .strip()
                                )
                            except:
                                print("Warning: Could not parse timing from output")
                if result.stderr:
                    print(f"ProGen2 stderr: {result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"ProGen2 command failed with exit code {e.returncode}")
                print(f"Command: {e.cmd}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                raise

            # Copy results back
            base_name = os.path.splitext(os.path.basename(remote_input_path))[0]
            remote_output_file = (
                f"{remote_output_dir}/{base_name}.{self.model_version}.csv"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                temp_output_path = f.name

            subprocess.run(
                ["scp", f"{self.remote_host}:{remote_output_file}", temp_output_path],
                check=True,
                capture_output=True,
            )

            # Read results
            output_df = pd.read_csv(temp_output_path)

            # Extract average perplexity scores (negate for consistency with other models)
            scores = [-score for score in output_df["average_perplexity"].tolist()]

            # Cleanup remote files
            cleanup_command = f"rm -f {remote_input_path} && rm -rf {remote_output_dir}"
            subprocess.run(
                ["ssh", self.remote_host, cleanup_command],
                check=True,
                capture_output=True,
            )

            return {"scores": scores, "wall_time_seconds": elapsed_time}

        finally:
            # Cleanup local temporary files
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if "temp_output_path" in locals() and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)


def get_cached_progen_wrapper(model_version="progen2-small"):
    """Get a cached ProGen wrapper instance with persistent caching.

    Args:
        model_version: ProGen2 model version

    Returns:
        Cached RemoteProGenWrapper instance
    """
    from .persistent_cache import cached_model_wrapper, get_global_cache

    print(f"🚀 Using remote ProGen2-{model_version} processing on GPU")
    CachedProGenWrapper = cached_model_wrapper(
        RemoteProGenWrapper, f"ProGen2-{model_version}-Remote", get_global_cache()
    )
    return CachedProGenWrapper(model_version)
