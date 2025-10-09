"""Remote DASM processing on GPU servers.

This module provides functionality to run DASM model inference on remote GPU servers
while maintaining the same interface as local processing.
"""

import os
import json
import subprocess
import tempfile
from typing import List, Tuple, Dict, Any
from pathlib import Path

from .local import get_remote_config, localify


class RemoteDASMEvaluator:
    """DASM model evaluator that runs on remote GPU servers."""

    def __init__(
        self,
        model_path: str = "DASM_TRAINED_MODELS_DIR/dasm_1m-v1tangCC50k+v2jaffePairedCC+v1vanwinklelightTrainCC100k-joint",
    ):
        self.model_path = model_path
        self.remote_config = get_remote_config()

    def evaluate_antibodies(
        self, sequences: List[Tuple[str, str]], antibody_group: str = "default"
    ) -> List[float]:
        """Evaluate sequences using DASM model on remote GPU server.

        Args:
            sequences: List of (heavy, light) sequence pairs
            antibody_group: Antibody group identifier (for compatibility)

        Returns:
            List of DASM scores (processed appropriately for comparison)
        """
        print(f"🚀 Running DASM on {self.remote_config['host']} GPU...")
        return self._run_remote_evaluation(sequences, include_timing=False)

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
        print(f"🚀 Running DASM with timing on {self.remote_config['host']} GPU...")
        return self._run_remote_evaluation(sequences, include_timing=True)

    def _run_remote_evaluation(
        self, sequences: List[Tuple[str, str]], include_timing: bool = False
    ):
        """Run DASM evaluation on remote server."""

        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as input_file:
            json.dump(
                {
                    "sequences": sequences,
                    "model_path": self.model_path,
                    "include_timing": include_timing,
                },
                input_file,
            )
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as output_file:
            output_path = output_file.name

        try:
            # Copy input file to remote server
            remote_input = f"/tmp/dasm_input_{os.path.basename(input_path)}"
            remote_output = f"/tmp/dasm_output_{os.path.basename(output_path)}"

            # Upload input file
            subprocess.run(
                ["scp", input_path, f"{self.remote_config['host']}:{remote_input}"],
                check=True,
            )

            # Create and upload remote script
            remote_script_path = "/tmp/remote_dasm_script.py"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file:
                script_file.write(self._create_remote_script())
                local_script_path = script_file.name

            # Upload script to remote server
            subprocess.run(
                [
                    "scp",
                    local_script_path,
                    f"{self.remote_config['host']}:{remote_script_path}",
                ],
                check=True,
            )

            # Run DASM processing on remote server
            ssh_cmd = [
                "ssh",
                self.remote_config["host"],
                f'cd {self.remote_config["dir"]} && '
                f'source {self.remote_config["venv"]} && '
                f"python {remote_script_path} {remote_input} {remote_output}",
            ]

            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Remote DASM processing failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, ssh_cmd)
            print(f"✅ Remote DASM processing completed")

            # Clean up remote script
            subprocess.run(
                ["ssh", self.remote_config["host"], f"rm -f {remote_script_path}"],
                check=True,
            )

            # Clean up local script
            os.unlink(local_script_path)

            # Download output file
            subprocess.run(
                ["scp", f"{self.remote_config['host']}:{remote_output}", output_path],
                check=True,
            )

            # Load results
            with open(output_path, "r") as f:
                results = json.load(f)

            # Clean up remote files
            subprocess.run(
                [
                    "ssh",
                    self.remote_config["host"],
                    f"rm -f {remote_input} {remote_output}",
                ],
                check=True,
            )

            if include_timing:
                return results  # {"scores": [...], "wall_time_seconds": X.X}
            else:
                return results["scores"]  # Just the scores for compatibility

        finally:
            # Clean up local temporary files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

    def _create_remote_script(self) -> str:
        """Create the Python script to run on the remote server."""
        return """
import sys
import json
import time
from netam.framework import load_crepe
from netam.common import pick_device
from dnsmex.local import localify

# Read input
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

sequences = data['sequences']
model_path = data['model_path']
include_timing = data.get('include_timing', False)

print(f"Loading DASM model from {model_path}...")
device = pick_device()
print(f"Using device: {device}")

# Load DASM model
crepe = load_crepe(localify(model_path), device=device)

if include_timing:
    print(f"Starting timed evaluation of {len(sequences)} sequences...")
    start_time = time.time()

# Process sequences - DASM expects list of [heavy, light] pairs
heavy_light_pairs = [[heavy, light] for heavy, light in sequences]
print(f"Processing {len(heavy_light_pairs)} sequence pairs...")

# Run DASM inference
dasm_results = crepe(heavy_light_pairs)

if include_timing:
    elapsed_time = time.time() - start_time
    print(f"✅ DASM processing complete in {elapsed_time:.2f}s")

# Convert results to serializable format
# DASM returns tensors, need to process them appropriately
scores = []
for i, (dasm_heavy, dasm_light) in enumerate(dasm_results):
    # Process DASM output similar to existing patterns
    # Average log-transformed heavy and light scores
    import torch
    log_dasm_heavy = torch.log(dasm_heavy).mean().item()
    log_dasm_light = torch.log(dasm_light).mean().item()
    avg_score = (log_dasm_heavy + log_dasm_light) / 2
    scores.append(avg_score)

print(f"Processed {len(scores)} DASM scores")

# Save results
result_data = {"scores": scores}
if include_timing:
    result_data["wall_time_seconds"] = elapsed_time

with open(sys.argv[2], 'w') as f:
    json.dump(result_data, f)

print("✅ Results saved")
"""


def get_remote_dasm_evaluator(
    model_path: str = "DASM_TRAINED_MODELS_DIR/dasm_1m-v1tangCC50k+v2jaffePairedCC+v1vanwinklelightTrainCC100k-joint",
):
    """Get remote DASM evaluator for specified model path."""
    return RemoteDASMEvaluator(model_path)
