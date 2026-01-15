"""Remote ESM processing on GPU servers.

This module provides functionality to run ESM model inference on remote GPU servers
while maintaining the same interface as local processing.
"""

import os
import json
import subprocess
import tempfile
from typing import List, Tuple, Dict, Any
from pathlib import Path

from .local import get_remote_config


class RemoteESMEvaluator:
    """ESM model evaluator that runs on remote GPU servers."""

    def __init__(self, model_size: str = "3B"):
        self.model_size = model_size
        self.remote_config = get_remote_config()

    def evaluate_antibodies(self, sequences: List[Tuple[str, str]]) -> List[float]:
        """Evaluate sequences using ESM model on remote GPU server.

        Args:
            sequences: List of (heavy, light) sequence pairs

        Returns:
            List of perplexity scores (averaged between heavy and light)
        """
        print(
            f"🚀 Running ESM-{self.model_size} on {self.remote_config['host']} GPU..."
        )

        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as input_file:
            json.dump(
                {"sequences": sequences, "model_size": self.model_size}, input_file
            )
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as output_file:
            output_path = output_file.name

        try:
            # Copy input file to remote server
            remote_input = f"/tmp/esm_input_{os.path.basename(input_path)}"
            remote_output = f"/tmp/esm_output_{os.path.basename(output_path)}"

            # Upload input file
            subprocess.run(
                ["scp", input_path, f"{self.remote_config['host']}:{remote_input}"],
                check=True,
            )

            # Create and upload remote script
            remote_script_path = "/tmp/remote_esm_script.py"
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

            # Run ESM processing on remote server
            ssh_cmd = [
                "ssh",
                self.remote_config["host"],
                f'cd {self.remote_config["dir"]} && '
                f'source {self.remote_config["venv"]} && '
                f"python {remote_script_path} {remote_input} {remote_output}",
            ]

            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Remote ESM processing failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, ssh_cmd)
            print(f"✅ Remote ESM processing completed")

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

            return results["scores"]

        finally:
            # Clean up local temporary files
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

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
        print(
            f"🚀 Running ESM-{self.model_size} with timing on {self.remote_config['host']} GPU..."
        )
        return self._run_remote_with_timing(sequences)

    def _run_remote_with_timing(
        self, sequences: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Run ESM evaluation with timing on remote server."""

        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as input_file:
            json.dump(
                {
                    "sequences": sequences,
                    "model_size": self.model_size,
                    "include_timing": True,
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
            remote_input = f"/tmp/esm_input_{os.path.basename(input_path)}"
            remote_output = f"/tmp/esm_output_{os.path.basename(output_path)}"

            # Upload input file
            subprocess.run(
                ["scp", input_path, f"{self.remote_config['host']}:{remote_input}"],
                check=True,
            )

            # Create and upload remote script with timing
            remote_script_path = "/tmp/remote_esm_timing_script.py"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as script_file:
                script_file.write(self._create_remote_timing_script())
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

            # Run ESM processing with timing on remote server
            ssh_cmd = [
                "ssh",
                self.remote_config["host"],
                f'cd {self.remote_config["dir"]} && '
                f'source {self.remote_config["venv"]} && '
                f"python {remote_script_path} {remote_input} {remote_output}",
            ]

            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Remote ESM timing processing failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, ssh_cmd)
            print(f"✅ Remote ESM timing processing completed")

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

            return results  # {"scores": [...], "wall_time_seconds": X.X}

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
from tqdm import tqdm
from dnsmex.esm_wrapper import esm2_wrapper_of_size
from netam.common import pick_device

# Read input
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

sequences = data['sequences']
model_size = data['model_size']

print(f"Loading ESM-{model_size} model...")
device = pick_device()
esm_wrapper = esm2_wrapper_of_size(model_size, device=device)

results = []
print(f"Processing {len(sequences)} sequence pairs...")

for i, (heavy, light) in enumerate(tqdm(sequences, desc=f"Computing ESM-{model_size}")):
    # Evaluate heavy and light chains separately
    heavy_score = esm_wrapper.pseudo_perplexity(heavy)
    light_score = esm_wrapper.pseudo_perplexity(light)
    
    # Average the perplexities (FLAb paper approach)
    avg_score = (heavy_score + light_score) / 2
    results.append(avg_score)
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(sequences)} pairs")

print(f"✅ ESM-{model_size} processing complete")

# Save results
with open(sys.argv[2], 'w') as f:
    json.dump({'scores': results}, f)
"""

    def _create_remote_timing_script(self) -> str:
        """Create the Python script with timing to run on the remote server."""
        return """
import sys
import json
import time
from tqdm import tqdm
from dnsmex.esm_wrapper import esm2_wrapper_of_size
from netam.common import pick_device

# Read input
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

sequences = data['sequences']
model_size = data['model_size']
include_timing = data.get('include_timing', False)

print(f"Loading ESM-{model_size} model...")
device = pick_device()
print(f"Using device: {device}")
esm_wrapper = esm2_wrapper_of_size(model_size, device=device)

results = []
print(f"Processing {len(sequences)} sequence pairs...")

if include_timing:
    print("Starting timed evaluation...")
    start_time = time.time()

for i, (heavy, light) in enumerate(tqdm(sequences, desc=f"Computing ESM-{model_size}")):
    # Evaluate heavy and light chains separately
    heavy_score = esm_wrapper.pseudo_perplexity(heavy)
    light_score = esm_wrapper.pseudo_perplexity(light)
    
    # Average the perplexities (FLAb paper approach)
    avg_score = (heavy_score + light_score) / 2
    results.append(avg_score)
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(sequences)} pairs")

if include_timing:
    elapsed_time = time.time() - start_time
    print(f"✅ ESM-{model_size} processing complete in {elapsed_time:.2f}s")
else:
    print(f"✅ ESM-{model_size} processing complete")

# Save results
result_data = {'scores': results}
if include_timing:
    result_data['wall_time_seconds'] = elapsed_time

with open(sys.argv[2], 'w') as f:
    json.dump(result_data, f)
"""


def get_remote_esm_evaluator(model_size: str = "3B"):
    """Get remote ESM evaluator for specified model size."""
    return RemoteESMEvaluator(model_size)
