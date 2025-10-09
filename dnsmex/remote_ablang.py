"""Remote AbLang model evaluation on GPU servers."""

import json
import tempfile
import subprocess
from typing import List, Tuple, Dict, Any
import pandas as pd


def get_remote_config() -> Dict[str, str]:
    """Get remote server configuration for AbLang processing."""
    return {
        "host": "ermine",
        "user": "matsen",
        "project_dir": "/home/matsen/re/dnsm-experiments-1",
    }


class RemoteAbLangEvaluator:
    """AbLang model evaluator that runs on remote GPU servers."""

    def __init__(self):
        self.remote_config = get_remote_config()

    def evaluate_antibodies(self, sequences: List[Tuple[str, str]]) -> List[float]:
        """Evaluate sequences using AbLang model on remote GPU server.

        Args:
            sequences: List of (heavy, light) sequence pairs

        Returns:
            List of pseudo-perplexity scores
        """
        print(f"🚀 Running AbLang2 on {self.remote_config['host']} GPU...")

        # Create temporary files for input/output
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as input_file:
            json.dump(
                {"sequences": [{"heavy": h, "light": l} for h, l in sequences]},
                input_file,
            )
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
            output_path = output_file.name

        # Create remote script
        remote_script = f"""
import json
import torch
import sys
sys.path.append('{self.remote_config["project_dir"]}')
from dnsmex.ablang_wrapper import AbLangWrapper
from netam.common import pick_device

# Load input
with open('/tmp/ablang_input.json', 'r') as f:
    data = json.load(f)

# Setup optimal device
device = pick_device()
print(f"Using device: {{device}}")

# Initialize AbLang
wrapper = AbLangWrapper(device=device)

# Process sequences
sequences = data['sequences']
heavy_light_pairs = [[seq['heavy'], seq['light']] for seq in sequences]

# Get scores
scores = wrapper.pseudo_perplexity(heavy_light_pairs)

# Convert to list if numpy/tensor
import numpy as np
if isinstance(scores, (np.ndarray, torch.Tensor)):
    scores = scores.tolist()

# Save results
with open('/tmp/ablang_output.json', 'w') as f:
    json.dump({{'scores': scores}}, f)

print(f"Processed {{len(sequences)}} sequences")
"""

        # Create script file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(remote_script)
            script_path = script_file.name

        try:
            # Copy files to remote
            host = f"{self.remote_config['user']}@{self.remote_config['host']}"

            subprocess.run(
                ["scp", input_path, f"{host}:/tmp/ablang_input.json"],
                check=True,
                capture_output=True,
            )

            subprocess.run(
                ["scp", script_path, f"{host}:/tmp/ablang_script.py"],
                check=True,
                capture_output=True,
            )

            # Run script on remote with proper Python environment
            cmd = f"cd {self.remote_config['project_dir']} && source ../netam/.venv/bin/activate && python /tmp/ablang_script.py"
            try:
                result = subprocess.run(
                    ["ssh", host, cmd], check=True, capture_output=True, text=True
                )

                if result.stdout:
                    print(f"   Remote output: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Remote execution failed with code {e.returncode}")
                print(f"   Command: {e.cmd}")
                if e.stdout:
                    print(f"   Stdout: {e.stdout}")
                if e.stderr:
                    print(f"   Stderr: {e.stderr}")
                raise

            # Copy results back
            subprocess.run(
                ["scp", f"{host}:/tmp/ablang_output.json", output_path],
                check=True,
                capture_output=True,
            )

            # Read results
            with open(output_path, "r") as f:
                results = json.load(f)

            scores = results["scores"]
            print(f"   ✅ Received {len(scores)} scores from remote")

            # Cleanup remote files
            subprocess.run(
                [
                    "ssh",
                    host,
                    "rm -f /tmp/ablang_input.json /tmp/ablang_output.json /tmp/ablang_script.py",
                ],
                capture_output=True,
            )

            return scores

        finally:
            # Cleanup local temp files
            import os

            for path in [input_path, output_path, script_path]:
                if os.path.exists(path):
                    os.unlink(path)

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
        print(f"🚀 Running AbLang2 with timing on remote GPU...")

        # Create temporary files
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as input_file:
            json.dump({"sequences": sequences}, input_file)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as output_file:
            output_path = output_file.name

        # Create timing-enabled remote script
        remote_script = f"""
import json
import time
import sys
sys.path.append('{self.remote_config["project_dir"]}')
from dnsmex.ablang_wrapper import AbLangWrapper
from netam.common import pick_device

# Load input
with open('/tmp/ablang_input.json', 'r') as f:
    data = json.load(f)

sequences = data['sequences']

# Setup optimal device
device = pick_device()
print(f"Using device: {{device}}")

# Create AbLang wrapper with appropriate device
print("Loading AbLang2 model...")
ablang_wrapper = AbLangWrapper(device=device if device != 'cpu' else 'cpu')

# Time the evaluation
print(f"Starting timed evaluation of {{len(sequences)}} sequences...")
start_time = time.time()

# Process sequences as heavy-light pairs
heavy_light_pairs = [[heavy, light] for heavy, light in sequences]
scores = ablang_wrapper.pseudo_perplexity(heavy_light_pairs)

elapsed_time = time.time() - start_time
print(f"✅ AbLang2 processing complete in {{elapsed_time:.2f}}s")

# Convert numpy arrays to lists for JSON serialization
if hasattr(scores, 'tolist'):
    scores = scores.tolist()
elif isinstance(scores, list) and len(scores) > 0 and hasattr(scores[0], 'tolist'):
    scores = [s.tolist() if hasattr(s, 'tolist') else s for s in scores]

# Save results with timing
with open('/tmp/ablang_output.json', 'w') as f:
    json.dump({{'scores': scores, 'wall_time_seconds': elapsed_time}}, f)

print(f"Processed {{len(sequences)}} sequences")
"""

        # Create script file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as script_file:
            script_file.write(remote_script)
            script_path = script_file.name

        try:
            # Copy files to remote
            host = f"{self.remote_config['user']}@{self.remote_config['host']}"

            subprocess.run(
                ["scp", input_path, f"{host}:/tmp/ablang_input.json"],
                check=True,
                capture_output=True,
            )

            subprocess.run(
                ["scp", script_path, f"{host}:/tmp/ablang_timing_script.py"],
                check=True,
                capture_output=True,
            )

            # Run script on remote with proper Python environment
            cmd = f"cd {self.remote_config['project_dir']} && source ../netam/.venv/bin/activate && python /tmp/ablang_timing_script.py"
            try:
                result = subprocess.run(
                    ["ssh", host, cmd], check=True, capture_output=True, text=True
                )

                if result.stdout:
                    print(f"   Remote output: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Remote execution failed with code {e.returncode}")
                print(f"   Command: {e.cmd}")
                if e.stdout:
                    print(f"   Stdout: {e.stdout}")
                if e.stderr:
                    print(f"   Stderr: {e.stderr}")
                raise

            # Copy results back
            subprocess.run(
                ["scp", f"{host}:/tmp/ablang_output.json", output_path],
                check=True,
                capture_output=True,
            )

            # Load results
            with open(output_path, "r") as f:
                results = json.load(f)

            print(f"   ✅ Received timing results from remote")

            # Cleanup remote files
            subprocess.run(
                [
                    "ssh",
                    host,
                    "rm -f /tmp/ablang_input.json /tmp/ablang_output.json /tmp/ablang_timing_script.py",
                ],
                capture_output=True,
            )

            return results  # {"scores": [...], "wall_time_seconds": X.X}

        finally:
            # Cleanup local temp files
            import os

            for path in [input_path, output_path, script_path]:
                if os.path.exists(path):
                    os.unlink(path)
