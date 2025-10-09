# ProGen2 Setup for Ermine Server

This directory contains scripts to set up ProGen2 for remote antibody sequence evaluation on the ermine GPU server.

## Quick Setup

Run these commands on **ermine server**:

```bash
# 1. Set up ProGen2 environment (downloads models, creates venv)
bash ~/re/dnsm-experiments-1/scripts/setup_progen_env.sh

# 2. Copy analysis scripts to ProGen2 directory  
bash ~/re/dnsm-experiments-1/scripts/setup_progen_scripts.sh
```

## What Gets Created

```
~/progen2/
├── .venv/                    # Isolated Python environment
├── checkpoints/
│   └── progen2-small/        # Pre-trained model weights
├── tokenizer.json            # ProGen2 tokenizer
├── flab_progen.py           # Analysis script (copied from dnsm-experiments-1)
├── activate_progen.sh       # Helper activation script
└── likelihood.py            # ProGen2 core functions (from repo)
```

## Usage

After setup, the remote ProGen2 wrapper will automatically:

1. **Connect to ermine** via SSH
2. **Copy input sequences** to `/home/matsen/progen2/temp_input_*.csv`
3. **Run ProGen2 scoring** in the isolated environment:
   ```bash
   cd /home/matsen/progen2
   source .venv/bin/activate
   python flab_progen.py --input_file temp_input_*.csv --model progen2-small --device cuda:0
   ```
4. **Retrieve results** and clean up temporary files
5. **Return negated perplexity scores** for consistency with other models

## Integration

The setup integrates seamlessly with the existing Kirby analysis pipeline:

```python
# In notebooks/dasm_paper/kirby_binary.ipynb
models = ['dasm', 'dasm_ft_mild', 'esm', 'ablang', 'progen']  # ProGen2 included

# In scripts/kirby_cache_model_scores.py  
partition.add_progen_scores(model_version="progen2-small")  # Automatic remote execution
```

## Manual Testing

To manually test ProGen2 on ermine:

```bash
# Activate environment
cd ~/progen2
source .venv/bin/activate

# Test with sample data
echo "sequence_id,heavy,light" > test.csv
echo "test1,QVQLVQ...,DIQMTQ..." >> test.csv

# Run ProGen2 scoring
python flab_progen.py --input_file test.csv --model progen2-small --device cuda:0 --output_dir output

# Check results
cat output/test.progen2-small.csv
```

## Troubleshooting

- **CUDA not available**: ProGen2 will fallback to CPU (very slow)
- **Import errors**: Ensure `flab_progen.py` is in the ProGen2 directory
- **Model download fails**: Check internet connection and disk space
- **Remote connection fails**: Verify SSH access to ermine server

## Model Sizes

Available ProGen2 model variants:
- `progen2-small` (151M params) - Default, fastest
- `progen2-medium` (764M params) - Better quality  
- `progen2-large` (2.7B params) - High quality, slower
- `progen2-xlarge` (6.4B params) - Best quality, very slow

The setup script downloads `progen2-small` by default. Additional models can be downloaded manually if needed.