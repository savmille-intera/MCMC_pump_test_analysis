# Dewatering Synthetic Example

This directory isolates dewatering-specific synthetic analysis from the main pump-test workflow.

## Contents

- `dewatering_analysis.py` - Pump-rate sweep and volume/price risk analysis utilities.
- `mine_dewatering/` - Synthetic mine dewatering test folder and generated outputs.

## Run

From the repository root:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcmc-pump-test-analysis
python "example/dewatering_example/dewatering_analysis.py"
```

This script writes sweep summaries and figures inside `example/dewatering_example/mine_dewatering/`.

## Run via `run_wells.py`

If you prefer the batch runner, use `wells_to_run_ex.txt` (already pointing to this synthetic example):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcmc-pump-test-analysis
cd "main python files"
python run_wells.py --wells ../wells_to_run_ex.txt
```

For Gelman-Rubin multi-chain mode:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcmc-pump-test-analysis
cd "main python files"
python run_wells.py --gr --wells ../wells_to_run_ex.txt
```

