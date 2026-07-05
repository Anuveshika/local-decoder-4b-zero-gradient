# Reproducing the Kaggle environment

## Exact executed context

- Python: 3.12.13, as recorded in notebook metadata.
- GPU: one Tesla T4 with 14.562 GiB reported binary capacity.
- Seed: 20260629 for Python, NumPy, PyTorch CPU and CUDA.
- Gradients: globally disabled before model construction.
- Core timer: begins after excluded data ingestion and covers initialization,
  training, all evaluation, figures and checkpoint serialization.

## Kaggle web-interface procedure

1. Upload or publish the prepared-data Dataset through the Kaggle web UI.
2. Upload the frozen benchmark input files as a public Dataset.
3. Verify the challenge host explicitly permits every training and
   conversational source and exact revision.
4. Create a new notebook and import
   `notebooks/source_six_cell_notebook.ipynb`.
5. Attach exactly one prepared-data Dataset and one benchmark-input Dataset.
6. Select a single NVIDIA T4. Do not choose P100, L4, dual GPU or TPU.
7. Enable Internet only as a fallback for public evaluation-file retrieval.
8. Run all six cells in order. Do not resume from a pretrained checkpoint.
9. Save a new Kaggle version with **Save & Run All**.
10. Compare the final output structure with `results/final_metrics.json`.

## Local CUDA environment

Install Python 3.12.13, create a clean environment and install the strict
requirements. A CUDA-enabled PyTorch wheel compatible with the local driver
may need to be installed from the official PyTorch wheel index instead of the
generic wheel in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate local-decoder-4b
pytest -q
```

## Inputs required for a full rerun

- Manifested FineWeb-Edu byte corpus used by the executed run.
- Manifested conversational positive/negative pairs used by the executed run.
- Frozen tokenized HellaSwag and PIQA records.
- Official WikiText-103 raw test Parquet.
- Official FastChat MT-Bench 80-question JSONL.

Prepared artifacts are verified by byte size and SHA-256 before model
construction. Never replace them silently. If the competition whitelist does
not include the exact training or conversation sources, the notebook is not
eligible even if it executes correctly.

## Verification commands

```bash
pytest -q
python scripts/extract_and_plot.py
```

The extraction script reads only the immutable executed notebook. It recreates
the result JSON files and media charts without modifying the run.

## Expected immutable values

```text
parameters              4,105,269,992
elapsed_seconds         6,343.889441096
peak_allocated_gib      7.7601146698
peak_reserved_gib       7.849609375
pretrain_steps          2,848
chat_steps              322
processed_tokens        1,623,040
```

