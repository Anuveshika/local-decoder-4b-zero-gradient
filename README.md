# Local Decoder 4B: Zero-Gradient Sparse Learning on One T4

This repository is the complete open research artifact for a randomly
initialized **4,105,269,992-parameter** sparse sequence model trained with
layer-local Forward-Forward losses and an independent local next-byte decoder.
It uses no automatic differentiation, no `backward()` call, no standard
optimizer, no pretrained weights, and no teacher model.

The immutable six-cell Kaggle execution completed on one Tesla T4 in
**105.73 minutes**, with **7.76 GiB peak allocated VRAM**. The run executed
bounded HellaSwag, PIQA and WikiText evaluations and generated both turns for
all 80 MT-Bench questions. The measured language results are near chance and
must not be presented as state of the art.

## Executed results

| Measurement | Executed value |
|---|---:|
| Parameters | 4,105,269,992 |
| Random seed | 20260629 |
| Hardware | One Tesla T4 |
| End-to-end core runtime | 6,343.89 s / 105.73 min |
| Positive tokens processed | 1,623,040 |
| Peak allocated / reserved VRAM | 7.76 / 7.85 GiB |
| Analytical AdamW state lower bound | 45.88 GiB |
| Reduction vs analytical bound | 83.09% |
| HellaSwag | 24.61%, deterministic n=256 subset |
| PIQA | 51.17%, deterministic n=256 subset |
| WikiText-103 | byte PPL 56.77 on 32,768 bytes |
| MT-Bench | 80 questions, 160 nonempty turns, no official judge score |

The AdamW number is an analytical 12-byte-per-parameter state bound, not a
matched full AdamW training run. Small measured AdamW probes allocate state
tensors only. WikiText byte perplexity is not directly comparable with
word/subword perplexity. MT-Bench requires an approved external judge for an
official score.

## Algorithm

For block `l`, positive real byte windows and locally corrupted negative
windows pass through the same sparse block. Goodness is measured before RMS
normalization:

```text
g_l(y) = mean(y_l²)
L_l = mean(softplus(theta - g_l(y+)))
    + mean(softplus(g_l(y-) - theta))
```

Every block explicitly forms derivatives using only its own inputs,
activations and goodness. Its update is:

```text
Delta W_l = -eta_l * clip(G_l / max(RMS(G_l), epsilon), -c, c)
```

No loss, target, derivative or message from block `l+1` is used to update
block `l`. The local decoder computes a normalized distribution over 264 active
byte/special IDs and updates only its own weight and bias. Decoder error is
never propagated into the backbone.

This is a sparse local Forward-Forward MoE with a local predictive decoder. It
is related to, but not identical to, Ororbia and Mali's recurrent Predictive
Forward-Forward circuit.

## Architecture

- Allocated vocabulary: 32,768 slots; 264 active byte/special IDs.
- Width: 1,024.
- Depth: 20 sparse local blocks.
- Experts: 24 per block.
- Expert hidden width: 4,096.
- Routing: one selected expert per token per block.
- Backbone parameters: 4,104,999,392.
- Local decoder parameters: 270,600.
- Total: 4,105,269,992.

## Repository map

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── notebooks/
│   ├── executed_kaggle_run.ipynb
│   └── source_six_cell_notebook.ipynb
├── src/
│   ├── local_decoder_4b.py
│   └── inference_cpu.py
├── scripts/
│   ├── extract_and_plot.py
│   └── quantize_checkpoint_int8.py
├── docs/
│   ├── KAGGLE_WRITEUP.md
│   ├── REPRODUCIBILITY.md
│   ├── DEPLOYMENT.md
│   ├── COMPLIANCE.md
│   ├── VIDEO_SCRIPT.md
│   └── SUBMISSION_CHECKLIST.md
├── media/
│   ├── cover.png
│   ├── memory_profile.png
│   ├── training_curves.png
│   ├── benchmark_summary.png
│   └── architecture.png
├── slides/oral_defense.pptx
├── results/
└── tests/
```

## Kaggle reproduction

The public execution artifact is
`notebooks/executed_kaggle_run.ipynb`. It has exactly six executed cells in the
required order. To run a new version:

1. Sign in to Kaggle and create a notebook.
2. Import `notebooks/source_six_cell_notebook.ipynb`.
3. Attach the public prepared-data Dataset used by the run.
4. Attach the frozen official WikiText-103 and MT-Bench evaluation inputs.
5. Confirm every attached training source and exact version is on the host's
   whitelist. Do not substitute private or modified data.
6. Select **GPU T4 x1** and Python 3.12.
7. Run **Save Version → Save & Run All**.
8. Confirm Cell 1 prints seed `20260629`, one T4 and gradients disabled.
9. Confirm Cell 3 prints `4,105,269,992` parameters and random initialization.
10. Confirm Cell 4 prints an empty forbidden-API violation list.
11. Confirm Cell 6 prints `valid_run: true`, `completed_under_180_minutes:
    true`, 20 block shards and a decoder checkpoint.
12. Make the successful notebook version public and attach that exact version
    to the Kaggle Project Files field.

See `docs/REPRODUCIBILITY.md` for local and Kaggle setup details.

## Local setup

Windows PowerShell:

```powershell
git clone https://github.com/Anuveshika/local-decoder-4b-zero-gradient.git
cd local-decoder-4b-zero-gradient
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/extract_and_plot.py
```

Linux:

```bash
git clone https://github.com/Anuveshika/local-decoder-4b-zero-gradient.git
cd local-decoder-4b-zero-gradient
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/extract_and_plot.py
```

The full CUDA reproduction requires one T4-class GPU and the public Kaggle
inputs. Tests and metric extraction do not require the 4B checkpoint.

## CPU inference

Download the successful Kaggle output checkpoint and run:

```bash
python -m src.inference_cpu \
  --checkpoint /path/to/checkpoint \
  --prompt "Explain why local learning reduces optimizer-state memory." \
  --max-new-bytes 64
```

The loader avoids constructing a second random 4B model. Native FP16 shards
occupy approximately 8.2 GB before framework overhead. A practical CPU host
should have at least 12–16 GB free RAM. Generation is research-grade and slow;
the measured benchmark quality is not competitive.

For edge storage export:

```bash
python scripts/quantize_checkpoint_int8.py \
  --checkpoint /path/to/checkpoint \
  --output /path/to/checkpoint-int8
```

This creates an approximately 4.1 GB symmetric-int8 shard archive for
integration with an optimized edge kernel. It is not a promise that phones,
microcontrollers or low-RAM boards can execute the full model. See
`docs/DEPLOYMENT.md`.

## Submission assets

1. Formal paper: `docs/KAGGLE_WRITEUP.md`.
2. Media Gallery: `media/`.
3. Execution artifact: `notebooks/executed_kaggle_run.ipynb`.
4. Oral defense: `slides/oral_defense.pptx` and `docs/VIDEO_SCRIPT.md`.
5. Deployment hub: https://github.com/Anuveshika/local-decoder-4b-zero-gradient

## Scientific and compliance boundaries

- The execution is stable and under 180 minutes.
- It does not meet the stated benchmark targets.
- HellaSwag and PIQA are bounded subsets, not full official evaluations.
- WikiText is bounded byte-token perplexity.
- MT-Bench outputs are unjudged.
- Only 1.623M positive tokens were processed.
- The exact dataset whitelist must be confirmed by the entrant before
  submission.
- No claim of SOTA, matched AdamW speed superiority or prize eligibility is
  made by this repository.

## References

- A. Ororbia and A. Mali, “The Predictive Forward-Forward Algorithm,”
  arXiv:2301.01452, 2023.
- L. Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,”
  arXiv:2306.05685, 2023.

## License

Apache-2.0. Dataset and benchmark licenses remain with their owners.
