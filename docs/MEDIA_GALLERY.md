# Media Gallery upload order and captions

## 1. Cover image

File: `media/cover.png`

**Caption:** Local Decoder 4B is a 4,105,269,992-parameter, zero-autograd sparse
local-learning experiment executed from random initialization on one Tesla T4
in 105.73 minutes. Peak allocated VRAM was 7.76 GiB. Language benchmarks
remained near chance.

## 2. Memory allocation

File: `media/memory_profile.png`

**Caption:** Measured allocated and reserved VRAM over wall-clock time. T4
capacity is hardware reported. The 45.88 GiB AdamW line is an analytical
12-byte-per-parameter state lower bound. The 25M and 75M AdamW lines are
measured state allocations only, not full training runs.

## 3. Training diagnostics

File: `media/training_curves.png`

**Caption:** Executed wall-clock traces for positive and negative energy,
energy margin, contrastive and next-byte accuracy, decoder reconstruction MSE,
sampled update coverage, effective throughput, VRAM, local loss and decoder
NLL. All 155 logged rows remained finite.

## 4. Architecture

File: `media/architecture.png`

**Caption:** Positive and active-ID-corrupted byte windows pass forward through
twenty top-1 sparse blocks. Each block computes and consumes its own goodness
derivative. The local byte decoder updates only itself; no downstream loss or
error returns to the backbone.

## 5. Benchmark summary

File: `media/benchmark_summary.png`

**Caption:** Executed bounded outputs: HellaSwag 24.61% and PIQA 51.17% on
deterministic 256-example subsets, WikiText-103 byte perplexity 56.77 on 32,768
bytes, and 160 nonempty MT-Bench turns. The chart mixes accuracy and
perplexity, so the bars are descriptive rather than directly comparable. No
official MT-Bench judge score is claimed.
