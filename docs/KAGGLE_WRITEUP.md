# Local Decoder 4B: Sparse Local Forward-Forward Learning on One T4

**Track:** Zero-Gradient Main Track  
**Artifact status:** Complete executed research artifact; benchmark targets not achieved

## Abstract

We present Local Decoder 4B, a 4,105,269,992-parameter sparse sequence model
trained from random initialization without automatic differentiation, global
error propagation, or a standard optimizer. Twenty mixture-of-experts blocks
use independent positive-versus-corrupted goodness objectives. Each block
forms its derivatives from block-owned activations, updates its tensors through
explicit normalized raw-tensor operations, discards its local derivative, and
passes only a normalized representation forward. A separate 270,600-parameter
byte decoder learns a local next-token distribution; its prediction error is
never returned to the backbone. The six-cell execution on one Tesla
T4 finished in 6,343.89 seconds, or 105.73 minutes. Peak allocated VRAM was
7.76 GiB, 83.09% below a 45.88 GiB analytical AdamW state lower bound, although
this is not a matched full AdamW training measurement. The run processed
1,623,040 positive tokens. It achieved 24.61% HellaSwag accuracy and 51.17%
PIQA accuracy on deterministic 256-example subsets, byte perplexity 56.77 on a
32,768-byte WikiText-103 slice, and generated 160 nonempty MT-Bench turns
without an official judge score. The experiment demonstrates stable
billion-scale local updates, bounded memory, normalized prediction, and
generation on constrained hardware. It does not demonstrate state-of-the-art
language modeling or a successor to backpropagation.

## 1. Problem and contribution

Backpropagation coordinates all layers through the derivative of a global
objective. Its practical implementation normally retains activations, stores a
gradient for every trainable parameter, and maintains optimizer state. At four
billion parameters, even weight and optimizer-state storage alone exceeds a
single T4 for conventional mixed-precision AdamW. This project asks a narrower
and experimentally testable question: can more than four billion parameters be
initialized, adapted through genuinely local rules, evaluated as a generative
sequence system, and serialized on one T4 within 180 minutes?

The result is Local Decoder 4B, a sparse local Forward-Forward mixture of
experts with an independent local byte decoder. Its main contributions are:

1. an audit-visible 4,105,269,992-parameter construction;
2. fixed random initialization with no imported weights;
3. explicit block-local derivatives and in-place tensor updates;
4. no `torch.autograd`, `backward()`, standard optimizer, teacher model, or
   distillation target;
5. one active expert per token and block, limiting transient storage;
6. a local normalized next-byte decoder whose error does not enter the
   backbone;
7. cached autoregressive generation over the same causal block computation;
8. a complete 105.73-minute T4 execution with memory and training traces;
9. four benchmark execution paths with explicit scope labels; and
10. a complete 20-block plus decoder checkpoint.

This is not evidence that local learning matches backpropagation in quality.
The downstream results remain near chance. The artifact is valuable because it
exposes both the engineering feasibility and the present scientific failure
mode.

## 2. Mathematical foundation

### 2.1 Causal sparse block

Let (X_l^+) and (X_l^-) be positive and negative representations received
by block (l), with shape (B\times T\times d). The negative sequence is made
by replacing tokens within the active 264-ID byte vocabulary. The causal
prefix operator excludes the current position:

$$
C(X)_t = \begin{cases}
0,& t=0,\\
\frac{1}{t}\sum_{i=0}^{t-1}X_i,& t>0.
\end{cases}
$$

The block concatenates the current state with its prefix context:

$$
Q_l=[X_l;C(X_l)].
$$

A gated stem computes

$$
P_l=Q_lM_l+b_l,
\qquad
H_l=P_l\odot\sigma(P_l).
$$

A local router selects one of (E=24) experts:

$$
r_l=\arg\max_e(H_lR_l+c_l)_e.
$$

For selected expert (e),

$$
A_{l,e}=H_lU_{l,e}+u_{l,e},
$$

$$
V_{l,e}=A_{l,e}\odot\sigma(A_{l,e}),
$$

$$
Y_l=H_l+V_{l,e}D_{l,e}+d_{l,e}.
$$

Only one expert is active per token, although all expert tensors contribute to
the parameter count.

### 2.2 Local goodness objective

Goodness is measured before RMS normalization:

$$
g_l(Y)=\frac{1}{d}\|Y\|_2^2.
$$

This ordering is essential. Measuring the mean square after RMS normalization
would make goodness approximately one for every token and destroy the
contrastive signal. The block-owned loss is

$$
\mathcal{L}_l =
\frac{1}{N_+}\sum_i \operatorname{softplus}(\theta-g_l(Y^+_{l,i}))+
\frac{1}{N_-}\sum_j \operatorname{softplus}(g_l(Y^-_{l,j})-\theta).
$$

Positive samples are encouraged above threshold (	heta); negative samples
are encouraged below it. The local output derivatives are

$$
\delta_{Y^+}=-\frac{\sigma(\theta-g^+)}{N_+}\frac{2Y^+}{d},
$$

$$
\delta_{Y^-}=\frac{\sigma(g^--\theta)}{N_-}\frac{2Y^-}{d}.
$$

Within the selected expert, block-owned outer products form

$$
G_{D_l}=V_l^\top\delta_Y,
\qquad
G_{U_l}=H_l^\top\delta_A,
\qquad
G_{M_l}=Q_l^\top\delta_P.
$$

These derivatives traverse operations inside block (l), but never cross a
block boundary. Local differentiation inside a layer is not the global chain
rule across the network.

### 2.3 Raw-tensor update

For every block-owned tensor, the implementation applies

$$
\Delta W_l=-\eta_l\operatorname{clip}\left(
\frac{G_l}{\max(\sqrt{\operatorname{mean}(G_l^2)},\epsilon)},-c,c
\right).
$$

Normalization is completed in FP32, with (epsilon=10^{-6}), before the
change is converted to the tensor dtype. Non-finite changes are replaced by
zero. This guard corrected an earlier FP16 denominator-underflow failure. The
weight is mutated directly with an in-place addition. There is no optimizer
object or hidden gradient tape.

The router receives only selected-route local reward. Positive routes use
(sigma(\theta-g^+)), negative routes use
(-\sigma(g^--\theta)), and a batch-local balancing bias discourages route
collapse. No derivative through `argmax` is calculated.

### 2.4 Local predictive decoder

The decoder receives the final representation (H), but its error is not sent
backward. For active byte vocabulary size (V_a=264),

$$
Z=HW_d+b_d,
\qquad
P=\operatorname{softmax}(Z).
$$

For observed next-byte targets (Y), its local derivative is

$$
D=P-\operatorname{onehot}(Y),
$$

$$
G_{W_d}=H^\top D,
\qquad
G_{b_d}=\sum D.
$$

The same normalized raw-tensor update is applied only to (W_d) and (b_d).
The implementation returns no decoder derivative to any backbone block.

### 2.5 Locality proof

Let (mathcal{S}_l) be the state observable inside block (l): its positive
and negative inputs, its own parameters, routes, activations, goodness values,
and fixed hyperparameters. The implemented update is a deterministic function

$$
\Delta W_l=f_l(\mathcal{S}_l).
$$

Neither the interface of `SparseLocalBlock.learn` nor any object reachable by
that function contains (mathcal{L}_{l+1}), a downstream target, or a
downstream derivative. After updating, the block returns only normalized
positive and negative representations plus scalar diagnostics. Therefore

$$
\frac{\partial \Delta W_l}{\partial \mathcal{L}_k}=0
\quad\text{for every } k>l
$$

by structural independence, not gradient blocking. Similarly, the decoder's
local error is consumed within the decoder and never appears in a backbone
update. This establishes the required locality property of the software
artifact.

## 3. Architecture design

The model allocates a 32,768 by 1,024 embedding table:

$$
32768\times1024=33,554,432
$$

parameters. The deterministic tokenizer actively uses 256 byte values, four
control IDs, and four dialogue-role IDs, for 264 active IDs. The larger table
is an architectural allocation rather than evidence that all vocabulary rows
were trained.

Each expert contains two matrices and two biases:

$$
1024\times4096+4096\times1024+4096+1024=8,393,728.
$$

Including the causal stem, router, biases, and 24 experts gives 203,572,248
parameters per block. With twenty blocks:

$$
33,554,432+20(203,572,248)=4,104,999,392
$$

backbone parameters. The decoder adds

$$
1024\times264+264=270,600,
$$

for an exact total of

$$
\boxed{4,105,269,992}.
$$

All weights are seeded random-normal tensors and all biases are zeros. No
checkpoint is read before training. Sparse routing limits active expert compute
and creates derivative tensors for one expert at a time. It does not reduce the
declared allocated weights.

## 4. Data, pretraining and conversational tuning

The prepared execution artifact contains 100,007,268 byte tokens available for
sampling and 24,002 conversational positive/negative pairs. The run samples
microbatches of four windows with length 128. Every prepared file is validated
by declared size and SHA-256 before model construction. Negative windows
replace approximately 15% of positions with IDs sampled only from the active
vocabulary, followed by one guaranteed final-position change.

The core timer begins after excluded ingestion. The first phase runs until
minute 90 and updates every sparse block plus the local decoder. The run
completed 2,848 pretraining steps. From minute 90 to minute 100, the same local
rules adapt on dialogue pairs marked with user, assistant, and turn-end IDs,
completing 322 conversational steps. Across both phases, 1,623,040 positive
tokens were processed.

Conversational adaptation is local predictive training, not evidence of a
high-quality assistant. It teaches the decoder byte patterns in dialogue-form
sequences while each block continues positive-versus-negative discrimination.
Cached autoregressive inference then processes prompts one token at a time,
maintaining a prefix sum at each layer. This avoids recomputing the full
128-token context for every generated byte.

Dataset eligibility is a separate compliance requirement. Before submission,
the entrant must verify that every exact training and conversational source is
on the host's current whitelist. Successful execution does not override a
whitelist restriction.

## 5. Hardware, timing and memory profiling

The executed notebook used one Tesla T4 reporting 14.562 GiB binary capacity.
The core timer covers 4.105B initialization, both training phases, validation,
four evaluation paths, figures, and sharded checkpoint serialization. It ended
at 6,343.889 seconds, or 105.731 minutes, leaving 74.269 minutes before the
180-minute limit.

Peak allocated VRAM was 7.760 GiB and peak reserved VRAM was 7.850 GiB. The
weight footprint dominates, while local derivative tensors are short-lived and
no global parameter-gradient or optimizer-state copy exists.

For the same 4,105,269,992 parameters, a minimal mixed-precision AdamW state
accounting of FP16 parameters, FP16 gradients, and two FP32 moments requires 12
bytes per parameter:

$$
12(4,105,269,992)/2^{30}=45.880\text{ GiB}.
$$

Relative to that analytical state bound,

$$
1-\frac{7.760}{45.880}=0.8309,
$$

or 83.09% lower allocated memory. The notebook also measures raw state
allocation for 25M and 75M parameter probes. Those probes do not perform a
matched AdamW training step, and the 45.88 GiB line is analytical. Therefore
the artifact supports a memory-state argument, not an empirical claim that the
algorithm is faster than a quality-matched AdamW run.

Throughput stabilized near 270 positive tokens per second. No matched
backpropagation throughput baseline was executed, so speed superiority is not
claimed.

## 6. Learning behavior and validation

All 155 logged trace rows are finite. During pretraining, average local loss
declined from 0.824 over the first fifth of logs to 0.733 over the last fifth.
Mean block-local accuracy increased from 50.56% to 54.89%. Decoder NLL declined
from 5.610 to 4.288 and next-byte accuracy increased from 3.73% to 8.44%.
Mean energy margin increased from 0.046 to 0.302, while contrastive accuracy
increased from 71.30% to 95.37%.

The improvement did not transfer strongly to held-out windows. Validation over
80 real-versus-corrupt comparisons reached 52.5%, with mean margin -0.00781.
This near-chance result indicates over-specialization to sampled training
windows or a mismatch between logged minibatch discrimination and held-out
generalization. Conversational adaptation also reduced the final energy margin.

## 7. Four benchmark execution paths

HellaSwag and PIQA use mean decoder continuation negative log-likelihood.
Deterministic 256-example subsets produced 24.61% HellaSwag accuracy and 51.17%
PIQA accuracy. These are approximately the four-choice and two-choice chance
levels and do not meet competitive thresholds. They are subsets, not full
official validation runs.

WikiText-103 evaluation uses 32,768 UTF-8 bytes from the raw test data. It
predicts 32,512 next bytes, yielding mean NLL 4.039 nats and byte perplexity
56.77. This is a valid metric for the model's byte tokenizer, but it is not
directly comparable with conventional word or subword perplexity and should
not be presented as satisfying a standard WikiText threshold.

MT-Bench generation runs both turns for all 80 official questions. The output
contains 160 nonempty turns in FastChat-compatible answer JSONL. Official
MT-Bench scoring normally uses a strong LLM judge, as documented by the
FastChat project. No organizer-approved judge was provided to this offline run,
so the official score is null. Nonempty output proves pipeline execution, not
answer quality.

The benchmark conclusion is therefore negative: the local decoder enables
normalized scoring and generation, but the short training exposure does not
produce competitive language understanding or dialogue.

## 8. Relationship to prior work

Ororbia and Mali's Predictive Forward-Forward algorithm combines recurrent
bottom-up and top-down representation dynamics, learnable lateral competition,
noise, and a separate generative circuit. Local Decoder 4B does not implement
that full recurrent design. It uses feed-forward sparse blocks with independent
goodness losses and one final local byte decoder. The prior work motivates
local predictive circuits, but its image-model results cannot be transferred
to this four-billion-parameter byte model without experiment.

MT-Bench is drawn from the FastChat evaluation framework introduced by Zheng
and collaborators. Its use of an LLM-as-judge is why this artifact separates
answer generation from official judging.

## 9. Limitations and threats to validity

The model processes only 1.623 million positive tokens, approximately 0.0406%
of four billion tokens. Parameter count is not training exposure. The context
length is 128 bytes, most allocated vocabulary rows are inactive, validation is
near chance, and benchmark performance is noncompetitive. The byte decoder can
produce malformed or low-quality text and has not undergone safety alignment.

The memory comparison is not a matched end-to-end AdamW baseline with identical
quality, microbatch and checkpointing. The 4.105B AdamW line is an analytical
state lower bound. The small measured probes allocate state but do not execute
backpropagation. The experiment therefore does not prove speed superiority.

The checkpoint is too large for most consumer edge devices. FP16 inference
requires roughly 8.2 GB for weights; int8 storage requires roughly 4.1 GB before
runtime workspace. The included CPU loader is intended for research machines,
not production service.

Finally, prize eligibility depends on the host's exact current dataset
whitelist and interpretation of benchmark scope. The entrant must confirm those
conditions. This paper does not convert a pending compliance item into a pass.

## 10. Conclusion

Local Decoder 4B demonstrates that 4.105B randomly initialized parameters can
be adapted with explicit layer-local rules, supplied with a separately local
predictive decoder, evaluated through four sequence benchmark paths, and saved
on one T4 in 105.73 minutes with peak allocation below 8 GiB. The source is
audit visible and the decoder error never updates the backbone. Training is
numerically stable and the local objectives improve.

The same experiment also shows the present gap: held-out discrimination and
language benchmarks remain near chance, WikiText uses a bounded byte metric,
and MT-Bench is unjudged. This is an engineering proof of scale and memory
economy, not a quality proof against backpropagation. Future work should expand
training exposure, improve locally predictive representations, run full
official evaluations, obtain an approved MT-Bench judgment, and measure a
quality-matched empirical AdamW baseline.

## Media captions

**Memory allocation.** Measured allocated and reserved VRAM over wall-clock
time. The T4 line is device-reported capacity. The 45.88 GiB AdamW line is an
analytical 12-byte-per-parameter state lower bound; the 25M and 75M lines are
measured allocation probes, not training runs.

**Training curves.** Measured pre-RMS energy, energy margin, local and decoder
accuracy, decoder reconstruction MSE, sampled update coverage, throughput,
VRAM, local loss, and decoder NLL.

**Architecture.** Positive and corrupted byte sequences pass forward through
twenty sparse blocks. Every block computes and consumes its own goodness
derivative. The final decoder consumes only the final representation and does
not return error to the backbone.

## References

1. A. Ororbia and A. Mali, “The Predictive Forward-Forward Algorithm,”
   arXiv:2301.01452, 2023. https://arxiv.org/abs/2301.01452
2. L. Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,”
   arXiv:2306.05685, 2023. https://arxiv.org/abs/2306.05685
3. LMSYS, FastChat MT-Bench evaluation repository.
   https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
