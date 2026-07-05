# Competition compliance matrix

| Requirement | Evidence | Status |
|---|---|---|
| One T4 or four-core CPU | Cell 1 prints Tesla T4 | Pass |
| At least 4B parameters | Cell 3 prints 4,105,269,992 | Pass |
| Random initialization | Cell 3; no checkpoint load | Pass |
| No automatic differentiation | Gradients disabled; source scan empty | Pass |
| No standard optimizer | Raw `plasticity_` updates | Pass |
| No downstream error to earlier blocks | Structural local block interface | Pass |
| Under 180 minutes | 105.73 minutes | Pass |
| At least 50% memory reduction | 83.09% vs analytical state bound | Partial: analytical comparison |
| HellaSwag | 24.61%, n=256 subset | Executed, target not met, not full set |
| PIQA | 51.17%, n=256 subset | Executed, target not met, not full set |
| WikiText-103 | byte PPL 56.77, 32,768 bytes | Executed bounded byte metric |
| MT-Bench | 80 questions / 160 turns | Generated; official judge score absent |
| Exact host-whitelisted training data | Entrant must verify current whitelist | Pending |
| SOTA benchmark targets | Results near chance | Fail |

`valid_run: true` in the metrics means the notebook completed its programmed
checks. It is not a declaration by Kaggle that the entry is prize eligible.

