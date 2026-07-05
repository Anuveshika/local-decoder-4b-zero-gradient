"""Extract immutable executed-notebook metrics and generate media charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "executed_kaggle_run.ipynb"
RESULTS = ROOT / "results"
MEDIA = ROOT / "media"
RESULTS.mkdir(exist_ok=True)
MEDIA.mkdir(exist_ok=True)

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert len(notebook["cells"]) == 6
assert [cell.get("execution_count") for cell in notebook["cells"]] == [1, 2, 3, 4, 5, 6]

cell5 = "".join(notebook["cells"][4]["outputs"][0].get("text", []))
trace = []
for line in cell5.splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(row, dict) and "phase" in row:
        trace.append(row)
assert trace and all(np.isfinite(float(row["local_loss"])) for row in trace)

cell6 = "".join(notebook["cells"][5]["outputs"][0].get("text", []))
metrics, _ = json.JSONDecoder().raw_decode(cell6.lstrip())
assert metrics["valid_run"] and metrics["completed_under_180_minutes"]

memory = [{key: row[key] for key in ("elapsed", "allocated_gib", "reserved_gib")}
          for row in trace]
(RESULTS / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(RESULTS / "training_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
(RESULTS / "memory_trace.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")

plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "figure.facecolor": "white"})
minutes = np.asarray([row["elapsed"] / 60 for row in trace])
series = lambda key: np.asarray([row[key] for row in trace], dtype=float)

fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True)
axes = axes.ravel()
axes[0].plot(minutes, series("positive_energy"), label="Positive energy", lw=1.8)
axes[0].plot(minutes, series("negative_energy"), label="Negative energy", lw=1.8)
axes[0].set_title("Positive and negative energy — measured"); axes[0].legend()
axes[1].plot(minutes, series("energy_margin"), color="#7c3aed", lw=1.8)
axes[1].axhline(0, color="black", lw=.8); axes[1].set_title("Energy margin — measured")
axes[2].plot(minutes, series("contrastive_accuracy"), label="Contrastive accuracy")
axes[2].plot(minutes, series("decoder_accuracy"), label="Next-byte accuracy")
axes[2].set_ylim(0, 1); axes[2].set_title("Local accuracies — measured"); axes[2].legend()
axes[3].plot(minutes, series("decoder_reconstruction_mse"), color="#92400e")
axes[3].set_title("Decoder one-hot reconstruction MSE — measured")
axes[4].plot(minutes, series("sampled_nonzero_parameter_update_coverage"), color="#15803d")
axes[4].set_ylim(0, 1); axes[4].set_title("Sampled nonzero update coverage — measured")
axes[5].plot(minutes, series("tokens_per_second"), color="#ea580c")
axes[5].set_title("Effective tokens/second — measured")
axes[6].plot(minutes, series("allocated_gib"), label="Allocated")
axes[6].plot(minutes, series("reserved_gib"), label="Reserved")
axes[6].axhline(14.56219482421875, color="black", ls=":", label="T4 reported capacity")
axes[6].set_title("VRAM over wall-clock time — measured"); axes[6].legend()
axes[7].plot(minutes, series("local_loss"), label="Forward-Forward local loss")
axes[7].plot(minutes, series("decoder_nll"), label="Decoder NLL")
axes[7].set_title("Local objectives — measured"); axes[7].legend()
for axis in axes:
    axis.grid(alpha=.25); axis.set_xlabel("Elapsed wall-clock time (minutes)")
fig.suptitle("Local Decoder 4B training trace", fontsize=20, fontweight="bold", y=.997)
fig.tight_layout(); fig.savefig(MEDIA / "training_curves.png", dpi=220); plt.close(fig)

adamw = metrics["adamw_12_bytes_per_parameter_lower_bound_gib"]
fig, axis = plt.subplots(figsize=(14, 7.5))
axis.plot(minutes, series("allocated_gib"), lw=2.2, label="Local Decoder 4B allocated — measured")
axis.plot(minutes, series("reserved_gib"), lw=2.2, label="Local Decoder 4B reserved — measured")
axis.axhline(14.56219482421875, color="black", ls=":", lw=2,
             label="T4 capacity — hardware reported (14.56 GiB)")
axis.axhline(adamw, color="#dc2626", ls="--", lw=2.2,
             label=f"AdamW 12 B/parameter lower bound — analytical ({adamw:.2f} GiB)")
axis.axhline(.28125, color="#7c3aed", ls="-.", label="AdamW 25M state allocation — measured")
axis.axhline(.840044, color="#0f766e", ls="-.", label="AdamW 75M state allocation — measured")
axis.set(title="Measured local-learning VRAM vs analytical and measured AdamW references",
         xlabel="Elapsed wall-clock time (minutes)", ylabel="Memory (GiB)", ylim=(0, 50))
axis.grid(alpha=.25); axis.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(MEDIA / "memory_profile.png", dpi=240); plt.close(fig)

labels = ["HellaSwag\n256 subset", "PIQA\n256 subset", "WikiText\nbyte PPL"]
values = [100 * metrics["hellaswag"]["accuracy"], 100 * metrics["piqa"]["accuracy"],
          metrics["wikitext_103"]["byte_perplexity"]]
fig, axis = plt.subplots(figsize=(11, 7))
bars = axis.bar(labels, values, color=["#111827", "#4b5563", "#f97316"])
axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=12)
axis.set_ylabel("Accuracy (%) for choice tasks; perplexity for WikiText")
axis.set_title("Executed benchmark outputs — metrics use different units")
axis.grid(axis="y", alpha=.2)
axis.text(.01, -.14, "MT-Bench: 80 questions / 160 nonempty turns generated; official judge score not available.",
          transform=axis.transAxes, fontsize=10)
fig.tight_layout(); fig.savefig(MEDIA / "benchmark_summary.png", dpi=220); plt.close(fig)

summary = {
    "runtime_minutes": metrics["elapsed_seconds"] / 60,
    "parameters": metrics["parameters"],
    "processed_tokens": metrics["processed_tokens"],
    "peak_allocated_gib": metrics["peak_allocated_gib"],
    "peak_reserved_gib": metrics["peak_reserved_gib"],
    "memory_reduction_vs_analytical_adamw": metrics["memory_reduction_vs_adamw_lower_bound"],
    "trace_rows": len(trace),
}
(RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Submission cover with only executed values.
fig = plt.figure(figsize=(16, 9), facecolor="#07111f")
axis = fig.add_axes([0, 0, 1, 1]); axis.set_axis_off()
axis.text(.06, .88, "ZERO-AUTOGRAD RESEARCH ARTIFACT", color="#fb923c", fontsize=17, weight="bold")
axis.text(.06, .68, "Local Decoder 4B", color="white", fontsize=50, weight="bold")
axis.text(.06, .57, "Sparse local Forward-Forward MoE with a local byte decoder", color="#cbd5e1", fontsize=21)
axis.text(.06, .38, "4,105,269,992 random parameters on one Tesla T4", color="white", fontsize=25)
panel = FancyBboxPatch((.67, .18), .25, .62, boxstyle="round,pad=.015,rounding_size=.015",
                       transform=axis.transAxes, facecolor="#13263d", edgecolor="#fb923c", linewidth=2)
axis.add_patch(panel)
axis.text(.795, .62, "105.73", ha="center", color="white", fontsize=44, weight="bold")
axis.text(.795, .56, "minutes", ha="center", color="#cbd5e1", fontsize=19)
axis.text(.795, .39, "7.76 GiB", ha="center", color="#fb923c", fontsize=38, weight="bold")
axis.text(.795, .33, "peak allocated VRAM", ha="center", color="#cbd5e1", fontsize=17)
axis.text(.06, .12, "Transparent result: stable local training; language benchmarks remain near chance.",
          color="#e2e8f0", fontsize=20)
fig.savefig(MEDIA / "cover.png", dpi=180, facecolor=fig.get_facecolor()); plt.close(fig)

# Architecture and locality diagram.
fig = plt.figure(figsize=(16, 9), facecolor="#07111f")
axis = fig.add_axes([0, 0, 1, 1]); axis.set_axis_off()
axis.text(.055, .9, "ARCHITECTURE", color="#fb923c", fontsize=16, weight="bold")
axis.text(.055, .82, "Every learning signal is consumed where it is created", color="white", fontsize=34, weight="bold")
boxes = [(.04,.42,.16,.23,"Positive + corrupted\nbyte windows","#34d399"),
         (.27,.35,.34,.37,"20 sparse local blocks\n\ncausal stem  •  top-1 of 24 experts\npre-RMS goodness  •  local update","#38bdf8"),
         (.68,.42,.15,.23,"Local byte decoder\n\n264 active IDs","#fb923c"),
         (.87,.42,.10,.23,"Probability\n+ generation","#f8fafc")]
for x,y,w,h,label,color in boxes:
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012",transform=axis.transAxes,
                         facecolor="#13263d",edgecolor=color,linewidth=2)
    axis.add_patch(patch); axis.text(x+w/2,y+h/2,label,ha="center",va="center",color=color,fontsize=16,weight="bold")
for a,b in [((.20,.535),(.27,.535)),((.61,.535),(.68,.535)),((.83,.535),(.87,.535))]:
    axis.add_patch(FancyArrowPatch(a,b,transform=axis.transAxes,arrowstyle="-|>",mutation_scale=22,color="#94a3b8",lw=2))
local=FancyBboxPatch((.30,.15),.50,.11,boxstyle="round,pad=.01",transform=axis.transAxes,
                     facecolor="#2a1b12",edgecolor="#fb923c",linewidth=2)
axis.add_patch(local)
axis.text(.55,.205,"Decoder error updates decoder only — no downstream error returns to the backbone",
          ha="center",va="center",color="#fdba74",fontsize=17,weight="bold")
axis.add_patch(FancyArrowPatch((.755,.42),(.60,.26),transform=axis.transAxes,arrowstyle="-|>",mutation_scale=20,color="#fb923c",lw=2))
axis.text(.055,.06,"Normalized representations flow forward. Block-local derivatives are discarded after each update.",
          color="#cbd5e1",fontsize=18)
fig.savefig(MEDIA / "architecture.png", dpi=180, facecolor=fig.get_facecolor()); plt.close(fig)
print(json.dumps(summary, indent=2))
