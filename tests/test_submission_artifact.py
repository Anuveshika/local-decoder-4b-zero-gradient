import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_six_executed_cells():
    notebook = json.loads((ROOT / "notebooks/executed_kaggle_run.ipynb").read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 6
    assert [cell["execution_count"] for cell in notebook["cells"]] == [1, 2, 3, 4, 5, 6]
    assert not any(output.get("output_type") == "error"
                   for cell in notebook["cells"] for output in cell.get("outputs", []))


def test_source_has_no_forbidden_training_api():
    source = (ROOT / "src/local_decoder_4b.py").read_text(encoding="utf-8")
    ast.parse(source)
    forbidden = ("backward(", "loss.backward", "torch.autograd", "torch.optim",
                 "optim.Adam", "optim.SGD", "jax.grad", "GradientTape")
    assert not [marker for marker in forbidden if marker in source]


def test_parameter_formula():
    vocab, width, depth, experts, hidden, active = 32768, 1024, 20, 24, 4096, 264
    per_expert = width*hidden + hidden*width + hidden + width
    per_block = 2*width*width + width + width*experts + experts + experts*per_expert
    backbone = vocab*width + depth*per_block
    total = backbone + width*active + active
    assert backbone == 4_104_999_392
    assert total == 4_105_269_992


def test_extracted_metrics_match_execution():
    metrics = json.loads((ROOT / "results/final_metrics.json").read_text(encoding="utf-8"))
    assert metrics["valid_run"] is True
    assert metrics["completed_under_180_minutes"] is True
    assert metrics["elapsed_seconds"] == 6343.889441096
    assert metrics["parameters"] == 4_105_269_992
