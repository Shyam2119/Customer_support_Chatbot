"""Quality gate: fail the build if intent accuracy on the held-out test set drops."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_held_out_accuracy_gate():
    subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, "scripts", "evaluate.py")], check=True, capture_output=True)
    with open(os.path.join(ROOT, "models", "eval_report.json")) as f:
        report = json.load(f)
    lr = report["logistic_regression"]
    assert lr["in_scope_accuracy"] >= 80
    assert lr["out_of_scope_rejected"] >= 70
    assert lr["false_rejections"] <= 15
    base = report["keyword_baseline"]
    if "in_scope_accuracy" in base:
        assert lr["in_scope_accuracy"] > base["in_scope_accuracy"]
