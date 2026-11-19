"""
evaluate.py - Train the intent classifier and measure it on held-out messages.

Two hand-written sets of realistic customer messages, none copied from the
training patterns (checked below):
  eval/dev_messages.jsonl   180 in-scope (10 per intent) + 25 out-of-scope.
                            Used to study errors, add training patterns and
                            choose the "unknown" threshold.
  eval/test_messages.jsonl  108 in-scope (6 per intent) + 15 out-of-scope.
                            Written before any tuning; used only for the
                            reported numbers.

Compares the logistic-regression classifier with the old keyword matcher.
Writes models/intent_clf.pkl and models/eval_report.json, and prints a table.

Usage: python scripts/evaluate.py
"""

import json
import os
import sys
import numpy as np
from sklearn.metrics import f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.intent_classifier import IntentClassifier, load_training_data  # noqa: E402

INTENTS = os.path.join(BASE_DIR, "data", "intents.json")
DEV = os.path.join(BASE_DIR, "eval", "dev_messages.jsonl")
TEST = os.path.join(BASE_DIR, "eval", "test_messages.jsonl")
OOS = "out_of_scope"


def _read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_splits():
    """dev: used for error analysis and to choose the threshold. test: written before any
    tuning and used only for the final numbers."""
    return _read(DEV), _read(TEST)


def score(predict, rows):
    """predict(text) -> intent or OOS. Returns metrics on in-scope and out-of-scope rows."""
    y_true = [r["intent"] for r in rows]
    y_pred = [predict(r["text"]) for r in rows]
    ins = [(t, p) for t, p in zip(y_true, y_pred) if t != OOS]
    oos = [(t, p) for t, p in zip(y_true, y_pred) if t == OOS]
    labels = sorted({t for t, _ in ins})
    return {
        "in_scope_accuracy": round(100 * np.mean([t == p for t, p in ins]), 1),
        "in_scope_macro_f1": round(100 * f1_score([t for t, _ in ins], [p for _, p in ins], labels=labels, average="macro", zero_division=0), 1),
        "out_of_scope_rejected": round(100 * np.mean([p == OOS for _, p in oos]), 1) if oos else None,
        "false_rejections": round(100 * np.mean([p == OOS for _, p in ins]), 1),
        "errors": [{"text": r["text"], "expected": t, "got": p} for r, t, p in zip(rows, y_true, y_pred) if t != p],
    }


def keyword_predict():
    """The previous approach (app.chatbot_engine keyword matcher), for comparison."""
    import pickle
    import subprocess
    kw_path = os.path.join(BASE_DIR, "models", "keyword_index.pkl")
    if not os.path.exists(kw_path):
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "setup.py")], check=True, capture_output=True)
    import re

    import nltk
    try:
        nltk.word_tokenize("test")
    except LookupError:  # tokenizer data not downloadable here; same split on letters/digits
        nltk.word_tokenize = lambda s: re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", s)
    from app.chatbot_engine import ChatbotEngine
    eng = ChatbotEngine.__new__(ChatbotEngine)
    with open(kw_path, "rb") as f:
        eng._keyword_index = pickle.load(f)

    def predict(text):
        res = eng._predict_keyword(text)
        # the app answers locally only above 0.5 confidence (see get_response)
        return res[0]["intent"] if res and res[0]["probability"] > 0.5 else OOS
    return predict


def main():
    train_texts, _ = load_training_data(INTENTS)
    dev, test = load_splits()
    overlap = {t.lower() for t in train_texts} & {r["text"].lower() for r in dev + test}
    assert not overlap, f"evaluation messages copied from training data: {overlap}"

    clf = IntentClassifier.train(INTENTS)

    # choose the threshold on dev: best balanced accuracy of in-scope vs out-of-scope, then accuracy
    best = None
    for t in np.arange(0.05, 0.80, 0.01):
        clf.threshold = float(t)
        s = score(lambda x: (clf.predict(x) or [{"intent": OOS}])[0]["intent"], dev)
        key = (s["in_scope_accuracy"] + s["out_of_scope_rejected"], s["in_scope_accuracy"])
        if best is None or key > best[0]:
            best = (key, float(t))
    clf.threshold = round(best[1], 2)

    report = {
        "train_patterns": len(train_texts),
        "dev_messages": len(dev),
        "test_messages": len(test),
        "threshold": clf.threshold,
        "logistic_regression": score(lambda x: (clf.predict(x) or [{"intent": OOS}])[0]["intent"], test),
    }
    try:
        report["keyword_baseline"] = score(keyword_predict(), test)
    except Exception as exc:  # baseline is optional
        report["keyword_baseline"] = {"skipped": str(exc)}

    clf.save()
    with open(os.path.join(BASE_DIR, "models", "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cols = ["in_scope_accuracy", "in_scope_macro_f1", "out_of_scope_rejected", "false_rejections"]
    n_oos = sum(r["intent"] == OOS for r in test)
    print(f"Test set: {len(test)} messages ({len(test) - n_oos} in scope, {n_oos} out of scope); threshold {clf.threshold}")
    print(f"{'model':<22}" + "".join(f"{c:>24}" for c in cols))
    for name in ("keyword_baseline", "logistic_regression"):
        r = report[name]
        if "skipped" in r:
            print(f"{name:<22} skipped: {r['skipped']}")
            continue
        print(f"{name:<22}" + "".join(f"{r[c]:>24}" for c in cols))
    print("\nLogistic regression errors on test:")
    for e in report["logistic_regression"]["errors"]:
        print(f"  {e['text']!r}: expected {e['expected']}, got {e['got']}")


if __name__ == "__main__":
    main()
