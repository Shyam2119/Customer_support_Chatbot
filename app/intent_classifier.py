"""
intent_classifier.py - Intent classifier trained on data/intents.json.

TF-IDF over word 1-2 grams and character 2-5 grams, then logistic regression.
Character n-grams make it tolerant of typos ("refnd", "pasword") and word forms
("charged" / "charge") without needing NLTK data. It trains in under a second,
so it can be rebuilt on every deploy, and the saved model is a few hundred KB.

If the top probability is below the threshold the message is treated as
"unknown" (and can go to the Gemini fallback). The threshold is chosen on a
dev split of eval/test_messages.jsonl, never on the test split.
"""

import json
import os
import pickle
import re

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'intent_clf.pkl')
DEFAULT_THRESHOLD = 0.30


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b[a-z]{2,3}-\d{4,}\b|\b\d{5,}\b", " ordernum ", text)  # order ids carry no intent
    text = re.sub(r"[^a-z0-9%'\s]", " ", text)
    return " ".join(text.split())


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=20, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def load_training_data(intents_path: str):
    with open(intents_path, encoding="utf-8") as f:
        data = json.load(f)
    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(intent["tag"])
    return texts, labels


class IntentClassifier:
    def __init__(self, pipeline: Pipeline, threshold: float = DEFAULT_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def train(cls, intents_path: str, threshold: float = DEFAULT_THRESHOLD) -> "IntentClassifier":
        texts, labels = load_training_data(intents_path)
        pipeline = build_pipeline()
        pipeline.fit(texts, labels)
        return cls(pipeline, threshold)

    @property
    def classes(self) -> list:
        return [str(c) for c in self.pipeline.classes_]

    def predict_proba(self, text: str) -> list:
        probs = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.pipeline.classes_, probs), key=lambda x: x[1], reverse=True)
        return [{"intent": str(tag), "probability": float(p)} for tag, p in ranked]

    def predict(self, text: str, top_k: int = 3) -> list:
        """Top intents above the threshold; empty list means 'unknown'."""
        ranked = self.predict_proba(text)
        if not ranked or ranked[0]["probability"] < self.threshold:
            return []
        return [r for r in ranked[:top_k] if r["probability"] >= self.threshold]

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "threshold": self.threshold}, f)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "IntentClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)  # our own file, written by save()
        return cls(obj["pipeline"], obj["threshold"])
