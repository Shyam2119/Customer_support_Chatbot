#!/usr/bin/env bash
# build.sh — Render build script
# Runs once at deploy time to set up the app environment.
set -o errexit   # exit on any error

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🧠 Building keyword NLP model + seeding database..."
python setup.py

echo "🎯 Training the intent classifier and measuring it on held-out messages..."
python scripts/evaluate.py

echo "✅ Build complete!"
