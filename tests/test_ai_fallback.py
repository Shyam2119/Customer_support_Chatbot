import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.chatbot_engine as chatbot_engine  # noqa: E402


def test_ai_fallback_used_for_out_of_scope(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake_key')
    engine = chatbot_engine.ChatbotEngine()
    calls = []

    def fake_gemini(system_prompt, message):
        calls.append(message)
        return "This is an AI generated response about baking a cake."

    monkeypatch.setattr(engine, '_call_gemini', fake_gemini)
    result = engine.process_message("How do I bake a chocolate cake?")
    assert result['intent'] == 'ai_fallback'
    assert "baking a cake" in result['response']
    assert calls == ["How do I bake a chocolate cake?"]


def test_ai_not_called_for_known_intent(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake_key')
    engine = chatbot_engine.ChatbotEngine()
    monkeypatch.setattr(engine, '_call_gemini', lambda *a: (_ for _ in ()).throw(AssertionError("should not call AI")))
    result = engine.process_message("I want a refund for my broken headphones")
    assert result['intent'] == 'return_refund'


def test_ai_failure_falls_back_gracefully(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'fake_key')
    engine = chatbot_engine.ChatbotEngine()

    def boom(*a):
        raise RuntimeError("network down")

    monkeypatch.setattr(engine, '_call_gemini', boom)
    result = engine.process_message("How do I bake a chocolate cake?")
    assert result['intent'] == 'unknown'
