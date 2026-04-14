import sys
import os
from unittest.mock import MagicMock

# Mock google.generativeai before importing chatbot_engine
mock_genai = MagicMock()
sys.modules['google.generativeai'] = mock_genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app.chatbot_engine as chatbot_engine

def test_ai_fallback():
    # Set fake API key
    os.environ['GEMINI_API_KEY'] = 'fake_key'
    
    engine = chatbot_engine.ChatbotEngine()
    
    # Mock the AI response
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "This is an AI generated response about baking a cake."
    engine.ai_model = mock_model
    engine.ai_enabled = True
    
    # Test a query that should trigger fallback
    query = "How do I bake a chocolate cake?"
    result = engine.process_message(query)
    
    print(f"\nQuery: {query}")
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Response: {result['response']}")
    
    assert result['intent'] == 'ai_fallback'
    assert "baking a cake" in result['response']
    print("\n✅ AI Fallback Test Passed!")

if __name__ == "__main__":
    test_ai_fallback()
