import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.chatbot_engine import get_engine

def test_nlp():
    engine = get_engine()
    test_queries = [
        "Where is my order?",
        "Track my order",
        "Hi there",
        "Tell me about Skyfii Pro",
        "Can I get a refund?",
        "Talk to a person"
    ]
    
    print("\n--- NLP Test Output ---")
    for q in test_queries:
        result = engine.process_message(q)
        print(f"Query: {q}")
        print(f"Top Intent: {result['intent']} ({result['confidence']:.1f}%)")
        print(f"Response (start): {result['response'][:60]}...")
        print("-" * 30)

if __name__ == "__main__":
    test_nlp()
