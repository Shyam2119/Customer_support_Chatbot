"""
chatbot_engine.py - Core NLP inference engine for intent classification and response generation
"""

import json
import pickle
import numpy as np
import random
import os
import re
import logging
from datetime import datetime
from typing import Optional

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Safe lemmatizer wrapper that falls back to basic stemming if wordnet unavailable
class SafeLemmatizer:
    def __init__(self):
        self._wn = None
        try:
            l = WordNetLemmatizer()
            l.lemmatize('test')  # verify it actually works
            self._wn = l
        except Exception:
            pass
    
    def lemmatize(self, word, pos='n'):
        if self._wn:
            try:
                return self._wn.lemmatize(word, pos)
            except Exception:
                pass
        # Fallback: basic suffix stripping
        for suffix in ['ing', 'tion', 'ions', 'ed', 'es', 's']:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')

ERROR_THRESHOLD = 0.35

lemmatizer = SafeLemmatizer()


class ChatbotEngine:
    """
    Core NLP chatbot engine for intent classification and response generation.
    Handles model loading, preprocessing, prediction, and entity extraction.
    """

    def __init__(self):
        self.model = None
        self.words = []
        self.classes = []
        self.intents = {}
        self.is_loaded = False
        self._keyword_index = {}
        self._use_keyword_fallback = False
        self._load_resources()

    def _load_resources(self):
        """Load model, vocabulary, and intents data"""
        model_path = os.path.join(MODEL_DIR, 'chatbot_model.h5')

        # Skip TensorFlow import entirely if no model file exists (prevents 10-min hang)
        if not os.path.exists(model_path):
            logger.info("No TF model found — loading keyword fallback directly.")
            self._load_keyword_fallback()
            return

        try:
            import tensorflow as tf

            words_path   = os.path.join(MODEL_DIR, 'words.pkl')
            classes_path = os.path.join(MODEL_DIR, 'classes.pkl')
            intents_path = DATA_PATH

            self.model = tf.keras.models.load_model(model_path)

            with open(words_path, 'rb') as f:
                self.words = pickle.load(f)

            with open(classes_path, 'rb') as f:
                self.classes = pickle.load(f)

            with open(intents_path, 'r') as f:
                data = json.load(f)
                self.intents = {i['tag']: i for i in data['intents']}

            self.is_loaded = True
            logger.info(f"TF model loaded: {len(self.classes)} intents, {len(self.words)} words")

        except Exception as e:
            logger.error(f"Error loading TF model: {e}")
            logger.info("Falling back to keyword model...")
            self._load_keyword_fallback()

    def _load_keyword_fallback(self):
        """Load keyword-based model as fallback when TF model is unavailable"""
        try:
            words_path = os.path.join(MODEL_DIR, 'words.pkl')
            classes_path = os.path.join(MODEL_DIR, 'classes.pkl')
            kw_path = os.path.join(MODEL_DIR, 'keyword_index.pkl')
            intents_path = DATA_PATH

            if not os.path.exists(kw_path):
                logger.warning("No keyword_index.pkl found. Run scripts/quick_train.py first.")
                return

            with open(kw_path, 'rb') as f:
                self._keyword_index = pickle.load(f)

            if os.path.exists(words_path):
                with open(words_path, 'rb') as f:
                    self.words = pickle.load(f)
            if os.path.exists(classes_path):
                with open(classes_path, 'rb') as f:
                    self.classes = pickle.load(f)

            with open(intents_path, 'r') as f:
                data = json.load(f)
                self.intents = {i['tag']: i for i in data['intents']}
                if not self.classes:
                    self.classes = sorted([i['tag'] for i in data['intents']])

            self.is_loaded = True
            self._use_keyword_fallback = True
            logger.info(f"Keyword fallback loaded: {len(self.classes)} intents")

        except Exception as e:
            logger.error(f"Keyword fallback also failed: {e}")

    def clean_up_sentence(self, sentence: str) -> list:
        """Tokenize and lemmatize input sentence"""
        sentence_words = nltk.word_tokenize(sentence.lower())
        sentence_words = [lemmatizer.lemmatize(w) for w in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence: str) -> np.ndarray:
        """Convert sentence to bag-of-words vector"""
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence: str) -> list:
        """Predict intent class with confidence scores"""
        if not self.is_loaded:
            return []

        # Keyword fallback when TF model not available
        if self._use_keyword_fallback or self.model is None:
            return self._predict_keyword(sentence)

        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]

        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                'intent': self.classes[r[0]],
                'probability': float(r[1])
            }
            for r in results
        ]

    def _predict_keyword(self, sentence: str) -> list:
        """Keyword-overlap intent prediction fallback"""
        if not self._keyword_index:
            return []
        tokens = set(
            lemmatizer.lemmatize(w.lower())
            for w in nltk.word_tokenize(sentence)
            if w.isalpha()
        )
        scores = {}
        for tag, kws in self._keyword_index.items():
            kw_set = set(kws)
            matches = len(tokens & kw_set)
            if matches > 0:
                scores[tag] = matches / (len(kw_set) + 1)
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {'intent': tag, 'probability': min(score + 0.35, 0.95)}
            for tag, score in sorted_tags[:3]
        ]

    def get_response(self, intents_list: list, sentence: str = "") -> dict:
        """Get response based on predicted intents"""
        if not intents_list:
            return {
                'response': "I'm sorry, I didn't quite understand that. Could you rephrase your question? You can also type 'help' to see what I can assist with.",
                'intent': 'unknown',
                'confidence': 0.0
            }

        top_intent = intents_list[0]
        tag = top_intent['intent']
        confidence = top_intent['probability']

        if tag in self.intents:
            responses = self.intents[tag]['responses']
            response = random.choice(responses)

            # Personalize response with entities if detected
            entities = self.extract_entities(sentence)
            if entities.get('order_number') and tag == 'order_status':
                response = f"Looking up order #{entities['order_number']}... " + response

            return {
                'response': response,
                'intent': tag,
                'confidence': confidence,
                'entities': entities,
                'context': self.intents[tag].get('context_set', '')
            }

        return {
            'response': "I can help you with orders, returns, payments, technical support, and more. What do you need assistance with?",
            'intent': 'fallback',
            'confidence': 0.0
        }

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from user input"""
        entities = {}

        # Order number patterns
        order_patterns = [
            r'\b(?:order|#|no\.?|number)\s*[:#]?\s*([A-Z0-9]{6,12})\b',
            r'\b([A-Z]{2,3}-\d{5,10})\b',
            r'\b(\d{6,12})\b'
        ]
        for pattern in order_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['order_number'] = match.group(1).upper()
                break

        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            entities['email'] = email_match.group()

        # Phone number extraction
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            entities['phone'] = phone_match.group()

        # Dollar amount extraction
        amount_pattern = r'\$\s?(\d+(?:\.\d{2})?)'
        amount_match = re.search(amount_pattern, text)
        if amount_match:
            entities['amount'] = amount_match.group(1)

        # Date extraction
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        date_match = re.search(date_pattern, text)
        if date_match:
            entities['date'] = date_match.group()

        return entities

    def analyze_sentiment(self, text: str) -> dict:
        """Simple rule-based sentiment analysis"""
        text_lower = text.lower()

        positive_words = [
            'great', 'excellent', 'awesome', 'amazing', 'perfect', 'wonderful',
            'fantastic', 'love', 'good', 'thank', 'happy', 'pleased', 'satisfied',
            'helpful', 'appreciate', 'best'
        ]
        negative_words = [
            'terrible', 'awful', 'bad', 'worst', 'hate', 'angry', 'frustrated',
            'disappointed', 'useless', 'broken', 'failed', 'horrible', 'furious',
            'unacceptable', 'disgusted', 'upset', 'annoyed'
        ]
        urgent_words = [
            'urgent', 'asap', 'immediately', 'emergency', 'critical', 'now',
            'right away', 'quickly', 'fast', 'important', 'serious'
        ]

        words = text_lower.split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        is_urgent = any(w in text_lower for w in urgent_words)

        if neg_count > pos_count:
            sentiment = 'negative'
            score = -neg_count / (len(words) + 1)
        elif pos_count > neg_count:
            sentiment = 'positive'
            score = pos_count / (len(words) + 1)
        else:
            sentiment = 'neutral'
            score = 0.0

        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'is_urgent': is_urgent,
            'positive_count': pos_count,
            'negative_count': neg_count
        }

    def process_message(self, message: str, session_id: str = None) -> dict:
        """
        Full pipeline: preprocess → predict → respond → analyze
        Returns complete response object
        """
        if not message or not message.strip():
            return {
                'response': "Please type a message so I can help you!",
                'intent': 'empty_input',
                'confidence': 0.0,
                'sentiment': {'sentiment': 'neutral', 'score': 0.0, 'is_urgent': False}
            }

        message = message.strip()

        # Sentiment analysis
        sentiment = self.analyze_sentiment(message)

        # Intent prediction
        intents_list = self.predict_class(message)

        # Get response
        result = self.get_response(intents_list, message)

        # If very negative sentiment and not complaint intent, acknowledge it
        if sentiment['sentiment'] == 'negative' and result['intent'] not in ['complaint']:
            result['response'] = "I sense some frustration, and I truly apologize. " + result['response']

        # If urgent, prioritize
        if sentiment['is_urgent']:
            result['response'] = "⚡ I understand this is urgent. " + result['response']

        result['sentiment'] = sentiment
        result['all_intents'] = intents_list[:3]  # Top 3 predictions
        result['timestamp'] = datetime.now().isoformat()
        result['message'] = message

        return result

    def get_model_info(self) -> dict:
        """Return model metadata"""
        metadata_path = os.path.join(MODEL_DIR, 'training_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        return {
            'is_loaded': self.is_loaded,
            'num_intents': len(self.classes),
            'num_words': len(self.words),
            'intents': self.classes,
            'metadata': metadata
        }

    def reload_model(self):
        """Reload the model from disk"""
        self.is_loaded = False
        self._load_resources()
        return self.is_loaded


# Singleton instance
_engine_instance: Optional[ChatbotEngine] = None


def get_engine() -> ChatbotEngine:
    """Get or create singleton chatbot engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatbotEngine()
    return _engine_instance
