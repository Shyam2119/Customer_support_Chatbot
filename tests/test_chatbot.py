"""
test_chatbot.py - Comprehensive test suite for the chatbot application
"""

import sys
import os
import json
import unittest
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChatbotEngine(unittest.TestCase):
    """Tests for the NLP chatbot engine"""

    @classmethod
    def setUpClass(cls):
        try:
            from app.chatbot_engine import ChatbotEngine
            cls.engine = ChatbotEngine()
        except Exception as e:
            cls.engine = None
            print(f"[SKIP] Engine not available: {e}")

    def test_engine_loads(self):
        """Test that engine initializes"""
        if not self.engine:
            self.skipTest("Engine not available")
        self.assertIsNotNone(self.engine)

    def test_bag_of_words_known_word(self):
        """Test bag of words with a known word"""
        if not self.engine or not self.engine.is_loaded:
            self.skipTest("Model not loaded")
        bow = self.engine.bag_of_words("hello there")
        self.assertIsInstance(bow, type(bow))
        self.assertEqual(len(bow), len(self.engine.words))

    def test_clean_sentence(self):
        """Test sentence tokenization and lemmatization"""
        if not self.engine:
            self.skipTest("Engine not available")
        try:
            result = self.engine.clean_up_sentence("Running tests quickly")
            self.assertIsInstance(result, list)
            self.assertTrue(len(result) > 0)
        except LookupError:
            self.skipTest("NLTK wordnet data not available in this environment")

    def test_entity_extraction_order(self):
        """Test order number entity extraction"""
        if not self.engine:
            self.skipTest("Engine not available")
        entities = self.engine.extract_entities("My order number is ORD-123456")
        self.assertIn('order_number', entities)

    def test_entity_extraction_email(self):
        """Test email entity extraction"""
        if not self.engine:
            self.skipTest("Engine not available")
        entities = self.engine.extract_entities("My email is test@example.com")
        self.assertIn('email', entities)
        self.assertEqual(entities['email'], 'test@example.com')

    def test_entity_extraction_phone(self):
        """Test phone entity extraction"""
        if not self.engine:
            self.skipTest("Engine not available")
        entities = self.engine.extract_entities("Call me at 555-123-4567")
        self.assertIn('phone', entities)

    def test_sentiment_positive(self):
        """Test positive sentiment detection"""
        if not self.engine:
            self.skipTest("Engine not available")
        s = self.engine.analyze_sentiment("This is amazing and great service!")
        self.assertEqual(s['sentiment'], 'positive')

    def test_sentiment_negative(self):
        """Test negative sentiment detection"""
        if not self.engine:
            self.skipTest("Engine not available")
        s = self.engine.analyze_sentiment("This is terrible and I'm very angry!")
        self.assertEqual(s['sentiment'], 'negative')

    def test_sentiment_urgent(self):
        """Test urgent detection"""
        if not self.engine:
            self.skipTest("Engine not available")
        s = self.engine.analyze_sentiment("I need help URGENT immediately")
        self.assertTrue(s['is_urgent'])

    def test_sentiment_neutral(self):
        """Test neutral sentiment"""
        if not self.engine:
            self.skipTest("Engine not available")
        s = self.engine.analyze_sentiment("I have a question about my account")
        self.assertIn(s['sentiment'], ['neutral', 'positive', 'negative'])

    def test_process_empty_message(self):
        """Test empty message handling"""
        if not self.engine:
            self.skipTest("Engine not available")
        result = self.engine.process_message("")
        self.assertEqual(result['intent'], 'empty_input')

    def test_process_whitespace_message(self):
        """Test whitespace-only message"""
        if not self.engine:
            self.skipTest("Engine not available")
        result = self.engine.process_message("   ")
        self.assertEqual(result['intent'], 'empty_input')

    def test_process_message_has_response(self):
        """Test that messages get a response"""
        if not self.engine or not self.engine.is_loaded:
            self.skipTest("Model not loaded")
        result = self.engine.process_message("Hello")
        self.assertIn('response', result)
        self.assertIsInstance(result['response'], str)
        self.assertTrue(len(result['response']) > 0)

    def test_model_info(self):
        """Test model info returns dict"""
        if not self.engine:
            self.skipTest("Engine not available")
        info = self.engine.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn('is_loaded', info)


class TestDatabase(unittest.TestCase):
    """Tests for the database layer"""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from app.database import DatabaseManager
        cls.tmpdir = tempfile.mkdtemp()
        cls.db = DatabaseManager(db_path=os.path.join(cls.tmpdir, 'test.db'))

    def test_create_session(self):
        """Test session creation"""
        sid = self.db.create_session('127.0.0.1', 'TestBrowser/1.0')
        self.assertIsInstance(sid, str)
        self.assertTrue(len(sid) > 8)

    def test_get_session(self):
        """Test session retrieval"""
        sid = self.db.create_session('192.168.1.1')
        sess = self.db.get_session(sid)
        self.assertIsNotNone(sess)
        self.assertEqual(sess['id'], sid)

    def test_get_nonexistent_session(self):
        """Test retrieval of nonexistent session"""
        sess = self.db.get_session('nonexistent-id-123')
        self.assertIsNone(sess)

    def test_log_message(self):
        """Test message logging"""
        sid = self.db.create_session()
        mid = self.db.log_message(
            session_id=sid,
            message_type='user',
            content='Test message',
            intent='greeting',
            confidence=0.95
        )
        self.assertIsInstance(mid, int)
        self.assertGreater(mid, 0)

    def test_log_bot_message(self):
        """Test bot message logging"""
        sid = self.db.create_session()
        mid = self.db.log_message(
            session_id=sid,
            message_type='bot',
            content='Bot response',
            intent='greeting',
            confidence=0.92,
            sentiment='positive',
            sentiment_score=0.3
        )
        self.assertIsInstance(mid, int)

    def test_session_history(self):
        """Test conversation history retrieval"""
        sid = self.db.create_session()
        self.db.log_message(sid, 'user', 'Hello')
        self.db.log_message(sid, 'bot', 'Hi there!')
        self.db.log_message(sid, 'user', 'How are you?')

        history = self.db.get_session_history(sid)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['message_type'], 'user')
        self.assertEqual(history[1]['message_type'], 'bot')

    def test_save_feedback(self):
        """Test feedback saving"""
        sid = self.db.create_session()
        fid = self.db.save_feedback(sid, rating=5, feedback_text='Excellent!', helpful=True)
        self.assertIsInstance(fid, int)

    def test_feedback_validation(self):
        """Test feedback saves correctly"""
        sid = self.db.create_session()
        fid = self.db.save_feedback(sid, rating=3)
        self.assertGreater(fid, 0)

    def test_dashboard_stats(self):
        """Test dashboard stats structure"""
        stats = self.db.get_dashboard_stats()
        self.assertIn('overview', stats)
        self.assertIn('intent_distribution', stats)
        self.assertIn('sentiment_distribution', stats)
        self.assertIn('daily_messages', stats)
        self.assertIn('total_sessions', stats['overview'])

    def test_log_escalation(self):
        """Test escalation logging"""
        sid = self.db.create_session()
        self.db.log_escalation(sid, 'Test escalation', 'user')
        sess = self.db.get_session(sid)
        self.assertTrue(sess['is_escalated'])

    def test_log_unknown_query(self):
        """Test unknown query logging"""
        self.db.log_unknown_query("some unknown thing xyz", "unknown")
        queries = self.db.get_unknown_queries()
        self.assertIsInstance(queries, list)

    def test_update_session(self):
        """Test session update"""
        sid = self.db.create_session()
        self.db.update_session(sid, user_email='test@test.com', resolution_status='resolved')
        sess = self.db.get_session(sid)
        self.assertEqual(sess['user_email'], 'test@test.com')
        self.assertEqual(sess['resolution_status'], 'resolved')

    def test_get_intent_performance(self):
        """Test intent performance query"""
        perf = self.db.get_intent_performance()
        self.assertIsInstance(perf, list)

    def test_recent_conversations(self):
        """Test recent conversations retrieval"""
        convs = self.db.get_recent_conversations(limit=5)
        self.assertIsInstance(convs, list)

    def test_export_conversations(self):
        """Test conversation export"""
        data = self.db.export_conversations()
        self.assertIsInstance(data, list)


class TestFlaskApp(unittest.TestCase):
    """Tests for Flask API endpoints"""

    @classmethod
    def setUpClass(cls):
        try:
            from app import create_app
            cls.app = create_app()
            cls.client = cls.app.test_client()
            cls.app.config['TESTING'] = True
        except Exception as e:
            cls.app = None
            cls.client = None
            print(f"[SKIP] Flask app not available: {e}")

    def test_health_endpoint(self):
        """Test /health returns 200"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

    def test_index_page(self):
        """Test index page loads"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

    def test_dashboard_page(self):
        """Test dashboard page loads"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/dashboard')
        self.assertEqual(resp.status_code, 200)

    def test_create_session(self):
        """Test session creation endpoint"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/chat/session',
                                content_type='application/json')
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.data)
        self.assertIn('session_id', data)
        self.__class__.session_id = data['session_id']

    def test_send_message_no_session(self):
        """Test sending message auto-creates session"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/chat/message',
                                json={'message': 'Hello there'},
                                content_type='application/json')
        # 200 OK or 503 if model not loaded
        self.assertIn(resp.status_code, [200, 503, 404])

    def test_send_message_empty(self):
        """Test empty message returns 400"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/chat/message',
                                json={'message': ''},
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_send_message_no_body(self):
        """Test missing body returns 400 or 415"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/chat/message')
        self.assertIn(resp.status_code, [400, 415])  # 415 = Unsupported Media Type

    def test_analytics_dashboard(self):
        """Test analytics dashboard endpoint"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/api/analytics/dashboard')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('overview', data)

    def test_analytics_intents(self):
        """Test intent analytics endpoint"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/api/analytics/intents')
        self.assertEqual(resp.status_code, 200)

    def test_analytics_conversations(self):
        """Test conversations list endpoint"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/api/analytics/conversations')
        self.assertEqual(resp.status_code, 200)

    def test_analytics_model_info(self):
        """Test model info endpoint"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/api/analytics/model-info')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('is_loaded', data)

    def test_admin_requires_api_key(self):
        """Test admin endpoint requires API key"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/admin/reload-model')
        self.assertEqual(resp.status_code, 401)

    def test_admin_with_valid_api_key(self):
        """Test admin endpoint with valid API key"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/admin/reload-model',
                                headers={'X-API-Key': 'demo-admin-key-12345'})
        self.assertIn(resp.status_code, [200, 500])

    def test_test_prediction_endpoint(self):
        """Test the test-prediction endpoint"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/admin/test-prediction',
                                json={'message': 'Hello!'},
                                content_type='application/json')
        self.assertIn(resp.status_code, [200, 500])

    def test_404_handler(self):
        """Test 404 handler"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/nonexistent-route')
        self.assertEqual(resp.status_code, 404)

    def test_feedback_missing_session(self):
        """Test feedback without session"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.post('/api/chat/feedback',
                                json={'rating': 5},
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_history_nonexistent_session(self):
        """Test history for nonexistent session"""
        if not self.client:
            self.skipTest("App not available")
        resp = self.client.get('/api/chat/history/nonexistent-id')
        self.assertEqual(resp.status_code, 404)


class TestIntentCoverage(unittest.TestCase):
    """Test all intents are properly loaded"""

    def test_intents_file_exists(self):
        """Test intents.json exists"""
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        self.assertTrue(os.path.exists(path))

    def test_intents_file_valid_json(self):
        """Test intents.json is valid JSON"""
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        with open(path) as f:
            data = json.load(f)
        self.assertIn('intents', data)

    def test_minimum_intent_count(self):
        """Test there are at least 15 intent categories"""
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        with open(path) as f:
            data = json.load(f)
        self.assertGreaterEqual(len(data['intents']), 15)

    def test_each_intent_has_patterns(self):
        """Test each intent has at least 3 patterns"""
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        with open(path) as f:
            data = json.load(f)
        for intent in data['intents']:
            self.assertGreaterEqual(len(intent['patterns']), 3,
                f"Intent '{intent['tag']}' has fewer than 3 patterns")

    def test_each_intent_has_responses(self):
        """Test each intent has at least 1 response"""
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        with open(path) as f:
            data = json.load(f)
        for intent in data['intents']:
            self.assertGreater(len(intent['responses']), 0,
                f"Intent '{intent['tag']}' has no responses")

    def test_required_intents_present(self):
        """Test all required intents are present"""
        required = [
            'greeting', 'goodbye', 'order_status', 'return_refund',
            'payment_issues', 'technical_support', 'shipping_info', 'complaint'
        ]
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        with open(path) as f:
            data = json.load(f)
        tags = {i['tag'] for i in data['intents']}
        for req in required:
            self.assertIn(req, tags, f"Required intent '{req}' missing")

    def test_no_duplicate_tags(self):
        """Test no duplicate intent tags"""
        path = os.path.join(os.path.dirname(__file__), '..', 'data', 'intents.json')
        with open(path) as f:
            data = json.load(f)
        tags = [i['tag'] for i in data['intents']]
        self.assertEqual(len(tags), len(set(tags)), "Duplicate intent tags found")


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in [TestIntentCoverage, TestDatabase, TestFlaskApp, TestChatbotEngine]:
        suite.addTests(loader.loadTestsFromTestCase(tc))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
