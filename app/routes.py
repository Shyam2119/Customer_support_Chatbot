"""
routes.py - Flask REST API routes for the customer support chatbot
"""

import time
import json
import logging
import os
from datetime import datetime
from functools import wraps

from flask import Blueprint, request, jsonify, render_template, session, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app.chatbot_engine import get_engine
from app.database import get_db

logger = logging.getLogger(__name__)

# Blueprints
chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')
main_bp = Blueprint('main', __name__)


def require_api_key(f):
    """Simple API key authentication decorator"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        valid_key = os.environ.get('ADMIN_API_KEY', 'demo-admin-key-12345')
        if api_key != valid_key:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────── Main Routes ───────────────────────────

@main_bp.route('/')
def index():
    """Serve the chat interface"""
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    """Serve the analytics dashboard"""
    return render_template('dashboard.html')

@main_bp.route('/health')
def health():
    """Health check endpoint"""
    engine = get_engine()
    return jsonify({
        'status': 'healthy',
        'model_loaded': engine.is_loaded,
        'timestamp': datetime.now().isoformat()
    })


# ─────────────────────────── Chat Routes ───────────────────────────

@chat_bp.route('/session', methods=['POST'])
def create_session():
    """Create a new chat session"""
    db = get_db()
    ip = request.remote_addr
    ua = request.headers.get('User-Agent', '')
    session_id = db.create_session(ip_address=ip, user_agent=ua)

    return jsonify({
        'session_id': session_id,
        'created_at': datetime.now().isoformat(),
        'message': 'Session created successfully'
    }), 201


@chat_bp.route('/message', methods=['POST'])
def send_message():
    """
    Process a chat message and return bot response.

    Request body:
    {
        "session_id": "uuid",
        "message": "user's message text"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    message = data.get('message', '').strip()
    session_id = data.get('session_id')

    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    if not session_id:
        # Auto-create session if not provided
        db = get_db()
        session_id = db.create_session(
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')
        )

    db = get_db()

    # Verify session exists
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({'error': 'Session not found'}), 404

    # Log user message
    db.log_message(
        session_id=session_id,
        message_type='user',
        content=message
    )

    # Process with chatbot engine
    start_time = time.time()
    engine = get_engine()
    result = engine.process_message(message, session_id)
    response_time_ms = int((time.time() - start_time) * 1000)

    # Log bot response
    sentiment = result.get('sentiment', {})
    entities = result.get('entities', {})

    db.log_message(
        session_id=session_id,
        message_type='bot',
        content=result['response'],
        intent=result.get('intent'),
        confidence=result.get('confidence'),
        sentiment=sentiment.get('sentiment'),
        sentiment_score=sentiment.get('score'),
        is_urgent=sentiment.get('is_urgent', False),
        entities=entities,
        response_time_ms=response_time_ms
    )

    # Log unknown queries for analysis
    if result.get('confidence', 1.0) < 0.4 or result.get('intent') in ['unknown', 'fallback']:
        db.log_unknown_query(message, result.get('intent'))

    # Auto-escalate if needed
    if result.get('intent') == 'human_agent':
        db.log_escalation(session_id, 'User requested human agent', 'user')

    return jsonify({
        'session_id': session_id,
        'response': result['response'],
        'intent': result.get('intent'),
        'confidence': round(result.get('confidence', 0) * 100, 1),
        'sentiment': result.get('sentiment'),
        'entities': result.get('entities', {}),
        'response_time_ms': response_time_ms,
        'timestamp': result.get('timestamp')
    })


@chat_bp.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get conversation history for a session"""
    db = get_db()
    sess = db.get_session(session_id)
    if not sess:
        return jsonify({'error': 'Session not found'}), 404

    history = db.get_session_history(session_id)
    return jsonify({
        'session_id': session_id,
        'messages': history,
        'total': len(history)
    })


@chat_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a conversation"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body required'}), 400

    session_id = data.get('session_id')
    rating = data.get('rating')
    feedback_text = data.get('feedback_text', '')
    intent_tag = data.get('intent_tag')
    helpful = data.get('helpful')

    if not session_id:
        return jsonify({'error': 'session_id required'}), 400
    if rating is not None and (not isinstance(rating, int) or not (1 <= rating <= 5)):
        return jsonify({'error': 'Rating must be integer between 1 and 5'}), 400

    db = get_db()
    feedback_id = db.save_feedback(
        session_id=session_id,
        rating=rating,
        feedback_text=feedback_text,
        intent_tag=intent_tag,
        helpful=helpful
    )

    return jsonify({
        'feedback_id': feedback_id,
        'message': 'Feedback submitted successfully'
    }), 201


@chat_bp.route('/typing', methods=['POST'])
def typing_indicator():
    """Simulate processing delay for typing indicator"""
    return jsonify({'status': 'typing'}), 200


# ─────────────────────────── Analytics Routes ───────────────────────────

@analytics_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Get comprehensive dashboard statistics"""
    db = get_db()
    stats = db.get_dashboard_stats()
    return jsonify(stats)


@analytics_bp.route('/intents', methods=['GET'])
def get_intent_analytics():
    """Get per-intent performance metrics"""
    db = get_db()
    data = db.get_intent_performance()
    return jsonify({
        'intent_performance': data,
        'total_intents': len(data)
    })


@analytics_bp.route('/conversations', methods=['GET'])
def get_conversations():
    """Get recent conversations list"""
    limit = int(request.args.get('limit', 20))
    offset = int(request.args.get('offset', 0))
    db = get_db()
    conversations = db.get_recent_conversations(limit=limit, offset=offset)
    return jsonify({
        'conversations': conversations,
        'limit': limit,
        'offset': offset
    })


@analytics_bp.route('/unknown-queries', methods=['GET'])
def get_unknown_queries():
    """Get frequently unrecognized queries"""
    db = get_db()
    queries = db.get_unknown_queries(limit=20)
    return jsonify({
        'unknown_queries': queries,
        'total': len(queries),
        'message': 'These queries had low confidence scores and may need new intent categories'
    })


@analytics_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    engine = get_engine()
    info = engine.get_model_info()
    return jsonify(info)


@analytics_bp.route('/export', methods=['GET'])
def export_data():
    """Export conversation data as JSON"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    db = get_db()
    data = db.export_conversations(start_date, end_date)
    return jsonify({
        'data': data,
        'total': len(data),
        'exported_at': datetime.now().isoformat()
    })


# ─────────────────────────── Admin Routes ───────────────────────────

@admin_bp.route('/reload-model', methods=['POST'])
@require_api_key
def reload_model():
    """Reload the chatbot model"""
    engine = get_engine()
    success = engine.reload_model()
    return jsonify({
        'success': success,
        'message': 'Model reloaded successfully' if success else 'Failed to reload model',
        'timestamp': datetime.now().isoformat()
    })


@admin_bp.route('/sessions/<session_id>/escalate', methods=['POST'])
@require_api_key
def escalate_session(session_id):
    """Manually escalate a session"""
    data = request.get_json() or {}
    reason = data.get('reason', 'Manual escalation by admin')
    db = get_db()
    db.log_escalation(session_id, reason, 'admin')
    db.update_session(session_id, resolution_status='escalated')
    return jsonify({'message': f'Session {session_id} escalated'})


@admin_bp.route('/sessions/<session_id>/resolve', methods=['POST'])
@require_api_key
def resolve_session(session_id):
    """Mark a session as resolved"""
    db = get_db()
    db.update_session(session_id, resolution_status='resolved')
    return jsonify({'message': f'Session {session_id} resolved'})


@admin_bp.route('/test-prediction', methods=['POST'])
def test_prediction():
    """Test the model with a message without logging"""
    data = request.get_json()
    if not data or not data.get('message'):
        return jsonify({'error': 'Message required'}), 400

    engine = get_engine()
    result = engine.process_message(data['message'])

    return jsonify({
        'input': data['message'],
        'top_intent': result.get('intent'),
        'confidence': round(result.get('confidence', 0) * 100, 2),
        'response': result.get('response'),
        'all_intents': result.get('all_intents', []),
        'sentiment': result.get('sentiment'),
        'entities': result.get('entities', {})
    })
