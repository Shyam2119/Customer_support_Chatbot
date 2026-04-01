"""
database.py - SQLite database models and manager for conversation logs, analytics, and user sessions
"""

import sqlite3
import json
import os
import logging
import uuid
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'chatbot.db')


class DatabaseManager:
    """
    Manages SQLite database for storing conversation logs,
    sessions, feedback, and analytics data.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def init_database(self):
        """Initialize all database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    total_messages INTEGER DEFAULT 0,
                    is_escalated BOOLEAN DEFAULT FALSE,
                    resolution_status TEXT DEFAULT 'open',
                    user_email TEXT,
                    user_name TEXT,
                    metadata TEXT
                )
            ''')

            # Conversation logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_type TEXT NOT NULL CHECK(message_type IN ('user', 'bot')),
                    content TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL,
                    sentiment TEXT,
                    sentiment_score REAL,
                    is_urgent BOOLEAN DEFAULT FALSE,
                    entities TEXT,
                    response_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    conversation_id INTEGER,
                    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
                    feedback_text TEXT,
                    intent_tag TEXT,
                    helpful BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Intent analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intent_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_tag TEXT NOT NULL,
                    total_hits INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    positive_feedback INTEGER DEFAULT 0,
                    negative_feedback INTEGER DEFAULT 0,
                    escalation_rate REAL DEFAULT 0.0,
                    date DATE NOT NULL,
                    UNIQUE(intent_tag, date)
                )
            ''')

            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_predictions INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    low_confidence_rate REAL DEFAULT 0.0,
                    fallback_rate REAL DEFAULT 0.0,
                    avg_response_time_ms REAL DEFAULT 0.0,
                    unique_sessions INTEGER DEFAULT 0,
                    date DATE NOT NULL UNIQUE
                )
            ''')

            # Escalations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    reason TEXT,
                    triggered_by TEXT,
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # FAQ suggestions table (auto-generated from low-confidence queries)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unknown_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    best_guess_intent TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_session ON conversation_logs(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_intent ON conversation_logs(intent)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_created ON conversation_logs(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)')

            logger.info("Database initialized successfully")

    # ─────────────────── Session Methods ───────────────────

    def create_session(self, ip_address: str = None, user_agent: str = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        with self.get_connection() as conn:
            conn.execute(
                '''INSERT INTO sessions (id, ip_address, user_agent)
                   VALUES (?, ?, ?)''',
                (session_id, ip_address, user_agent)
            )
        return session_id

    def update_session(self, session_id: str, **kwargs):
        """Update session fields"""
        if not kwargs:
            return
        fields = ', '.join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [session_id]
        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE sessions SET {fields}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                values
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM sessions WHERE id = ?', (session_id,)
            ).fetchone()
            return dict(row) if row else None

    # ─────────────────── Conversation Log Methods ───────────────────

    def log_message(self, session_id: str, message_type: str, content: str,
                    intent: str = None, confidence: float = None,
                    sentiment: str = None, sentiment_score: float = None,
                    is_urgent: bool = False, entities: dict = None,
                    response_time_ms: int = None) -> int:
        """Log a conversation message"""
        entities_json = json.dumps(entities) if entities else None
        with self.get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO conversation_logs
                   (session_id, message_type, content, intent, confidence,
                    sentiment, sentiment_score, is_urgent, entities, response_time_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (session_id, message_type, content, intent, confidence,
                 sentiment, sentiment_score, is_urgent, entities_json, response_time_ms)
            )
            # Update session message count
            conn.execute(
                'UPDATE sessions SET total_messages = total_messages + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (session_id,)
            )
            # Update intent analytics
            self._update_intent_analytics(conn, intent, confidence)
            return cursor.lastrowid

    def _update_intent_analytics(self, conn, intent: str, confidence: float):
        """Update daily intent analytics"""
        if not intent:
            return
        today = datetime.now().date().isoformat()
        conn.execute(
            '''INSERT INTO intent_analytics (intent_tag, total_hits, avg_confidence, date)
               VALUES (?, 1, ?, ?)
               ON CONFLICT(intent_tag, date) DO UPDATE SET
               total_hits = total_hits + 1,
               avg_confidence = (avg_confidence * total_hits + excluded.avg_confidence) / (total_hits + 1)''',
            (intent, confidence or 0, today)
        )

    def get_session_history(self, session_id: str) -> List[dict]:
        """Get full conversation history for a session"""
        with self.get_connection() as conn:
            rows = conn.execute(
                '''SELECT * FROM conversation_logs
                   WHERE session_id = ?
                   ORDER BY created_at ASC''',
                (session_id,)
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d.get('entities'):
                    try:
                        d['entities'] = json.loads(d['entities'])
                    except:
                        pass
                result.append(d)
            return result

    # ─────────────────── Feedback Methods ───────────────────

    def save_feedback(self, session_id: str, rating: int, feedback_text: str = None,
                      intent_tag: str = None, helpful: bool = None) -> int:
        """Save user feedback"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO feedback (session_id, rating, feedback_text, intent_tag, helpful)
                   VALUES (?, ?, ?, ?, ?)''',
                (session_id, rating, feedback_text, intent_tag, helpful)
            )
            return cursor.lastrowid

    # ─────────────────── Analytics Methods ───────────────────

    def get_dashboard_stats(self) -> dict:
        """Get comprehensive dashboard statistics"""
        with self.get_connection() as conn:
            # Total conversations
            total_sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]

            # Total messages
            total_messages = conn.execute('SELECT COUNT(*) FROM conversation_logs WHERE message_type = "user"').fetchone()[0]

            # Today's stats
            today = datetime.now().date().isoformat()
            today_sessions = conn.execute(
                'SELECT COUNT(*) FROM sessions WHERE DATE(created_at) = ?', (today,)
            ).fetchone()[0]
            today_messages = conn.execute(
                '''SELECT COUNT(*) FROM conversation_logs
                   WHERE message_type = "user" AND DATE(created_at) = ?''', (today,)
            ).fetchone()[0]

            # Average confidence
            avg_confidence = conn.execute(
                'SELECT AVG(confidence) FROM conversation_logs WHERE confidence IS NOT NULL'
            ).fetchone()[0] or 0

            # Intent distribution
            intent_dist = conn.execute(
                '''SELECT intent, COUNT(*) as count, AVG(confidence) as avg_conf
                   FROM conversation_logs
                   WHERE intent IS NOT NULL AND message_type = 'bot'
                   GROUP BY intent
                   ORDER BY count DESC
                   LIMIT 10'''
            ).fetchall()

            # Sentiment distribution
            sentiment_dist = conn.execute(
                '''SELECT sentiment, COUNT(*) as count
                   FROM conversation_logs
                   WHERE sentiment IS NOT NULL
                   GROUP BY sentiment'''
            ).fetchall()

            # Average rating
            avg_rating = conn.execute('SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL').fetchone()[0] or 0

            # Total feedback count
            total_feedback = conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]

            # Escalation count
            total_escalations = conn.execute('SELECT COUNT(*) FROM escalations').fetchone()[0]

            # Urgent messages
            urgent_count = conn.execute(
                'SELECT COUNT(*) FROM conversation_logs WHERE is_urgent = TRUE'
            ).fetchone()[0]

            # Messages over last 7 days
            daily_messages = conn.execute(
                '''SELECT DATE(created_at) as date, COUNT(*) as count
                   FROM conversation_logs
                   WHERE message_type = 'user' AND created_at >= datetime('now', '-7 days')
                   GROUP BY DATE(created_at)
                   ORDER BY date ASC'''
            ).fetchall()

            # Fallback rate
            fallback_count = conn.execute(
                "SELECT COUNT(*) FROM conversation_logs WHERE intent IN ('unknown', 'fallback')"
            ).fetchone()[0]
            fallback_rate = (fallback_count / total_messages * 100) if total_messages > 0 else 0

            # Response time stats
            avg_response_time = conn.execute(
                'SELECT AVG(response_time_ms) FROM conversation_logs WHERE response_time_ms IS NOT NULL'
            ).fetchone()[0] or 0

            # Low confidence rate
            low_conf_count = conn.execute(
                'SELECT COUNT(*) FROM conversation_logs WHERE confidence < 0.5 AND confidence IS NOT NULL'
            ).fetchone()[0]
            total_with_conf = conn.execute(
                'SELECT COUNT(*) FROM conversation_logs WHERE confidence IS NOT NULL'
            ).fetchone()[0]
            low_conf_rate = (low_conf_count / total_with_conf * 100) if total_with_conf > 0 else 0

            return {
                'overview': {
                    'total_sessions': total_sessions,
                    'total_messages': total_messages,
                    'today_sessions': today_sessions,
                    'today_messages': today_messages,
                    'avg_confidence': round(avg_confidence * 100, 2),
                    'avg_rating': round(avg_rating, 2),
                    'total_feedback': total_feedback,
                    'total_escalations': total_escalations,
                    'urgent_messages': urgent_count,
                    'fallback_rate': round(fallback_rate, 2),
                    'avg_response_time_ms': round(avg_response_time, 2),
                    'low_confidence_rate': round(low_conf_rate, 2)
                },
                'intent_distribution': [
                    {'intent': r[0], 'count': r[1], 'avg_confidence': round((r[2] or 0) * 100, 2)}
                    for r in intent_dist
                ],
                'sentiment_distribution': [
                    {'sentiment': r[0], 'count': r[1]}
                    for r in sentiment_dist
                ],
                'daily_messages': [
                    {'date': r[0], 'count': r[1]}
                    for r in daily_messages
                ]
            }

    def get_recent_conversations(self, limit: int = 20, offset: int = 0) -> List[dict]:
        """Get recent conversation sessions"""
        with self.get_connection() as conn:
            rows = conn.execute(
                '''SELECT s.id, s.created_at, s.total_messages, s.resolution_status,
                          s.is_escalated, s.user_email,
                          COUNT(DISTINCT f.id) as feedback_count,
                          AVG(f.rating) as avg_rating,
                          MIN(c.content) as first_message
                   FROM sessions s
                   LEFT JOIN feedback f ON f.session_id = s.id
                   LEFT JOIN conversation_logs c ON c.session_id = s.id AND c.message_type = 'user'
                   GROUP BY s.id
                   ORDER BY s.created_at DESC
                   LIMIT ? OFFSET ?''',
                (limit, offset)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_intent_performance(self) -> List[dict]:
        """Get per-intent performance metrics"""
        with self.get_connection() as conn:
            rows = conn.execute(
                '''SELECT intent_tag, SUM(total_hits) as total,
                          AVG(avg_confidence) as avg_conf,
                          MAX(date) as last_seen
                   FROM intent_analytics
                   GROUP BY intent_tag
                   ORDER BY total DESC'''
            ).fetchall()
            return [
                {
                    'intent': r[0],
                    'total_hits': r[1],
                    'avg_confidence': round((r[2] or 0) * 100, 2),
                    'last_seen': r[3]
                }
                for r in rows
            ]

    def log_unknown_query(self, query: str, best_guess: str = None):
        """Log low-confidence/unknown queries for analysis"""
        query_lower = query.lower().strip()
        with self.get_connection() as conn:
            existing = conn.execute(
                'SELECT id, frequency FROM unknown_queries WHERE query = ?', (query_lower,)
            ).fetchone()
            if existing:
                conn.execute(
                    'UPDATE unknown_queries SET frequency = frequency + 1, last_seen = CURRENT_TIMESTAMP WHERE id = ?',
                    (existing[0],)
                )
            else:
                conn.execute(
                    'INSERT INTO unknown_queries (query, best_guess_intent) VALUES (?, ?)',
                    (query_lower, best_guess)
                )

    def get_unknown_queries(self, limit: int = 20) -> List[dict]:
        """Get frequently unknown queries"""
        with self.get_connection() as conn:
            rows = conn.execute(
                '''SELECT query, frequency, best_guess_intent, first_seen, last_seen
                   FROM unknown_queries
                   ORDER BY frequency DESC
                   LIMIT ?''',
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def log_escalation(self, session_id: str, reason: str, triggered_by: str = 'user'):
        """Log a human escalation"""
        with self.get_connection() as conn:
            conn.execute(
                '''INSERT INTO escalations (session_id, reason, triggered_by)
                   VALUES (?, ?, ?)''',
                (session_id, reason, triggered_by)
            )
            conn.execute(
                'UPDATE sessions SET is_escalated = TRUE WHERE id = ?',
                (session_id,)
            )

    def export_conversations(self, start_date: str = None, end_date: str = None) -> List[dict]:
        """Export conversation logs for a date range"""
        query = '''
            SELECT c.*, s.ip_address, s.resolution_status
            FROM conversation_logs c
            JOIN sessions s ON s.id = c.session_id
        '''
        params = []
        if start_date:
            query += ' WHERE DATE(c.created_at) >= ?'
            params.append(start_date)
        if end_date:
            query += (' AND' if start_date else ' WHERE') + ' DATE(c.created_at) <= ?'
            params.append(end_date)
        query += ' ORDER BY c.created_at ASC'

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]


# Singleton instance
_db_instance: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get or create singleton database manager"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
