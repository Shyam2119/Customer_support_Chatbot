"""
__init__.py - Flask application factory
"""

import os
import logging
from flask import Flask
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)


def create_app(config_name: str = None) -> Flask:
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    )

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chatbot-dev-secret-2024')
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

    # CORS
    CORS(app, origins=['*'])

    # Register blueprints
    from app.routes import chat_bp, analytics_bp, admin_bp, main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(admin_bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        from flask import jsonify
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def server_error(e):
        from flask import jsonify
        return jsonify({'error': 'Internal server error'}), 500

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        from flask import jsonify
        return jsonify({'error': 'Rate limit exceeded. Please slow down.'}), 429

    logging.getLogger(__name__).info(f"Application created in '{config_name}' mode")
    return app
