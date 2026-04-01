"""
run.py - Application entry point
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

logger = logging.getLogger(__name__)


def main():
    """Start the Flask application"""
    app = create_app()

    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'

    print("""
╔══════════════════════════════════════════════════════════════╗
║       🤖 INTELLIGENT CUSTOMER SUPPORT CHATBOT API           ║
║                                                              ║
║  Chat UI  → http://localhost:{port}                          ║
║  Dashboard → http://localhost:{port}/dashboard               ║
║  API Base → http://localhost:{port}/api                      ║
║                                                              ║
║  Key Endpoints:                                              ║
║  POST /api/chat/session    - Create session                  ║
║  POST /api/chat/message    - Send message                    ║
║  GET  /api/analytics/dashboard - Analytics                   ║
║  GET  /health              - Health check                    ║
╚══════════════════════════════════════════════════════════════╝
    """.format(port=port))

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
