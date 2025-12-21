"""
Euchre Web UI Flask Application
"""

import os
from flask import Flask, session
from flask_session import Session
import redis


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)

    # Configuration
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_USE_SIGNER"] = True
    app.config["SESSION_REDIS"] = redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/1")
    )

    # Initialize session
    Session(app)

    # API URL for backend
    app.config["EUCHRE_API_URL"] = os.getenv("EUCHRE_API_URL", "http://euchre-api:5000")
    app.config["AI_TRAINER_URL"] = os.getenv("AI_TRAINER_URL", "http://ai-trainer:5003")

    # Register blueprints
    from app.routes.main_routes import main_bp

    app.register_blueprint(main_bp)

    # Health check endpoint
    @app.route("/health")
    def health():
        return {"status": "healthy", "service": "web-ui"}

    return app
