"""Routes package for AI Trainer service"""

from .training_routes import training_bp
from .model_routes import model_bp
from .config_routes import config_bp
from .analysis_routes import analysis_bp

__all__ = ["training_bp", "model_bp", "config_bp", "analysis_bp"]
