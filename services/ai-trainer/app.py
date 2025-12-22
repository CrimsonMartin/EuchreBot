"""
AI Trainer Service - PyTorch Neural Network Training with Genetic Algorithm
Continuous Training Mode with ELO Ratings
"""

import os
import sys
import torch
import multiprocessing
from flask import Flask

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

# Create Flask app
app = Flask(__name__)

# Configuration
app.config["DATABASE_URL"] = os.getenv(
    "DATABASE_URL", "postgresql://euchre:euchre_dev_pass@postgres:5432/euchrebot"
)

# Parallel processing configuration
app.config["NUM_WORKERS"] = int(
    os.getenv("NUM_WORKERS", str(max(1, multiprocessing.cpu_count() - 1)))
)
app.config["PARALLEL_MODE"] = os.getenv(
    "PARALLEL_MODE", "thread"
)  # "thread", "process", or "sequential"
app.config["USE_CUDA"] = os.getenv("USE_CUDA", "true").lower() == "true"

print(f"AI Trainer Configuration:")
print(f"  NUM_WORKERS: {app.config['NUM_WORKERS']}")
print(f"  PARALLEL_MODE: {app.config['PARALLEL_MODE']}")
print(f"  USE_CUDA: {app.config['USE_CUDA']}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")

# Register blueprints
from routes import training_bp, model_bp, config_bp, analysis_bp

app.register_blueprint(training_bp)
app.register_blueprint(model_bp)
app.register_blueprint(config_bp)
app.register_blueprint(analysis_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
