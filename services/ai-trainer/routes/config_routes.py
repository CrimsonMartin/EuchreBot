"""Configuration routes for AI Trainer service"""

import torch
import multiprocessing
from flask import Blueprint, jsonify, request, current_app

config_bp = Blueprint("config", __name__)


@config_bp.route("/health")
def health():
    return {
        "status": "healthy",
        "service": "ai-trainer",
        "config": {
            "num_workers": current_app.config["NUM_WORKERS"],
            "parallel_mode": current_app.config["PARALLEL_MODE"],
            "use_cuda": current_app.config["USE_CUDA"],
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cpu_count": multiprocessing.cpu_count(),
        },
    }


@config_bp.route("/api/config", methods=["GET"])
def get_config():
    """Get current training configuration"""
    return jsonify(
        {
            "num_workers": current_app.config["NUM_WORKERS"],
            "parallel_mode": current_app.config["PARALLEL_MODE"],
            "use_cuda": current_app.config["USE_CUDA"],
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "cpu_count": multiprocessing.cpu_count(),
            "description": {
                "num_workers": "Number of parallel workers for tournament groups",
                "parallel_mode": "thread (recommended for GPU), process (CPU-only), or sequential",
                "use_cuda": "Whether to use GPU acceleration",
            },
        }
    )


@config_bp.route("/api/config", methods=["POST"])
def update_config():
    """Update training configuration (affects new training runs only)"""
    data = request.json

    if "num_workers" in data:
        current_app.config["NUM_WORKERS"] = int(data["num_workers"])
    if "parallel_mode" in data:
        if data["parallel_mode"] in ["thread", "process", "sequential"]:
            current_app.config["PARALLEL_MODE"] = data["parallel_mode"]
        else:
            return (
                jsonify(
                    {
                        "error": "Invalid parallel_mode. Use: thread, process, or sequential"
                    }
                ),
                400,
            )
    if "use_cuda" in data:
        current_app.config["USE_CUDA"] = bool(data["use_cuda"])

    return jsonify(
        {
            "status": "updated",
            "config": {
                "num_workers": current_app.config["NUM_WORKERS"],
                "parallel_mode": current_app.config["PARALLEL_MODE"],
                "use_cuda": current_app.config["USE_CUDA"],
            },
            "note": "Changes will apply to new training runs only",
        }
    )
