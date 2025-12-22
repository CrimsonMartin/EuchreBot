"""Model routes for AI Trainer service"""

import psycopg2
from flask import Blueprint, jsonify, request, current_app
from model_manager import ModelManager

model_bp = Blueprint("model", __name__)


@model_bp.route("/api/models", methods=["GET"])
def list_models():
    """List all trained models"""
    try:
        conn = psycopg2.connect(current_app.config["DATABASE_URL"])
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, architecture, generation, elo_rating, created_at
            FROM ai_models
            WHERE active = true
            ORDER BY elo_rating DESC
            LIMIT 50
        """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        models = []
        for row in rows:
            models.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "architecture": row[2],
                    "generation": row[3],
                    "elo_rating": row[4] or 1500,
                    "created_at": row[5].isoformat() if row[5] else None,
                }
            )

        return jsonify({"models": models})
    except Exception as e:
        print(f"Error listing models: {e}")
        return jsonify({"models": []})


@model_bp.route("/api/models/<model_id>/predict", methods=["POST"])
def predict_move(model_id):
    """Predict the best move for a given game state using a trained model"""
    try:
        data = request.json
        game_state = data.get("game_state")
        valid_cards = data.get("valid_cards", [])

        if not game_state:
            return jsonify({"error": "game_state required"}), 400

        # Load the model
        model_manager = ModelManager(current_app.config["DATABASE_URL"])
        model = model_manager.load_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Encode game state
        from networks.basic_nn import encode_game_state

        state_encoding = encode_game_state(game_state)

        # Get prediction
        card_index = model.predict_card(state_encoding)

        # Map card index to actual card
        all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                all_cards.append(f"{rank}{suit}")

        # Get predicted card
        if 0 <= card_index < len(all_cards):
            predicted_card = all_cards[card_index]

            # If valid_cards provided, ensure prediction is valid
            if valid_cards and predicted_card not in valid_cards:
                # Fall back to first valid card
                predicted_card = valid_cards[0] if valid_cards else predicted_card

            return jsonify(
                {
                    "card": predicted_card,
                    "model_id": model_id,
                    "card_index": card_index,
                }
            )
        else:
            # Invalid index, return first valid card or error
            if valid_cards:
                return jsonify(
                    {
                        "card": valid_cards[0],
                        "model_id": model_id,
                        "fallback": True,
                    }
                )
            return jsonify({"error": "Invalid prediction"}), 500

    except Exception as e:
        print(f"Error predicting move: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
