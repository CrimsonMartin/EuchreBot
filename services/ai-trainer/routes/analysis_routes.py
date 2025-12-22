"""Analysis routes for AI Trainer service - Trump strategy analysis"""

import torch
import numpy as np
from flask import Blueprint, jsonify, current_app
from model_manager import ModelManager
from networks.basic_nn import encode_trump_state

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/api/models/<model_id>/analyze-trump", methods=["GET"])
def analyze_trump_strategy(model_id):
    """
    Analyze a model's trump calling strategy by testing various scenarios.
    Returns detailed statistics on when the model calls vs passes.
    """
    try:
        # Load the model
        model_manager = ModelManager(current_app.config["DATABASE_URL"])
        model = model_manager.load_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Card encoding reference
        all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                all_cards.append(f"{rank}{suit}")

        # Define test scenarios
        scenarios = []

        # Suits for testing
        suits = ["C", "D", "H", "S"]
        suit_names = {"C": "Clubs", "D": "Diamonds", "H": "Hearts", "S": "Spades"}
        opposite_suit = {"C": "S", "S": "C", "D": "H", "H": "D"}

        for trump_suit in suits:
            left_suit = opposite_suit[trump_suit]

            # Strong hands - Both bowers
            scenarios.append(
                {
                    "hand": [
                        f"J{trump_suit}",
                        f"J{left_suit}",
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"Q{trump_suit}",
                    ],
                    "turned_up": f"9{trump_suit}",
                    "description": f"Both bowers + A,K,Q of {suit_names[trump_suit]}",
                    "category": "both_bowers",
                    "trump_count": 5,
                    "has_right": True,
                    "has_left": True,
                    "off_aces": 0,
                }
            )

            # Right bower + support
            scenarios.append(
                {
                    "hand": [
                        f"J{trump_suit}",
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"9{left_suit}",
                        f"10{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"Right bower + A,K of {suit_names[trump_suit]}",
                    "category": "right_bower",
                    "trump_count": 3,
                    "has_right": True,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Right bower alone
            scenarios.append(
                {
                    "hand": [
                        f"J{trump_suit}",
                        f"9{left_suit}",
                        f"10{left_suit}",
                        f"Q{left_suit}",
                        f"K{left_suit}",
                    ],
                    "turned_up": f"A{trump_suit}",
                    "description": f"Right bower only, no other trump",
                    "category": "right_bower_alone",
                    "trump_count": 1,
                    "has_right": True,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Left bower + support
            scenarios.append(
                {
                    "hand": [
                        f"J{left_suit}",
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"9{left_suit}",
                        f"10{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"Left bower + A,K of {suit_names[trump_suit]}",
                    "category": "left_bower",
                    "trump_count": 3,
                    "has_right": False,
                    "has_left": True,
                    "off_aces": 0,
                }
            )

            # Left bower alone
            scenarios.append(
                {
                    "hand": [
                        f"J{left_suit}",
                        f"9{trump_suit}",
                        f"10{left_suit}",
                        f"Q{left_suit}",
                        f"K{left_suit}",
                    ],
                    "turned_up": f"A{trump_suit}",
                    "description": f"Left bower + 9 of trump only",
                    "category": "left_bower_weak",
                    "trump_count": 2,
                    "has_right": False,
                    "has_left": True,
                    "off_aces": 0,
                }
            )

            # No bowers, strong trump
            scenarios.append(
                {
                    "hand": [
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"Q{trump_suit}",
                        f"10{trump_suit}",
                        f"9{left_suit}",
                    ],
                    "turned_up": f"9{trump_suit}",
                    "description": f"A,K,Q,10 of {suit_names[trump_suit]}, no bowers",
                    "category": "no_bowers_strong",
                    "trump_count": 4,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # No bowers, medium trump
            scenarios.append(
                {
                    "hand": [
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"9{trump_suit}",
                        f"10{left_suit}",
                        f"Q{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"A,K,9 of {suit_names[trump_suit]}, no bowers",
                    "category": "no_bowers_medium",
                    "trump_count": 3,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Weak trump with off-aces
            other_suits = [s for s in suits if s != trump_suit and s != left_suit]
            scenarios.append(
                {
                    "hand": [
                        f"9{trump_suit}",
                        f"10{trump_suit}",
                        f"A{other_suits[0]}",
                        f"A{other_suits[1]}",
                        f"K{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"9,10 of trump + 2 off-aces",
                    "category": "weak_trump_off_aces",
                    "trump_count": 2,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 2,
                }
            )

            # Garbage hand
            scenarios.append(
                {
                    "hand": [
                        f"9{left_suit}",
                        f"10{left_suit}",
                        f"Q{other_suits[0]}",
                        f"K{other_suits[0]}",
                        f"9{other_suits[1]}",
                    ],
                    "turned_up": f"A{trump_suit}",
                    "description": f"No trump, no aces - garbage",
                    "category": "garbage",
                    "trump_count": 0,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Borderline - 2 trump no bowers
            scenarios.append(
                {
                    "hand": [
                        f"K{trump_suit}",
                        f"Q{trump_suit}",
                        f"A{left_suit}",
                        f"K{left_suit}",
                        f"Q{other_suits[0]}",
                    ],
                    "turned_up": f"9{trump_suit}",
                    "description": f"K,Q of trump + off-ace",
                    "category": "borderline",
                    "trump_count": 2,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 1,
                }
            )

        # Test each scenario at each position
        results = {
            "model_id": model_id,
            "total_scenarios": 0,
            "position_stats": {
                "0": {
                    "name": "1st (Left of Dealer)",
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                },
                "1": {"name": "2nd", "calls": 0, "passes": 0, "total": 0},
                "2": {"name": "3rd", "calls": 0, "passes": 0, "total": 0},
                "3": {"name": "Dealer", "calls": 0, "passes": 0, "total": 0},
            },
            "category_stats": {},
            "bower_stats": {
                "both_bowers": {"calls": 0, "passes": 0, "total": 0},
                "right_only": {"calls": 0, "passes": 0, "total": 0},
                "left_only": {"calls": 0, "passes": 0, "total": 0},
                "no_bowers": {"calls": 0, "passes": 0, "total": 0},
            },
            "trump_count_stats": {},
            "detailed_results": [],
        }

        # Initialize category stats
        for scenario in scenarios:
            cat = scenario["category"]
            if cat not in results["category_stats"]:
                results["category_stats"][cat] = {
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                    "description": "",
                }

        # Initialize trump count stats
        for i in range(6):
            results["trump_count_stats"][str(i)] = {"calls": 0, "passes": 0, "total": 0}

        # Run each scenario at each position
        for scenario in scenarios:
            for position in range(4):
                # Create game state for encoding
                game_state = {
                    "hand": scenario["hand"],
                    "current_player_position": position,
                    "dealer_position": 3,  # Dealer is always position 3 for consistency
                }

                # Encode trump state
                trump_encoding = encode_trump_state(game_state, scenario["turned_up"])

                # Get model prediction with probabilities
                with torch.no_grad():
                    x = torch.FloatTensor(trump_encoding).unsqueeze(0).to(model.device)
                    output = model.forward_trump(x)
                    probs = output.cpu().numpy()[0]
                    decision_idx = int(np.argmax(probs))

                # decision_idx: 0-3 = call suits, 4 = pass
                is_call = decision_idx != 4
                called_suit = ["C", "D", "H", "S", "PASS"][decision_idx]

                # Update statistics
                results["total_scenarios"] += 1
                pos_key = str(position)

                if is_call:
                    results["position_stats"][pos_key]["calls"] += 1
                else:
                    results["position_stats"][pos_key]["passes"] += 1
                results["position_stats"][pos_key]["total"] += 1

                # Category stats
                cat = scenario["category"]
                if is_call:
                    results["category_stats"][cat]["calls"] += 1
                else:
                    results["category_stats"][cat]["passes"] += 1
                results["category_stats"][cat]["total"] += 1
                results["category_stats"][cat]["description"] = scenario["description"]

                # Bower stats
                if scenario["has_right"] and scenario["has_left"]:
                    bower_key = "both_bowers"
                elif scenario["has_right"]:
                    bower_key = "right_only"
                elif scenario["has_left"]:
                    bower_key = "left_only"
                else:
                    bower_key = "no_bowers"

                if is_call:
                    results["bower_stats"][bower_key]["calls"] += 1
                else:
                    results["bower_stats"][bower_key]["passes"] += 1
                results["bower_stats"][bower_key]["total"] += 1

                # Trump count stats
                tc_key = str(scenario["trump_count"])
                if is_call:
                    results["trump_count_stats"][tc_key]["calls"] += 1
                else:
                    results["trump_count_stats"][tc_key]["passes"] += 1
                results["trump_count_stats"][tc_key]["total"] += 1

                # Detailed result with confidence
                results["detailed_results"].append(
                    {
                        "hand": scenario["hand"],
                        "turned_up": scenario["turned_up"],
                        "position": position,
                        "position_name": results["position_stats"][pos_key]["name"],
                        "decision": "CALL" if is_call else "PASS",
                        "called_suit": called_suit,
                        "confidence": round(float(probs[decision_idx]) * 100, 1),
                        "category": cat,
                        "description": scenario["description"],
                        "trump_count": scenario["trump_count"],
                        "has_right": scenario["has_right"],
                        "has_left": scenario["has_left"],
                    }
                )

        # Calculate percentages
        for pos_key in results["position_stats"]:
            stats = results["position_stats"][pos_key]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        for cat in results["category_stats"]:
            stats = results["category_stats"][cat]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        for bower_key in results["bower_stats"]:
            stats = results["bower_stats"][bower_key]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        for tc_key in results["trump_count_stats"]:
            stats = results["trump_count_stats"][tc_key]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        return jsonify(results)

    except Exception as e:
        print(f"Error analyzing trump strategy: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
