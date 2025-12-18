"""
AI model routes
"""

from flask import Blueprint, request, jsonify

ai_bp = Blueprint('ai', __name__)


@ai_bp.route('/models', methods=['GET'])
def list_models():
    """List available AI models"""
    # This would query the database for active AI models
    return jsonify({
        'models': [
            {
                'id': 'random',
                'name': 'Random Player',
                'type': 'random',
                'active': True
            }
        ]
    })


@ai_bp.route('/predict', methods=['POST'])
def predict_move():
    """Get AI move recommendation"""
    data = request.json
    game_state = data.get('game_state')
    model_id = data.get('model_id', 'random')
    
    # This would call the AI trainer service
    # For now, return a placeholder
    
    return jsonify({
        'recommended_card': '9H',
        'confidence': 0.75,
        'model_id': model_id
    })


@ai_bp.route('/models/<model_id>/play', methods=['POST'])
def ai_play(model_id):
    """Have AI play a move"""
    data = request.json
    game_id = data.get('game_id')
    
    return jsonify({
        'success': True,
        'model_id': model_id,
        'card_played': '9H'
    })
