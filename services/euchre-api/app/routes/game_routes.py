"""
Game management routes
"""

from flask import Blueprint, request, jsonify
import json
import uuid
from euchre_core import EuchreGame, PlayerType, Card, Suit
from app import redis_client

game_bp = Blueprint('game', __name__)


def save_game_to_redis(game: EuchreGame):
    """Save game state to Redis"""
    key = f"game:{game.state.game_id}"
    state_dict = game.get_state(perspective_position=None)
    redis_client.set(key, json.dumps(state_dict), ex=3600)  # 1 hour expiry


def load_game_from_redis(game_id: str) -> EuchreGame:
    """Load game state from Redis"""
    key = f"game:{game_id}"
    data = redis_client.get(key)
    if not data:
        return None
    
    # For now, we'll need to reconstruct the game
    # In production, consider storing serialized game object
    state_dict = json.loads(data)
    
    # This is simplified - full implementation would reconstruct game state
    game = EuchreGame(game_id)
    return game


@game_bp.route('/games', methods=['POST'])
def create_game():
    """Create a new game"""
    data = request.json
    players = data.get('players', [])
    
    if len(players) != 4:
        return jsonify({'error': 'Exactly 4 players required'}), 400
    
    game_id = str(uuid.uuid4())
    game = EuchreGame(game_id)
    
    for player_data in players:
        name = player_data.get('name', 'Player')
        player_type_str = player_data.get('type', 'human')
        
        player_type = PlayerType.HUMAN
        if player_type_str == 'random_ai':
            player_type = PlayerType.RANDOM_AI
        elif player_type_str == 'neural_net_ai':
            player_type = PlayerType.NEURAL_NET_AI
        
        game.add_player(name, player_type)
    
    # Start first hand
    game.start_new_hand()
    
    save_game_to_redis(game)
    
    return jsonify({
        'game_id': game_id,
        'state': game.get_state()
    }), 201


@game_bp.route('/games/<game_id>', methods=['GET'])
def get_game(game_id):
    """Get game state"""
    perspective = request.args.get('perspective', type=int)
    
    # For now, return mock state since we haven't implemented full serialization
    return jsonify({
        'game_id': game_id,
        'phase': 'playing',
        'message': 'Game state endpoint - full implementation pending'
    })


@game_bp.route('/games/<game_id>/move', methods=['POST'])
def play_move(game_id):
    """Play a card"""
    data = request.json
    card_str = data.get('card')
    
    if not card_str:
        return jsonify({'error': 'Card required'}), 400
    
    try:
        card = Card.from_string(card_str)
    except Exception as e:
        return jsonify({'error': f'Invalid card: {str(e)}'}), 400
    
    # Load game, play card, save game
    # Full implementation would load from Redis, play card, save back
    
    return jsonify({
        'success': True,
        'message': 'Move played',
        'card': card_str
    })


@game_bp.route('/games/<game_id>/trump', methods=['POST'])
def call_trump(game_id):
    """Call trump suit"""
    data = request.json
    suit_str = data.get('suit')
    go_alone = data.get('go_alone', False)
    pass_trump = data.get('pass', False)
    
    if pass_trump:
        return jsonify({
            'success': True,
            'action': 'pass'
        })
    
    if suit_str:
        try:
            suit = Suit.from_string(suit_str)
        except Exception as e:
            return jsonify({'error': f'Invalid suit: {str(e)}'}), 400
    else:
        suit = None
    
    return jsonify({
        'success': True,
        'action': 'call_trump',
        'suit': suit_str,
        'go_alone': go_alone
    })


@game_bp.route('/games/<game_id>', methods=['DELETE'])
def delete_game(game_id):
    """Delete/abandon a game"""
    key = f"game:{game_id}"
    redis_client.delete(key)
    
    return jsonify({'success': True, 'message': 'Game deleted'})
