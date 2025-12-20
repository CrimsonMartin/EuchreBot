"""
Test valid moves endpoint
"""

import pytest
from app import create_app
from euchre_core import EuchreGame, PlayerType, Card, Suit
from euchre_core.game import GamePhase

@pytest.fixture
def client():
    """Create test client"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['REDIS_URL'] = 'redis://localhost:6379/1'

    with app.test_client() as client:
        yield client

def test_get_valid_moves_during_playing_phase(client):
    """Test getting valid moves during the playing phase"""
    # Create a game
    response = client.post('/games', json={
        'players': [
            {'name': 'Player1', 'type': 'human'},
            {'name': 'Player2', 'type': 'human'},
            {'name': 'Player3', 'type': 'human'},
            {'name': 'Player4', 'type': 'human'},
        ]
    })
    assert response.status_code == 201
    game_id = response.json['game_id']

    # Get initial game state
    state_response = client.get(f'/games/{game_id}')
    assert state_response.status_code == 200
    state = state_response.json

    # Call trump to move to playing phase
    trump_response = client.post(f'/games/{game_id}/trump', json={
        'suit': 'H',
        'go_alone': False
    })
    assert trump_response.status_code == 200

    # Get valid moves for current player
    valid_moves_response = client.get(f'/games/{game_id}/valid-moves')
    assert valid_moves_response.status_code == 200

    data = valid_moves_response.json
    assert 'valid_cards' in data
    assert 'hand' in data
    assert 'player_position' in data
    assert 'must_follow_suit' in data
    assert 'current_trick_lead_suit' in data
    assert 'trump_suit' in data

    # Valid cards should be a subset of hand
    valid_cards = data['valid_cards']
    hand = data['hand']
    assert all(card in hand for card in valid_cards)

    # Trump suit should be set
    assert data['trump_suit'] == 'H'

    # Clean up
    client.delete(f'/games/{game_id}')

def test_get_valid_moves_for_specific_player(client):
    """Test getting valid moves for a specific player perspective"""
    # Create a game
    response = client.post('/games', json={
        'players': [
            {'name': 'Player1', 'type': 'human'},
            {'name': 'Player2', 'type': 'human'},
            {'name': 'Player3', 'type': 'human'},
            {'name': 'Player4', 'type': 'human'},
        ]
    })
    assert response.status_code == 201
    game_id = response.json['game_id']

    # Call trump to move to playing phase
    trump_response = client.post(f'/games/{game_id}/trump', json={
        'suit': 'D',
        'go_alone': False
    })
    assert trump_response.status_code == 200

    # Get valid moves for player 1 (regardless of who's turn it is)
    valid_moves_response = client.get(f'/games/{game_id}/valid-moves?perspective=1')
    assert valid_moves_response.status_code == 200

    data = valid_moves_response.json
    assert data['player_position'] == 1
    assert 'valid_cards' in data
    assert 'hand' in data

    # Clean up
    client.delete(f'/games/{game_id}')

def test_get_valid_moves_before_trump_called(client):
    """Test getting valid moves before trump is called (should return empty)"""
    # Create a game
    response = client.post('/games', json={
        'players': [
            {'name': 'Player1', 'type': 'human'},
            {'name': 'Player2', 'type': 'human'},
            {'name': 'Player3', 'type': 'human'},
            {'name': 'Player4', 'type': 'human'},
        ]
    })
    assert response.status_code == 201
    game_id = response.json['game_id']

    # Get valid moves during trump selection phase
    valid_moves_response = client.get(f'/games/{game_id}/valid-moves')
    assert valid_moves_response.status_code == 200

    data = valid_moves_response.json
    assert data['valid_cards'] == []
    assert data['trump_suit'] is None

    # Clean up
    client.delete(f'/games/{game_id}')

def test_get_valid_moves_follow_suit_logic(client):
    """Test that valid moves correctly enforce follow suit rules"""
    # Create a game
    response = client.post('/games', json={
        'players': [
            {'name': 'Player1', 'type': 'human'},
            {'name': 'Player2', 'type': 'human'},
            {'name': 'Player3', 'type': 'human'},
            {'name': 'Player4', 'type': 'human'},
        ]
    })
    assert response.status_code == 201
    game_id = response.json['game_id']

    # Call trump to move to playing phase
    trump_response = client.post(f'/games/{game_id}/trump', json={
        'suit': 'C',
        'go_alone': False
    })
    assert trump_response.status_code == 200

    # Get valid moves
    valid_moves_response = client.get(f'/games/{game_id}/valid-moves')
    assert valid_moves_response.status_code == 200

    data = valid_moves_response.json
    valid_cards = data['valid_cards']
    hand = data['hand']

    # If player has cards in lead suit, they must play one
    if len(valid_cards) < len(hand):
        assert data['must_follow_suit'] is True
    else:
        assert data['must_follow_suit'] is False

    # Clean up
    client.delete(f'/games/{game_id}')
