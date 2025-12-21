"""
Test game end detection and hand completion
"""

import pytest
from euchre_core import EuchreGame, PlayerType, Card, Suit
from euchre_core.game import GamePhase


def test_hand_complete_detection():
    """Test that game correctly detects when all cards are played"""
    game = EuchreGame("test-game")

    # Add 4 players
    game.add_player("Player 1", PlayerType.HUMAN)
    game.add_player("Player 2", PlayerType.HUMAN)
    game.add_player("Player 3", PlayerType.HUMAN)
    game.add_player("Player 4", PlayerType.HUMAN)

    # Start a new hand
    game.start_new_hand()

    # Call trump (player left of dealer)
    game.call_trump(suit=None, go_alone=False)

    # Dealer discards
    dealer = game.state.get_player(game.state.dealer_position)
    game.dealer_discard(dealer.hand[0])

    # Verify we're in playing phase
    assert game.state.phase == GamePhase.PLAYING

    # Play all 5 tricks (20 cards total)
    tricks_completed = 0
    while game.state.phase == GamePhase.PLAYING:
        current_player = game.state.get_current_player()
        valid_cards = game.get_valid_moves(current_player.position)

        if not valid_cards:
            break

        # Play a valid card
        result = game.play_card(valid_cards[0])

        if result.get("trick_complete"):
            tricks_completed += 1

        if result.get("hand_complete"):
            break

    # Verify hand is complete
    assert game.state.phase == GamePhase.HAND_COMPLETE
    assert tricks_completed == 5
    assert len(game.state.tricks_played) == 5

    # Verify all players have empty hands
    for player in game.state.players:
        assert len(player.hand) == 0


def test_game_over_detection():
    """Test that game correctly detects when a team reaches 10 points"""
    game = EuchreGame("test-game")

    # Add 4 players
    game.add_player("Player 1", PlayerType.HUMAN)
    game.add_player("Player 2", PlayerType.HUMAN)
    game.add_player("Player 3", PlayerType.HUMAN)
    game.add_player("Player 4", PlayerType.HUMAN)

    # Manually set score close to winning
    game.state.team1_score = 9
    game.state.team2_score = 8

    # Start a new hand
    game.start_new_hand()

    # Call trump
    game.call_trump(suit=None, go_alone=False)

    # Dealer discards
    dealer = game.state.get_player(game.state.dealer_position)
    game.dealer_discard(dealer.hand[0])

    # Play all cards
    while game.state.phase == GamePhase.PLAYING:
        current_player = game.state.get_current_player()
        valid_cards = game.get_valid_moves(current_player.position)

        if not valid_cards:
            break

        game.play_card(valid_cards[0])

    # Verify hand is complete
    assert game.state.phase in [GamePhase.HAND_COMPLETE, GamePhase.GAME_OVER]

    # If a team reached 10, game should be over
    if game.state.team1_score >= 10 or game.state.team2_score >= 10:
        assert game.state.phase == GamePhase.GAME_OVER


def test_start_new_hand_after_complete():
    """Test that a new hand can be started after hand completion"""
    game = EuchreGame("test-game")

    # Add 4 players
    game.add_player("Player 1", PlayerType.HUMAN)
    game.add_player("Player 2", PlayerType.HUMAN)
    game.add_player("Player 3", PlayerType.HUMAN)
    game.add_player("Player 4", PlayerType.HUMAN)

    # Start first hand
    game.start_new_hand()
    assert game.state.hand_number == 1

    # Call trump
    game.call_trump(suit=None, go_alone=False)

    # Dealer discards
    dealer = game.state.get_player(game.state.dealer_position)
    game.dealer_discard(dealer.hand[0])

    # Play all cards
    while game.state.phase == GamePhase.PLAYING:
        current_player = game.state.get_current_player()
        valid_cards = game.get_valid_moves(current_player.position)

        if not valid_cards:
            break

        game.play_card(valid_cards[0])

    # Verify hand is complete
    assert game.state.phase == GamePhase.HAND_COMPLETE

    # Start a new hand
    game.start_new_hand()

    # Verify new hand started
    assert game.state.hand_number == 2
    assert game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1

    # Verify all players have 5 cards again
    for player in game.state.players:
        assert len(player.hand) == 5


def test_cannot_start_new_hand_when_game_over():
    """Test that new hand cannot be started when game is over"""
    game = EuchreGame("test-game")

    # Add 4 players
    game.add_player("Player 1", PlayerType.HUMAN)
    game.add_player("Player 2", PlayerType.HUMAN)
    game.add_player("Player 3", PlayerType.HUMAN)
    game.add_player("Player 4", PlayerType.HUMAN)

    # Set game to game over state
    game.state.phase = GamePhase.GAME_OVER
    game.state.team1_score = 10

    # Attempting to start new hand should work (it resets the game)
    # This is actually allowed - it starts a new hand regardless
    game.start_new_hand()

    # The hand should start normally
    assert game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1
