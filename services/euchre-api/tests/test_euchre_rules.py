"""
Comprehensive tests for Euchre game rules
"""

import pytest
from euchre_core import EuchreGame, PlayerType, Card, Suit, Rank
from euchre_core.game import GamePhase


class TestDealingRules:
    """Test dealing rules"""

    def test_each_player_gets_5_cards(self):
        """Each player should receive exactly 5 cards"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        for player in game.state.players:
            assert len(player.hand) == 5, f"{player.name} should have 5 cards"

    def test_turned_up_card_exists(self):
        """A card should be turned up for trump selection"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        assert game.state.turned_up_card is not None
        assert isinstance(game.state.turned_up_card, Card)


class TestDealerDiscard:
    """Test dealer discard after picking up in round 1"""

    def test_dealer_has_6_cards_after_pickup(self):
        """Dealer should have 6 cards after picking up turned-up card"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Player 1 calls trump (picks up)
        game.call_trump(suit=None, go_alone=False)

        # Dealer should now have 6 cards
        dealer = game.state.get_player(game.state.dealer_position)
        assert len(dealer.hand) == 6

    def test_dealer_has_5_cards_after_discard(self):
        """Dealer should have 5 cards after discarding"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Player 1 calls trump (picks up)
        game.call_trump(suit=None, go_alone=False)

        # Dealer discards a card
        dealer = game.state.get_player(game.state.dealer_position)
        card_to_discard = dealer.hand[0]
        game.dealer_discard(card_to_discard)

        # Dealer should now have 5 cards
        assert len(dealer.hand) == 5

    def test_game_phase_after_discard(self):
        """Game should move to PLAYING phase after dealer discards"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Player 1 calls trump (picks up)
        game.call_trump(suit=None, go_alone=False)
        assert game.state.phase == GamePhase.DEALER_DISCARD

        # Dealer discards a card
        dealer = game.state.get_player(game.state.dealer_position)
        card_to_discard = dealer.hand[0]
        game.dealer_discard(card_to_discard)

        assert game.state.phase == GamePhase.PLAYING

    def test_round2_no_discard_needed(self):
        """In round 2, no discard is needed"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Everyone passes in round 1
        game.pass_trump()  # Player 1
        game.pass_trump()  # Player 2
        game.pass_trump()  # Player 3
        game.pass_trump()  # Player 4 (dealer)

        # Now in round 2, player 1 calls trump
        turned_up_suit = game.state.turned_up_card.suit
        other_suits = [
            s
            for s in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
            if s != turned_up_suit
        ]
        game.call_trump(suit=other_suits[0], go_alone=False)

        # Should go directly to PLAYING phase
        assert game.state.phase == GamePhase.PLAYING

        # Dealer should still have 5 cards
        dealer = game.state.get_player(game.state.dealer_position)
        assert len(dealer.hand) == 5


class TestFollowSuitRules:
    """Test follow suit rules"""

    def test_must_follow_suit_when_able(self):
        """Player must follow suit if they have cards in the lead suit"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up a known game state
        game.state.trump = Suit.HEARTS
        game.state.phase = GamePhase.PLAYING
        game.state.current_player_position = 0

        # Give player 1 some hearts and some clubs
        game.state.players[0].hand = [
            Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.HEARTS, Rank.TEN),
            Card(Suit.CLUBS, Rank.NINE),
        ]

        # Start a trick with a heart lead
        from euchre_core.trick import Trick

        game.state.current_trick = Trick(0, Suit.HEARTS)
        game.state.current_trick.add_card(0, Card(Suit.HEARTS, Rank.ACE))
        game.state.current_player_position = 1

        # Give player 2 hearts and clubs
        game.state.players[1].hand = [
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.CLUBS, Rank.KING),
        ]

        # Get valid moves for player 2
        valid_cards = game.get_valid_moves(1)

        # Should only be able to play hearts
        assert len(valid_cards) == 1
        assert valid_cards[0].suit == Suit.HEARTS

    def test_can_play_any_card_when_no_lead_suit(self):
        """Player can play any card when they don't have the lead suit"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up a known game state
        game.state.trump = Suit.HEARTS
        game.state.phase = GamePhase.PLAYING

        # Give player 2 only clubs (no hearts)
        game.state.players[1].hand = [
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.SPADES, Rank.NINE),
        ]

        # Start a trick with a heart lead
        from euchre_core.trick import Trick

        game.state.current_trick = Trick(0, Suit.HEARTS)
        game.state.current_trick.add_card(0, Card(Suit.HEARTS, Rank.ACE))
        game.state.current_player_position = 1

        # Get valid moves for player 2
        valid_cards = game.get_valid_moves(1)

        # Should be able to play any card
        assert len(valid_cards) == 3


class TestLeftBowerRules:
    """Test left bower (Jack of same color) rules"""

    def test_left_bower_counts_as_trump(self):
        """Jack of same color should count as trump suit"""
        # Jack of Diamonds when Hearts is trump
        card = Card(Suit.DIAMONDS, Rank.JACK)
        assert card.effective_suit(Suit.HEARTS) == Suit.HEARTS

        # Jack of Clubs when Spades is trump
        card = Card(Suit.CLUBS, Rank.JACK)
        assert card.effective_suit(Suit.SPADES) == Suit.SPADES

    def test_left_bower_must_follow_trump(self):
        """Left bower must be played when trump is led"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set trump to hearts
        game.state.trump = Suit.HEARTS
        game.state.phase = GamePhase.PLAYING

        # Give player 2 the left bower (Jack of Diamonds) and a club
        game.state.players[1].hand = [
            Card(Suit.DIAMONDS, Rank.JACK),  # Left bower
            Card(Suit.CLUBS, Rank.NINE),
        ]

        # Start a trick with trump (hearts) lead
        from euchre_core.trick import Trick

        game.state.current_trick = Trick(0, Suit.HEARTS)
        game.state.current_trick.add_card(0, Card(Suit.HEARTS, Rank.ACE))
        game.state.current_player_position = 1

        # Get valid moves
        valid_cards = game.get_valid_moves(1)

        # Should only be able to play the left bower
        assert len(valid_cards) == 1
        assert valid_cards[0] == Card(Suit.DIAMONDS, Rank.JACK)


class TestTrumpSelectionRules:
    """Test trump selection rules"""

    def test_cannot_call_turned_up_suit_in_round2(self):
        """Cannot call the turned up suit in round 2"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        turned_up_suit = game.state.turned_up_card.suit

        # Everyone passes in round 1
        game.pass_trump()
        game.pass_trump()
        game.pass_trump()
        game.pass_trump()

        # Try to call the turned up suit in round 2
        with pytest.raises(ValueError, match="Cannot call turned up suit in round 2"):
            game.call_trump(suit=turned_up_suit, go_alone=False)

    def test_dealer_cannot_pass_in_round2(self):
        """Dealer cannot pass in round 2 (stick the dealer)"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Everyone passes in round 1
        game.pass_trump()
        game.pass_trump()
        game.pass_trump()
        game.pass_trump()

        # Now in round 2
        assert game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2

        # Pass until dealer's turn
        game.pass_trump()
        game.pass_trump()
        game.pass_trump()

        # Dealer should not be able to pass
        assert game.state.current_player_position == game.state.dealer_position
        with pytest.raises(ValueError, match="Dealer cannot pass in round 2"):
            game.pass_trump()


class TestScoringRules:
    """Test scoring rules"""

    def test_winning_team_gets_1_point_for_3_or_4_tricks(self):
        """Calling team gets 1 point for winning 3 or 4 tricks"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up game state
        game.state.trump = Suit.HEARTS
        game.state.trump_caller_position = 0  # Team 1 called
        game.state.team1_tricks = 3
        game.state.team2_tricks = 2
        game.state.phase = GamePhase.HAND_COMPLETE

        result = game._determine_hand_winner()

        assert result["winning_team"] == 1
        assert result["points_awarded"] == 1

    def test_march_gives_2_points(self):
        """Calling team gets 2 points for winning all 5 tricks (march)"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up game state
        game.state.trump = Suit.HEARTS
        game.state.trump_caller_position = 0  # Team 1 called
        game.state.team1_tricks = 5
        game.state.team2_tricks = 0
        game.state.going_alone = False

        result = game._determine_hand_winner()

        assert result["winning_team"] == 1
        assert result["points_awarded"] == 2

    def test_euchre_gives_2_points(self):
        """Defending team gets 2 points for euchring (calling team gets < 3 tricks)"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up game state
        game.state.trump = Suit.HEARTS
        game.state.trump_caller_position = 0  # Team 1 called
        game.state.team1_tricks = 2  # Less than 3
        game.state.team2_tricks = 3

        result = game._determine_hand_winner()

        assert result["winning_team"] == 2
        assert result["points_awarded"] == 2  # Euchre

    def test_alone_march_gives_4_points(self):
        """Going alone and winning all 5 tricks gives 4 points"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up game state
        game.state.trump = Suit.HEARTS
        game.state.trump_caller_position = 0  # Team 1 called
        game.state.team1_tricks = 5
        game.state.team2_tricks = 0
        game.state.going_alone = True

        result = game._determine_hand_winner()

        assert result["winning_team"] == 1
        assert result["points_awarded"] == 4


class TestGameFlow:
    """Test overall game flow"""

    def test_game_ends_at_10_points(self):
        """Game should end when a team reaches 10 points"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        # Set up game state
        game.state.trump = Suit.HEARTS
        game.state.trump_caller_position = 0
        game.state.team1_tricks = 5
        game.state.team2_tricks = 0
        game.state.team1_score = 9  # One point away from winning

        result = game._determine_hand_winner()

        assert result["game_over"] is True
        assert game.state.phase == GamePhase.GAME_OVER
        assert game.state.team1_score >= 10

    def test_dealer_rotates_after_hand(self):
        """Dealer position should rotate after each hand"""
        game = EuchreGame("test-game")
        game.add_player("Player 1", PlayerType.HUMAN)
        game.add_player("Player 2", PlayerType.HUMAN)
        game.add_player("Player 3", PlayerType.HUMAN)
        game.add_player("Player 4", PlayerType.HUMAN)

        game.start_new_hand()

        initial_dealer = game.state.dealer_position

        # Set up game state to complete a hand
        game.state.trump = Suit.HEARTS
        game.state.trump_caller_position = 0
        game.state.team1_tricks = 3
        game.state.team2_tricks = 2

        game._determine_hand_winner()

        # Dealer should have rotated
        assert game.state.dealer_position == (initial_dealer + 1) % 4
