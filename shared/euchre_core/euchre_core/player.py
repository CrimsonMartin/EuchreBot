"""
Player abstraction for Euchre
"""

from enum import Enum
from typing import List, Optional
from .card import Card, Suit


class PlayerType(Enum):
    """Types of players in the game"""
    HUMAN = "human"
    RANDOM_AI = "random_ai"
    NEURAL_NET_AI = "neural_net_ai"


class Player:
    """Represents a player in a Euchre game"""

    def __init__(self, name: str, player_type: PlayerType, position: int):
        """
        Initialize a player.
        
        Args:
            name: Player's display name
            player_type: Type of player (human, AI, etc.)
            position: Position at table (0-3)
        """
        self.name = name
        self.player_type = player_type
        self.position = position
        self.hand: List[Card] = []
        self.ai_model_id: Optional[str] = None  # For neural net players

    def add_cards(self, cards: List[Card]):
        """Add cards to player's hand"""
        self.hand.extend(cards)

    def remove_card(self, card: Card) -> Card:
        """Remove and return a card from the player's hand"""
        if card not in self.hand:
            raise ValueError(f"Card {card} not in hand")
        self.hand.remove(card)
        return card

    def has_card(self, card: Card) -> bool:
        """Check if player has a specific card"""
        return card in self.hand

    def clear_hand(self):
        """Remove all cards from hand"""
        self.hand = []

    def get_valid_cards(self, lead_suit: Optional[Suit], trump: Optional[Suit]) -> List[Card]:
        """
        Get list of valid cards that can be played.
        
        In Euchre, you must follow suit if possible.
        Left bower is considered trump suit for following purposes.
        """
        if not lead_suit:
            # First card of trick - any card is valid
            return self.hand.copy()

        # Find cards that match the lead suit (accounting for left bower)
        valid_cards = []
        for card in self.hand:
            effective_suit = card.effective_suit(trump)
            if effective_suit == lead_suit:
                valid_cards.append(card)

        # If player has cards in lead suit, must play one
        if valid_cards:
            return valid_cards

        # Otherwise, can play any card
        return self.hand.copy()

    def hand_str(self) -> str:
        """Get string representation of hand"""
        return " ".join(str(card) for card in sorted(
            self.hand, 
            key=lambda c: (c.suit.value, c.rank.value)
        ))

    def __str__(self):
        return f"{self.name} ({self.player_type.value})"

    def __repr__(self):
        return f"Player(name={self.name}, type={self.player_type}, pos={self.position})"

    def to_dict(self):
        """Convert player to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "type": self.player_type.value,
            "position": self.position,
            "hand_size": len(self.hand),
            "ai_model_id": self.ai_model_id,
        }
