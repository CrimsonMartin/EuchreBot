"""
Trick management for Euchre
"""

from typing import List, Optional, Tuple
from .card import Card, Suit
from .player import Player


class Trick:
    """Represents a single trick in Euchre"""

    def __init__(self, lead_player_position: int, trump: Suit):
        """
        Initialize a trick.
        
        Args:
            lead_player_position: Position of the player who leads
            trump: The trump suit for this hand
        """
        self.lead_position = lead_player_position
        self.trump = trump
        self.cards: List[Tuple[int, Card]] = []  # (player_position, card)
        self.lead_suit: Optional[Suit] = None

    def add_card(self, player_position: int, card: Card):
        """Add a card played by a player to this trick"""
        if len(self.cards) >= 4:
            raise ValueError("Trick already has 4 cards")
        
        # First card determines lead suit
        if not self.cards:
            self.lead_suit = card.effective_suit(self.trump)
        
        self.cards.append((player_position, card))

    def is_complete(self) -> bool:
        """Check if all 4 players have played"""
        return len(self.cards) == 4

    def get_winner(self) -> int:
        """
        Determine which player won the trick.
        Returns the position of the winning player.
        """
        if not self.is_complete():
            raise ValueError("Cannot determine winner - trick not complete")

        winning_position = self.cards[0][0]
        winning_card = self.cards[0][1]
        winning_value = winning_card.value(self.trump, self.lead_suit)

        for position, card in self.cards[1:]:
            card_value = card.value(self.trump, self.lead_suit)
            if card_value > winning_value:
                winning_value = card_value
                winning_card = card
                winning_position = position

        return winning_position

    def get_card_for_position(self, position: int) -> Optional[Card]:
        """Get the card played by a specific player position"""
        for pos, card in self.cards:
            if pos == position:
                return card
        return None

    def __str__(self):
        cards_str = ", ".join(f"P{pos}: {card}" for pos, card in self.cards)
        return f"Trick(lead={self.lead_position}, trump={self.trump}, cards=[{cards_str}])"

    def to_dict(self):
        """Convert trick to dictionary for JSON serialization"""
        return {
            "lead_position": self.lead_position,
            "trump": self.trump.value,
            "lead_suit": self.lead_suit.value if self.lead_suit else None,
            "cards": [{"position": pos, "card": str(card)} for pos, card in self.cards],
            "is_complete": self.is_complete(),
            "winner": self.get_winner() if self.is_complete() else None,
        }
