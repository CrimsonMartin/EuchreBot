"""
Card and Suit definitions for Euchre
"""

from enum import Enum
from typing import Optional


class Suit(Enum):
    """Card suits in Euchre"""
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"

    def __str__(self):
        symbols = {
            "C": "♣",
            "D": "♦",
            "H": "♥",
            "S": "♠"
        }
        return symbols[self.value]

    @classmethod
    def from_string(cls, s: str) -> "Suit":
        """Create Suit from string representation"""
        mapping = {
            "C": cls.CLUBS,
            "D": cls.DIAMONDS,
            "H": cls.HEARTS,
            "S": cls.SPADES,
            "♣": cls.CLUBS,
            "♦": cls.DIAMONDS,
            "♥": cls.HEARTS,
            "♠": cls.SPADES,
        }
        return mapping[s.upper()]

    def opposite(self) -> "Suit":
        """Get the opposite color suit (for determining left bower)"""
        opposites = {
            Suit.CLUBS: Suit.SPADES,
            Suit.SPADES: Suit.CLUBS,
            Suit.DIAMONDS: Suit.HEARTS,
            Suit.HEARTS: Suit.DIAMONDS,
        }
        return opposites[self]


class Rank(Enum):
    """Card ranks in Euchre (9 through Ace)"""
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self):
        names = {
            9: "9",
            10: "10",
            11: "J",
            12: "Q",
            13: "K",
            14: "A"
        }
        return names[self.value]

    @classmethod
    def from_string(cls, s: str) -> "Rank":
        """Create Rank from string representation"""
        mapping = {
            "9": cls.NINE,
            "10": cls.TEN,
            "J": cls.JACK,
            "Q": cls.QUEEN,
            "K": cls.KING,
            "A": cls.ACE,
        }
        return mapping[s.upper()]


class Card:
    """Represents a single card in Euchre"""

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank.name})"

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def is_bower(self, trump: Suit) -> bool:
        """Check if this card is a bower (Jack of trump or same color)"""
        return self.rank == Rank.JACK and (
            self.suit == trump or self.suit == trump.opposite()
        )

    def is_right_bower(self, trump: Suit) -> bool:
        """Check if this card is the right bower (Jack of trump suit)"""
        return self.rank == Rank.JACK and self.suit == trump

    def is_left_bower(self, trump: Suit) -> bool:
        """Check if this card is the left bower (Jack of same color)"""
        return self.rank == Rank.JACK and self.suit == trump.opposite()

    def effective_suit(self, trump: Optional[Suit]) -> Suit:
        """Get the effective suit of the card (left bower counts as trump)"""
        if trump and self.is_left_bower(trump):
            return trump
        return self.suit

    def value(self, trump: Suit, lead_suit: Optional[Suit] = None) -> int:
        """
        Get card value for comparison in a trick.
        Higher value wins the trick.
        """
        # Right bower is highest
        if self.is_right_bower(trump):
            return 1000

        # Left bower is second highest
        if self.is_left_bower(trump):
            return 900

        effective_suit = self.effective_suit(trump)

        # Trump suit (non-bower)
        if effective_suit == trump:
            return 100 + self.rank.value

        # Lead suit (if not trump)
        if lead_suit and effective_suit == lead_suit:
            return self.rank.value

        # Off-suit (can't win)
        return 0

    @classmethod
    def from_string(cls, s: str) -> "Card":
        """
        Create a Card from string like '9C', 'AS', 'JH', etc.
        """
        if len(s) < 2:
            raise ValueError(f"Invalid card string: {s}")
        
        rank_str = s[:-1]
        suit_str = s[-1]
        
        rank = Rank.from_string(rank_str)
        suit = Suit.from_string(suit_str)
        
        return cls(suit, rank)
