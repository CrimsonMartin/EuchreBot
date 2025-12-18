"""
EuchreBot Core Game Engine
"""

from .card import Card, Suit, Rank
from .deck import Deck
from .player import Player, PlayerType
from .trick import Trick
from .game import EuchreGame, GameState, GamePhase

__version__ = "0.1.0"

__all__ = [
    "Card",
    "Suit",
    "Rank",
    "Deck",
    "Player",
    "PlayerType",
    "Trick",
    "EuchreGame",
    "GameState",
    "GamePhase",
]
