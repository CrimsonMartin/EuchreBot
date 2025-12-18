"""
Deck implementation for Euchre
"""

import random
from typing import List
from .card import Card, Suit, Rank


class Deck:
    """A Euchre deck (24 cards: 9-A in each suit)"""

    def __init__(self):
        self.cards: List[Card] = []
        self.reset()

    def reset(self):
        """Reset the deck to a full, ordered euchre deck"""
        self.cards = []
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(suit, rank))

    def shuffle(self):
        """Shuffle the deck"""
        random.shuffle(self.cards)

    def deal(self, num_cards: int) -> List[Card]:
        """
        Deal a specified number of cards from the deck.
        Returns a list of cards and removes them from the deck.
        """
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards, only {len(self.cards)} remaining")
        
        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards

    def deal_hand(self) -> List[Card]:
        """Deal a standard euchre hand (5 cards)"""
        return self.deal(5)

    def top_card(self) -> Card:
        """Get the top card without removing it"""
        if not self.cards:
            raise ValueError("Deck is empty")
        return self.cards[0]

    def draw(self) -> Card:
        """Draw a single card from the top of the deck"""
        if not self.cards:
            raise ValueError("Deck is empty")
        return self.cards.pop(0)

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return f"Deck({len(self.cards)} cards)"

    def __repr__(self):
        return f"Deck(cards={[str(c) for c in self.cards]})"
