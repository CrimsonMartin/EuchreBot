"""
Genetic Algorithm for Evolving Euchre Neural Networks
"""

import random
import torch
import copy
import sys
import os
from typing import List

# Add parent directory to path to import euchre_core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from euchre_core.game import EuchreGame, GamePhase
from euchre_core.player import PlayerType
from euchre_core.card import Card, Suit
from networks.basic_nn import BasicEuchreNN, encode_game_state


class GeneticAlgorithm:
    """Genetic algorithm for evolving neural network weights"""

    def __init__(
        self, population_size=20, mutation_rate=0.1, crossover_rate=0.7, elite_size=2
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population: List[BasicEuchreNN] = []
        self.fitness_scores: List[float] = []

        # Card mapping for predictions
        self.all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                self.all_cards.append(f"{rank}{suit}")

    def initialize_population(self):
        """Create initial random population"""
        self.population = [BasicEuchreNN() for _ in range(self.population_size)]

    def play_game_with_model(
        self,
        model: BasicEuchreNN,
        opponent_models: List[BasicEuchreNN],
        model_position: int = 0,
    ) -> int:
        """
        Play a single game with the model at a specific position.

        Args:
            model: The model being evaluated
            opponent_models: List of 3 opponent models
            model_position: Position (0-3) where the model plays

        Returns:
            1 if model's team won, 0 if lost
        """
        # Create game
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        # Add players (all AI for training)
        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Start first hand
        game.start_new_hand()

        # Play until game over
        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position

            # Determine which model to use
            if current_pos == model_position:
                current_model = model
            else:
                # Map other positions to opponent models
                opponent_idx = (
                    current_pos if current_pos < model_position else current_pos - 1
                )
                opponent_idx = min(opponent_idx, len(opponent_models) - 1)
                current_model = opponent_models[opponent_idx]

            # Handle different game phases
            if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                # Simple strategy: pass for now (can be improved)
                try:
                    game.pass_trump()
                except:
                    # Dealer must call
                    game.call_trump(game.state.turned_up_card.suit)

            elif game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
                # Simple strategy: call a random suit (not turned up)
                available_suits = [
                    s
                    for s in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    if s != game.state.turned_up_card.suit
                ]
                if game.state.current_player_position == game.state.dealer_position:
                    # Dealer must call
                    game.call_trump(random.choice(available_suits))
                else:
                    try:
                        game.pass_trump()
                    except:
                        game.call_trump(random.choice(available_suits))

            elif game.state.phase == GamePhase.DEALER_DISCARD:
                # Dealer discards lowest card
                dealer = game.state.get_player(game.state.dealer_position)
                card_to_discard = dealer.hand[0]  # Simple: discard first card
                game.dealer_discard(card_to_discard)

            elif game.state.phase == GamePhase.PLAYING:
                # Use neural network to select card
                game_state_dict = game.get_state(perspective_position=current_pos)
                state_encoding = encode_game_state(game_state_dict)

                # Get valid moves
                valid_cards = game.get_valid_moves(current_pos)

                if valid_cards:
                    # Get model prediction
                    card_idx = current_model.predict_card(state_encoding)
                    predicted_card_str = (
                        self.all_cards[card_idx]
                        if card_idx < len(self.all_cards)
                        else None
                    )

                    # Check if predicted card is valid
                    selected_card = None
                    if predicted_card_str:
                        for card in valid_cards:
                            if str(card) == predicted_card_str:
                                selected_card = card
                                break

                    # If prediction not valid, pick first valid card
                    if selected_card is None:
                        selected_card = valid_cards[0]

                    # Play the card
                    result = game.play_card(selected_card)

                    # Check if hand complete
                    if result.get("hand_complete"):
                        # Start new hand if game not over
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()

            elif game.state.phase == GamePhase.HAND_COMPLETE:
                # Start new hand
                game.start_new_hand()

        # Determine if model's team won
        model_team = game.state.get_team_for_position(model_position)
        if model_team == 1:
            return 1 if game.state.team1_score >= 10 else 0
        else:
            return 1 if game.state.team2_score >= 10 else 0

    def evaluate_fitness(self, model: BasicEuchreNN, num_games=10) -> float:
        """
        Evaluate fitness of a model by having it play games.

        Args:
            model: The model to evaluate
            num_games: Number of games to play for evaluation

        Returns:
            Fitness score (win rate)
        """
        wins = 0

        # Play multiple games against random opponents from population
        for _ in range(num_games):
            # Select 3 random opponent models
            opponent_models = random.sample(
                self.population, min(3, len(self.population))
            )
            if len(opponent_models) < 3:
                # Fill with copies if not enough models
                while len(opponent_models) < 3:
                    opponent_models.append(random.choice(self.population))

            # Play from random position
            position = random.randint(0, 3)

            try:
                result = self.play_game_with_model(model, opponent_models, position)
                wins += result
            except Exception as e:
                # If game fails, count as loss
                print(f"Game error: {e}")
                pass

        return wins / num_games if num_games > 0 else 0.0

    def selection(self) -> List[BasicEuchreNN]:
        """Select parents for next generation using tournament selection"""
        parents = []

        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(
                list(zip(self.population, self.fitness_scores)),
                min(tournament_size, len(self.population)),
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)

        return parents

    def crossover(
        self, parent1: BasicEuchreNN, parent2: BasicEuchreNN
    ) -> BasicEuchreNN:
        """Crossover two parent networks to create offspring"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)

        child = BasicEuchreNN()

        # Crossover weights
        for child_param, p1_param, p2_param in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            # Uniform crossover
            mask = torch.rand_like(child_param) > 0.5
            child_param.data = torch.where(mask, p1_param.data, p2_param.data)

        return child

    def mutate(self, model: BasicEuchreNN):
        """Mutate model weights"""
        for param in model.parameters():
            if random.random() < self.mutation_rate:
                # Add Gaussian noise to weights
                noise = torch.randn_like(param) * 0.1
                param.data += noise

    def evolve(self, generations=10, callback=None):
        """
        Run the genetic algorithm for specified generations.

        Args:
            generations: Number of generations to evolve
            callback: Optional callback function called each generation with (gen, best_fitness, avg_fitness)

        Returns:
            Best model from final generation
        """
        self.initialize_population()

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")

            # Evaluate fitness
            self.fitness_scores = []
            for i, model in enumerate(self.population):
                fitness = self.evaluate_fitness(model, num_games=5)
                self.fitness_scores.append(fitness)
                print(f"  Model {i+1}/{self.population_size}: Fitness = {fitness:.3f}")

            # Sort by fitness
            sorted_pop = sorted(
                zip(self.population, self.fitness_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            best_fitness = sorted_pop[0][1]
            avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)

            print(
                f"Generation {generation + 1}: Best = {best_fitness:.3f}, Avg = {avg_fitness:.3f}"
            )

            # Call callback if provided
            if callback:
                callback(generation + 1, best_fitness, avg_fitness)

            # Keep elite
            elites = [
                copy.deepcopy(model) for model, _ in sorted_pop[: self.elite_size]
            ]

            # Select parents
            parents = self.selection()

            # Create offspring
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = self.crossover(parents[i], parents[i + 1])
                    child2 = self.crossover(parents[i + 1], parents[i])
                    self.mutate(child1)
                    self.mutate(child2)
                    offspring.extend([child1, child2])

            # New population
            self.population = (
                elites + offspring[: self.population_size - self.elite_size]
            )

        # Return best model
        final_sorted = sorted(
            zip(self.population, self.fitness_scores), key=lambda x: x[1], reverse=True
        )
        return final_sorted[0][0]
