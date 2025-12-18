"""
Genetic Algorithm for Evolving Euchre Neural Networks
"""

import random
import torch
import copy
from typing import List
from networks.basic_nn import BasicEuchreNN


class GeneticAlgorithm:
    """Genetic algorithm for evolving neural network weights"""
    
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7, elite_size=2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population: List[BasicEuchreNN] = []
        self.fitness_scores: List[float] = []
    
    def initialize_population(self):
        """Create initial random population"""
        self.population = [BasicEuchreNN() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, model: BasicEuchreNN, num_games=10) -> float:
        """
        Evaluate fitness of a model by having it play games.
        
        Returns:
            Fitness score (win rate)
        """
        # Play games against other models or baseline
        # Return win percentage
        return random.random()  # Placeholder
    
    def selection(self) -> List[BasicEuchreNN]:
        """Select parents for next generation using tournament selection"""
        parents = []
        
        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(list(zip(self.population, self.fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: BasicEuchreNN, parent2: BasicEuchreNN) -> BasicEuchreNN:
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
    
    def evolve(self, generations=10):
        """Run the genetic algorithm for specified generations"""
        self.initialize_population()
        
        for generation in range(generations):
            # Evaluate fitness
            self.fitness_scores = [
                self.evaluate_fitness(model) for model in self.population
            ]
            
            # Sort by fitness
            sorted_pop = sorted(
                zip(self.population, self.fitness_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep elite
            elites = [model for model, _ in sorted_pop[:self.elite_size]]
            
            # Select parents
            parents = self.selection()
            
            # Create offspring
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = self.crossover(parents[i], parents[i+1])
                    child2 = self.crossover(parents[i+1], parents[i])
                    self.mutate(child1)
                    self.mutate(child2)
                    offspring.extend([child1, child2])
            
            # New population
            self.population = elites + offspring[:self.population_size - self.elite_size]
            
            best_fitness = sorted_pop[0][1]
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")
        
        return sorted_pop[0][0]  # Return best model
