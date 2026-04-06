"""
Genetisk Algoritme til Snake AI.
Uændret logik - fitness evaluering sker nu i Trainer.
"""

import random
import numpy as np
from typing import List, Tuple


class SnakeAISpecie:
    """Repræsenterer ét individ: DNA er neural network vægte."""

    def __init__(self, genome_len: int, DNA: np.ndarray | None = None):
        self.genome_len = genome_len
        self.fitness_score = 0.0
        
        if DNA is not None:
            self.DNA = DNA
        else:
            self.DNA = np.random.uniform(-1.0, 1.0, size=genome_len)

    def __str__(self) -> str:
        return f"SnakeAI({len(self.DNA)} weights, fitness={self.fitness_score:.0f})"
    
    def __lt__(self, other: 'SnakeAISpecie') -> bool:
        return self.fitness_score < other.fitness_score

    def mutate(self, mutation_rate: float, mutation_strength: float = 0.5) -> None:
        """Tilføj støj til vægte baseret på mutation_rate."""
        mask = np.random.rand(self.genome_len) < mutation_rate
        noise = np.random.randn(self.genome_len) * mutation_strength
        self.DNA[mask] += noise[mask]

    @classmethod
    def copulate(
        cls, 
        parents: Tuple['SnakeAISpecie', 'SnakeAISpecie'], 
        genome_len: int
    ) -> 'SnakeAISpecie':
        """Skab nyt individ ved crossover."""
        DNA_p1, DNA_p2 = parents[0].DNA, parents[1].DNA
        
        # Uniform crossover
        mask = np.random.rand(genome_len) < 0.5
        baby_DNA = np.where(mask, DNA_p1, DNA_p2)
        
        return cls(genome_len=genome_len, DNA=baby_DNA)


class Population:
    """Håndterer evolution af population."""

    def __init__(
        self, 
        population_size: int, 
        genome_len: int,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.5,
        survival_rate: float = 0.5
    ):
        self.population_size = population_size
        self.genome_len = genome_len
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.survival_rate = survival_rate
        
        self.population: List[SnakeAISpecie] = [
            SnakeAISpecie(genome_len=genome_len) 
            for _ in range(population_size)
        ]

    def mutate(self) -> None:
        """Mutér hele populationen."""
        for specie in self.population:
            specie.mutate(self.mutation_rate, self.mutation_strength)

    def selection(self) -> None:
        """Udvælg de bedste og skab ny generation."""
        # Sorter efter fitness
        self.population.sort(key=lambda s: s.fitness_score, reverse=True)
        
        # Survivors
        num_survivors = int(len(self.population) * self.survival_rate)
        survivors = self.population[:num_survivors]
        
        # Ny population
        new_population: List[SnakeAISpecie] = list(survivors)
        
        while len(new_population) < self.population_size:
            parents = tuple(random.sample(survivors, 2))
            baby = SnakeAISpecie.copulate(parents, self.genome_len)
            new_population.append(baby)

        self.population = new_population

    def get_best_specimen(self) -> SnakeAISpecie:
        """Returner individ med højeste fitness."""
        return max(self.population, key=lambda s: s.fitness_score)
    
    def get_stats(self) -> dict:
        """Returner population statistik."""
        fitness_scores = [s.fitness_score for s in self.population]
        return {
            "best": max(fitness_scores),
            "avg": np.mean(fitness_scores),
            "worst": min(fitness_scores),
            "std": np.std(fitness_scores),
        }