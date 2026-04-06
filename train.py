"""
Træningsmodul til Snake AI.
Kan importeres i notebook eller køres fra CLI.
"""

import numpy as np
import time
import json
import os
from typing import List, Dict, Callable
from dataclasses import dataclass, field, asdict

from configs.config import get_config
from genetic_algo import Population
from snake_game import SnakeGame
from nn_model import NeuralNetwork


@dataclass
class TrainingMetrics:
    """Holder styr på alle metrics under træning."""
    generations: List[int] = field(default_factory=list)
    
    # Fitness
    best_fitness: List[float] = field(default_factory=list)
    avg_fitness: List[float] = field(default_factory=list)
    
    # Score (mad spist)
    best_score: List[int] = field(default_factory=list)
    avg_score: List[float] = field(default_factory=list)
    
    # Moves
    best_moves: List[int] = field(default_factory=list)
    avg_moves: List[float] = field(default_factory=list)
    
    # Effektivitet
    best_moves_per_food: List[float] = field(default_factory=list)
    
    # Diversity
    diversity: List[float] = field(default_factory=list)
    
    # Death reasons (per generation)
    deaths_wall: List[int] = field(default_factory=list)
    deaths_tail: List[int] = field(default_factory=list)
    deaths_loop: List[int] = field(default_factory=list)
    deaths_max_moves: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    """
    Håndterer træning af Snake AI.
    Kan bruges fra notebook eller CLI.
    """
    
    def __init__(self, config_name: str = "baseline"):
        self.config = get_config(config_name)
        self.metrics = TrainingMetrics()
        self.population = None
        self.game = None
        self.best_ever_fitness = 0
        self.results_dir = f"results/{config_name}"
        
        # Callbacks for custom logging/plotting
        self.on_generation: Callable = None
        
    def setup(self):
        """Initialiser game og population."""
        self.game = SnakeGame(
            xsize=self.config["grid_size"],
            ysize=self.config["grid_size"],
            visual=False
        )
        self.population = Population(
            population_size=self.config["population_size"],
            genome_len=self.config["genome_len"]
        )
        
        # Opret results mappe
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f" Træning: {self.config['name']}")
        print(f"   Population: {self.config['population_size']}")
        print(f"   Hidden nodes: {self.config['hidden_nodes']}")
        print(f"   Input nodes: {self.config['input_nodes']}")
        print(f"   Generations: {self.config['generations']}")
        print(f"   Results: {self.results_dir}/")
        
    def _update_metrics(self, gen: int, generation_results: List):
        """Opdater metrics for generation med detaljerede stats."""
        
        # Fitness stats
        fitness_scores = [r.fitness for r in generation_results]
        scores = [r.score for r in generation_results]
        moves = [r.moves for r in generation_results]
        
        # Find best individual
        best_idx = np.argmax(fitness_scores)
        best_result = generation_results[best_idx]
        
        # Death reasons
        death_counts = {"wall": 0, "tail": 0, "loop": 0, "max_moves": 0}
        for r in generation_results:
            death_counts[r.death_reason] = death_counts.get(r.death_reason, 0) + 1
        
        # Diversity
        all_weights = np.array([s.DNA for s in self.population.population])
        diversity = np.mean(np.std(all_weights, axis=0))
        
        # Gem metrics
        self.metrics.generations.append(gen)
        
        self.metrics.best_fitness.append(max(fitness_scores))
        self.metrics.avg_fitness.append(np.mean(fitness_scores))
        
        self.metrics.best_score.append(best_result.score)
        self.metrics.avg_score.append(np.mean(scores))
        
        self.metrics.best_moves.append(best_result.moves)
        self.metrics.avg_moves.append(np.mean(moves))
        
        self.metrics.best_moves_per_food.append(best_result.moves_per_food if best_result.score > 0 else 0)
        
        self.metrics.diversity.append(diversity)
        
        self.metrics.deaths_wall.append(death_counts["wall"])
        self.metrics.deaths_tail.append(death_counts["tail"])
        self.metrics.deaths_loop.append(death_counts["loop"])
        self.metrics.deaths_max_moves.append(death_counts["max_moves"])
        
    def train(
        self, 
        print_every: int = 10,
        save_every: int = 25,
        callback_every: int = 5
    ):
        """Kør træningsløkke."""
        if self.population is None:
            self.setup()
            
        start_time = time.time()
        cfg = self.config
        
        print("=" * 50)
        print("🚀 STARTER TRÆNING")
        print("=" * 50)
        
        for gen in range(1, cfg["generations"] + 1):
            gen_start = time.time()
            
            # Adaptive mutation rate
            progress = gen / cfg["generations"]
            self.population.mutation_rate = (
                cfg["initial_mutation_rate"] * (1 - progress) + 
                cfg["final_mutation_rate"] * progress
            )
            
            # Evaluate med detailed results
            generation_results = []
            for specie in self.population.population:
                nn = NeuralNetwork.from_config(cfg, specie.DNA)
                result = self.game.run_headless_detailed(
                    nn,
                    max_moves=cfg["max_moves"],
                    fitness_config=cfg["fitness"],
                    input_mode=cfg["input_nodes"]
                )
                specie.fitness_score = result.fitness
                generation_results.append(result)
            
            # Selection & Mutation
            self.population.selection()
            self.population.mutate()
            
            # Track metrics
            self._update_metrics(gen, generation_results)
            
            # Print progress
            if gen == 1 or gen % print_every == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / gen) * (cfg["generations"] - gen)
                
                print(f"Gen {gen:3d}/{cfg['generations']} | "
                      f"Best: {self.metrics.best_fitness[-1]:8.0f} | "
                      f"Score: {self.metrics.best_score[-1]:2d} | "
                      f"Avg Score: {self.metrics.avg_score[-1]:.1f} | "
                      f"Div: {self.metrics.diversity[-1]:.3f} | "
                      f"ETA: {eta/60:.1f}min")
            
            # Callback
            if self.on_generation and gen % callback_every == 0:
                self.on_generation(self, gen)
            
            # Save checkpoints
            best_fitness = self.metrics.best_fitness[-1]
            if gen % save_every == 0 or best_fitness > self.best_ever_fitness:
                if best_fitness > self.best_ever_fitness:
                    self.best_ever_fitness = best_fitness
                    self._save_model(gen, is_best=True)
                else:
                    self._save_model(gen)
        
        # Final save
        total_time = time.time() - start_time
        print("=" * 50)
        print(f"🏁 TRÆNING FÆRDIG!")
        print(f"   Tid: {total_time/60:.1f} min")
        print(f"   Best fitness: {self.best_ever_fitness:.0f}")
        print(f"   Best score: {max(self.metrics.best_score)}")
        print("=" * 50)
        
        self._save_results()
        return self.metrics
    
    def _save_model(self, gen: int, is_best: bool = False):
        """Gem model vægte."""
        best = self.population.get_best_specimen()
        prefix = "best" if is_best else f"gen{gen}"
        path = f"{self.results_dir}/{prefix}_snake.npy"
        np.save(path, best.DNA)
        
    def _save_results(self):
        """Gem metrics og config."""
        # Metrics
        with open(f"{self.results_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        # Config snapshot
        with open(f"{self.results_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
            
        print(f" Resultater gemt i {self.results_dir}/")
        
        # Generer plots
        try:
            from plot_utils import plot_single_experiment
            plot_single_experiment(self.config["name"], show=False)
        except ImportError:
            print("  plot_utils.py ikke fundet - skip plot generering")


def run_experiment(config_name: str, **kwargs) -> TrainingMetrics:
    """Convenience function til at køre et eksperiment."""
    trainer = Trainer(config_name)
    return trainer.train(**kwargs)


# CLI support
if __name__ == "__main__":
    import sys
    config_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    run_experiment(config_name)