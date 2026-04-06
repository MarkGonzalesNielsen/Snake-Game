"""
Central konfiguration for Snake AI eksperimenter.
Alle hyperparametre samlet ét sted for nem sammenligning.
"""

EXPERIMENTS = {
    "baseline": {
        # Neural Network
        "hidden_nodes": 8,
        "activation": "softmax",
        
        # Population / Genetisk Algoritme
        "population_size": 200,
        "generations": 100,
        "initial_mutation_rate": 0.1,
        "final_mutation_rate": 0.02,
        "mutation_strength": 0.5,
        "survival_rate": 0.5,
        
        # Game Settings
        "grid_size": 15,
        "max_moves": 500,
        
        # Fitness Funktion
        "fitness": {
            "score_multiplier": 5000,
            "score_exponent": 2,
            "move_bonus": 2,
            "proximity_multiplier": 50,
            "loop_death_factor": 30,
        }
    },
    
    "larger_pop": {
        "population_size": 100,
    },
    
    "larger_nn": {
        "hidden_nodes": 16,
    },
    
    "aggressive_fitness": {
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 1,
            "proximity_multiplier": 30,
            "loop_death_factor": 20,
        }
    },
    
    "high_mutation": {
        "initial_mutation_rate": 0.2,
        "final_mutation_rate": 0.05,
        "mutation_strength": 0.7,
    },
    
    "small_grid": {
        "grid_size": 10,
        "max_moves": 300,
    },
    
    # === V2 EKSPERIMENTER (anti-loop) ===
    
    "baseline_v2": {
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 100,
            "loop_death_factor": 20,
        }
    },
    
    "larger_pop_v2": {
        "population_size": 100,
        "generations": 200,
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 100,
            "loop_death_factor": 20,
        }
    },
    
    "larger_nn_v2": {
        "hidden_nodes": 16,
        "generations": 200,
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 100,
            "loop_death_factor": 20,
        }
    },
    
    "franken": {
        "hidden_nodes": 16,
        "population_size": 100,
        "generations": 400,
        "initial_mutation_rate": 0.20,
        "final_mutation_rate": 0.02,
        "mutation_strength": 0.4,
        "survival_rate": 0.35,
        "grid_size": 15,
        "max_moves": 600,
        
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 100,
            "loop_death_factor": 25,
        }
    },
    
    "optimized": {
        "hidden_nodes": 8,
        "population_size": 150,
        "generations": 300,
        "initial_mutation_rate": 0.15,
        "final_mutation_rate": 0.02,
        "mutation_strength": 0.4,
        "survival_rate": 0.4,
        "grid_size": 15,
        "max_moves": 600,
        
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 120,
            "loop_death_factor": 15,
        }
    },
    
    # === V3 EKSPERIMENTER (24 inputs med hale-vision) ===
    
    "franken_v3": {
        "input_nodes": 24,
        "hidden_nodes": 18,
        "activation": "softmax",
        
        "population_size": 150,
        "generations": 500,
        "initial_mutation_rate": 0.20,
        "final_mutation_rate": 0.02,
        "mutation_strength": 0.4,
        "survival_rate": 0.30,
        
        "grid_size": 15,
        "max_moves": 200,  # Base moves
        
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 100,
            "loop_death_factor": 40,
            "moves_per_food": 50,  # +50 moves per mad spist
        }
    },

    "franken_v4": {
        "input_nodes": 24,
        "hidden_nodes": 18,
        "activation": "softmax",
        
        "population_size": 200,
        "generations": 700,
        "initial_mutation_rate": 0.20,
        "final_mutation_rate": 0.02,
        "mutation_strength": 0.4,
        "survival_rate": 0.30,
        
        "grid_size": 15,
        "max_moves": 200,  # Base moves
        
        "fitness": {
            "score_multiplier": 10000,
            "score_exponent": 2.5,
            "move_bonus": 0,
            "proximity_multiplier": 100,
            "loop_death_factor": 40,
            "moves_per_food": 50,  # +50 moves per mad spist
        }
    },
}


def get_config(name: str) -> dict:
    import copy
    
    if name not in EXPERIMENTS:
        raise ValueError(f"Ukendt eksperiment: {name}. Valg: {list(EXPERIMENTS.keys())}")
    
    config = copy.deepcopy(EXPERIMENTS["baseline"])
    
    if name != "baseline":
        exp = EXPERIMENTS[name]
        for key, value in exp.items():
            if key == "fitness" and "fitness" in config:
                config["fitness"].update(value)
            else:
                config[key] = value
    
    config["name"] = name
    config["input_nodes"] = config.get("input_nodes", 11)
    config["output_nodes"] = 3
    config["genome_len"] = (config["input_nodes"] * config["hidden_nodes"]) + \
                           (config["hidden_nodes"] * config["output_nodes"])
    
    return config


def list_experiments() -> None:
    print(" Tilgængelige eksperimenter:")
    print("-" * 40)
    for name in EXPERIMENTS.keys():
        cfg = get_config(name)
        inputs = cfg.get('input_nodes', 11)
        print(f"  {name:20} | pop={cfg['population_size']:3} | hidden={cfg['hidden_nodes']:2} | inputs={inputs:2} | grid={cfg['grid_size']}")


if __name__ == "__main__":
    list_experiments()