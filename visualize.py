"""
Visualisering af trænede Snake AI modeller.
Understøtter både 11 og 24 input modes.
"""

import numpy as np
import pygame
import time
from typing import Optional, Set

from configs.config import get_config
from nn_model import NeuralNetwork
from snake_game import Vector, Snake, Food


class SnakeVisualizer:
    """Visualiser en trænet Snake AI model."""
    
    def __init__(self, experiment_name: str, results_dir: str = "results"):
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.config = get_config(experiment_name)
        
        # Load best model
        weights_path = f"{results_dir}/{experiment_name}/best_snake.npy"
        self.weights = np.load(weights_path)
        self.nn = NeuralNetwork.from_config(self.config, self.weights)
        
        # Game settings
        self.grid_size = self.config["grid_size"]
        self.scale = 30
        self.fps = 10
        self.input_mode = self.config.get("input_nodes", 11)
        
        # Dynamisk max moves settings
        self.base_max_moves = self.config.get("max_moves", 600)
        self.moves_per_food = self.config.get("fitness", {}).get("moves_per_food", 50)
        
    def run(self, speed: int = 10, show_info: bool = True):
        """
        Kør visualisering af modellen.
        
        Args:
            speed: Frames per second (lavere = langsommere)
            show_info: Vis score, moves og andre stats
        """
        self.fps = speed
        
        pygame.init()
        
        info_width = 200 if show_info else 0
        screen_width = self.grid_size * self.scale + info_width
        screen_height = self.grid_size * self.scale
        
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(f"Snake AI - {self.experiment_name}")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
        font_large = pygame.font.Font(None, 36)
        
        snake = self._create_snake()
        food = self._create_food(snake)
        snake.v = Vector(1, 0)
        
        moves = 0
        moves_since_food = 0
        running = True
        game_over = False
        death_reason = ""
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and game_over:
                        snake = self._create_snake()
                        food = self._create_food(snake)
                        snake.v = Vector(1, 0)
                        moves = 0
                        moves_since_food = 0
                        game_over = False
                        death_reason = ""
                    if event.key == pygame.K_UP:
                        self.fps = min(60, self.fps + 5)
                    if event.key == pygame.K_DOWN:
                        self.fps = max(1, self.fps - 5)
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            if not game_over:
                state = self._get_game_state(snake, food)
                action = self.nn.predict(state)
                snake.v = self._translate_action(action, snake.v)
                
                snake.move()
                moves += 1
                moves_since_food += 1
                snake.moves_since_last_food = moves_since_food
                
                if not snake.p.within(Vector(self.grid_size, self.grid_size)):
                    game_over = True
                    death_reason = "WALL"
                elif snake.cross_own_tail:
                    game_over = True
                    death_reason = "TAIL"
                elif moves_since_food > len(snake.body) * self.config["fitness"]["loop_death_factor"]:
                    game_over = True
                    death_reason = "LOOP"
                
                # Dynamisk max_moves: base + bonus per mad spist
                dynamic_max_moves = self.base_max_moves + (snake.score * self.moves_per_food)
                if moves >= dynamic_max_moves:
                    game_over = True
                    death_reason = "MAX MOVES"
                
                if snake.p == food.p:
                    snake.add_score()
                    food = self._create_food(snake)
                    moves_since_food = 0
                    snake.moves_since_last_food = 0
            
            # === RENDERING ===
            screen.fill((17, 24, 39))
            
            game_area = pygame.Rect(0, 0, self.grid_size * self.scale, self.grid_size * self.scale)
            pygame.draw.rect(screen, (31, 41, 55), game_area)
            
            for i in range(self.grid_size + 1):
                pygame.draw.line(screen, (55, 65, 81), 
                               (i * self.scale, 0), 
                               (i * self.scale, self.grid_size * self.scale), 1)
                pygame.draw.line(screen, (55, 65, 81), 
                               (0, i * self.scale), 
                               (self.grid_size * self.scale, i * self.scale), 1)
            
            food_rect = pygame.Rect(food.p.x * self.scale + 2, food.p.y * self.scale + 2, 
                                   self.scale - 4, self.scale - 4)
            pygame.draw.rect(screen, (239, 68, 68), food_rect, border_radius=4)
            
            for i, p in enumerate(snake.body):
                if i == 0:
                    color = (34, 197, 94)
                    rect = pygame.Rect(p.x * self.scale + 1, p.y * self.scale + 1, 
                                      self.scale - 2, self.scale - 2)
                    pygame.draw.rect(screen, color, rect, border_radius=6)
                else:
                    green = max(100, 180 - i * 8)
                    color = (34, green, 80)
                    rect = pygame.Rect(p.x * self.scale + 2, p.y * self.scale + 2, 
                                      self.scale - 4, self.scale - 4)
                    pygame.draw.rect(screen, color, rect, border_radius=4)
            
            if show_info:
                panel_x = self.grid_size * self.scale + 10
                
                title = font_large.render(self.experiment_name, True, (255, 255, 255))
                screen.blit(title, (panel_x, 10))
                
                # Beregn dynamisk max moves til visning
                current_max = self.base_max_moves + (snake.score * self.moves_per_food)
                
                stats = [
                    f"Score: {snake.score}",
                    f"Moves: {moves}",
                    f"Max: {current_max}",
                    f"",
                    f"Speed: {self.fps} FPS",
                    f"",
                    f"Config:",
                    f"  Inputs: {self.input_mode}",
                    f"  Hidden: {self.config['hidden_nodes']}",
                    f"  Pop: {self.config['population_size']}",
                    f"  Grid: {self.grid_size}x{self.grid_size}",
                ]
                
                for i, stat in enumerate(stats):
                    text = font.render(stat, True, (156, 163, 175))
                    screen.blit(text, (panel_x, 50 + i * 22))
                
                controls = [
                    "Controls:",
                    "  Up/Down: Speed",
                    "  SPACE: Restart",
                    "  ESC: Exit"
                ]
                for i, ctrl in enumerate(controls):
                    text = font.render(ctrl, True, (107, 114, 128))
                    screen.blit(text, (panel_x, screen_height - 100 + i * 20))
            
            if game_over:
                overlay = pygame.Surface((self.grid_size * self.scale, self.grid_size * self.scale))
                overlay.fill((0, 0, 0))
                overlay.set_alpha(180)
                screen.blit(overlay, (0, 0))
                
                go_text = font_large.render("GAME OVER", True, (255, 255, 255))
                go_rect = go_text.get_rect(center=(self.grid_size * self.scale // 2, 
                                                   self.grid_size * self.scale // 2 - 40))
                screen.blit(go_text, go_rect)
                
                death_text = font.render(f"Reason: {death_reason}", True, (239, 68, 68))
                death_rect = death_text.get_rect(center=(self.grid_size * self.scale // 2, 
                                                         self.grid_size * self.scale // 2))
                screen.blit(death_text, death_rect)
                
                score_text = font.render(f"Final Score: {snake.score} | Moves: {moves}", True, (156, 163, 175))
                score_rect = score_text.get_rect(center=(self.grid_size * self.scale // 2, 
                                                         self.grid_size * self.scale // 2 + 30))
                screen.blit(score_text, score_rect)
                
                hint_text = font.render("Press SPACE to restart", True, (107, 114, 128))
                hint_rect = hint_text.get_rect(center=(self.grid_size * self.scale // 2, 
                                                       self.grid_size * self.scale // 2 + 70))
                screen.blit(hint_text, hint_rect)
            
            pygame.display.flip()
            clock.tick(self.fps)
        
        pygame.quit()
    
    def _create_snake(self) -> Snake:
        """Opret snake med dummy game reference."""
        class DummyGame:
            def __init__(self, grid_size):
                self.grid = Vector(grid_size, grid_size)
        
        dummy = DummyGame(self.grid_size)
        return Snake(game=dummy)
    
    def _create_food(self, snake: Snake = None):
        """Opret food der ikke spawner på slangen."""
        class DummyFood:
            def __init__(self, grid_size, snake):
                while True:
                    self.p = Vector.random_within(Vector(grid_size, grid_size))
                    if snake is None or self.p not in snake.body:
                        break
        
        return DummyFood(self.grid_size, snake)
    
    def _get_game_state(self, snake: Snake, food) -> list:
        """Hent game state baseret på input_mode."""
        if self.input_mode == 24:
            return self._get_state_24(snake, food)
        else:
            return self._get_state_11(snake, food)
    
    def _get_state_11(self, snake: Snake, food) -> list:
        """Original 11-input state."""
        head = snake.p
        v = snake.v
        grid = Vector(self.grid_size, self.grid_size)
        
        state = [
            not (head + v).within(grid),
            not (head + Vector(-v.y, v.x)).within(grid),
            not (head + Vector(v.y, -v.x)).within(grid),
            food.p.y < head.y,
            food.p.y > head.y,
            food.p.x < head.x,
            food.p.x > head.x,
            abs(food.p.x - head.x) / self.grid_size,
            abs(food.p.y - head.y) / self.grid_size,
            len(snake.body) / (self.grid_size * self.grid_size),
            min(snake.moves_since_last_food / 100.0, 1.0)
        ]
        
        return [float(x) for x in state]
    
    def _get_state_24(self, snake: Snake, food) -> list:
        """Udvidet 24-input state med fuld vision."""
        head = snake.p
        v = snake.v
        grid = Vector(self.grid_size, self.grid_size)
        
        directions = [
            Vector(v.x, v.y),                                    # Frem
            Vector(-v.y, v.x),                                   # Venstre
            Vector(v.y, -v.x),                                   # Højre
            Vector(-v.x, -v.y),                                  # Bagud
            self._normalize_diagonal(v.x - v.y, v.y + v.x),      # Frem-venstre
            self._normalize_diagonal(v.x + v.y, v.y - v.x),      # Frem-højre
            self._normalize_diagonal(-v.x - v.y, -v.y + v.x),    # Bag-venstre
            self._normalize_diagonal(-v.x + v.y, -v.y - v.x),    # Bag-højre
        ]
        
        body_set = set((seg.x, seg.y) for seg in snake.body)
        max_dist = float(max(self.grid_size, self.grid_size))
        
        state = []
        for d in directions:
            wall_dist = self._distance_to_wall(head, d, grid)
            body_dist = self._distance_to_body(head, d, body_set, grid)
            food_visible = self._food_in_direction(head, d, food.p)
            
            state.append(wall_dist / max_dist)
            state.append(body_dist / max_dist if body_dist < float('inf') else 1.0)
            state.append(1.0 if food_visible else 0.0)
        
        return state
    
    def _normalize_diagonal(self, x: int, y: int) -> Vector:
        norm_x = 0 if x == 0 else (1 if x > 0 else -1)
        norm_y = 0 if y == 0 else (1 if y > 0 else -1)
        return Vector(norm_x, norm_y)
    
    def _distance_to_wall(self, start: Vector, direction: Vector, grid: Vector) -> int:
        pos = start
        distance = 0
        while True:
            pos = pos + direction
            distance += 1
            if not pos.within(grid):
                return distance
            if distance > grid.x + grid.y:
                return distance
    
    def _distance_to_body(self, start: Vector, direction: Vector, body_set: Set, grid: Vector) -> float:
        pos = start
        distance = 0
        while True:
            pos = pos + direction
            distance += 1
            if not pos.within(grid):
                return float('inf')
            if (pos.x, pos.y) in body_set:
                return distance
            if distance > grid.x + grid.y:
                return float('inf')
    
    def _food_in_direction(self, head: Vector, direction: Vector, food_pos: Vector) -> bool:
        dx = food_pos.x - head.x
        dy = food_pos.y - head.y
        
        if dx == 0 and dy == 0:
            return False
        
        dir_x = direction.x
        dir_y = direction.y
        
        if dir_x == 0 and dir_y != 0:
            return dx == 0 and (dy * dir_y > 0)
        elif dir_y == 0 and dir_x != 0:
            return dy == 0 and (dx * dir_x > 0)
        elif dir_x != 0 and dir_y != 0:
            if dx * dir_x > 0 and dy * dir_y > 0:
                return abs(dx) == abs(dy)
        
        return False
    
    def _translate_action(self, action: int, current_v: Vector) -> Vector:
        if action == 1:
            return current_v
        elif action == 0:
            return Vector(-current_v.y, current_v.x)
        elif action == 2:
            return Vector(current_v.y, -current_v.x)
        return current_v


def watch(experiment_name: str, speed: int = 10):
    """Se en model spille."""
    viz = SnakeVisualizer(experiment_name)
    viz.run(speed=speed)


def compare_models(experiment_names: list, speed: int = 10):
    """Se flere modeller efter hinanden."""
    for name in experiment_names:
        print(f"\n Viser: {name}")
        print("   (Luk vinduet for at se næste model)")
        watch(name, speed)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        watch(sys.argv[1])
    else:
        print("Usage: python visualize.py <experiment_name>")
        print("Example: python visualize.py franken_v3")