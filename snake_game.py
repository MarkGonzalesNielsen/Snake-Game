"""
Snake Game med konfigurerbar fitness-funktion og detaljeret stats.
Understøtter både 11 inputs (legacy) og 24 inputs (v3).
"""

import random
from collections import deque
from typing import List, Dict, Set
import pygame

from nn_model import NeuralNetwork


class Vector:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __str__(self):
        return f'Vector({self.x}, {self.y})'

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def within(self, scope: 'Vector') -> bool:
        return 0 <= self.x < scope.x and 0 <= self.y < scope.y

    def __eq__(self, other: 'Vector') -> bool:
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def manhattan_distance(self, other: 'Vector') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    @classmethod
    def random_within(cls, scope: 'Vector') -> 'Vector':
        return Vector(random.randint(0, scope.x - 1), random.randint(0, scope.y - 1))


class GameResult:
    """Detaljeret resultat fra et spil."""
    def __init__(self, score: int, moves: int, fitness: float, death_reason: str):
        self.score = score
        self.moves = moves
        self.fitness = fitness
        self.death_reason = death_reason
        
    @property
    def moves_per_food(self) -> float:
        return self.moves / self.score if self.score > 0 else float('inf')
    
    def __repr__(self):
        return f"GameResult(score={self.score}, moves={self.moves}, fitness={self.fitness:.0f}, death='{self.death_reason}')"


class SnakeGame:
    def __init__(self, xsize: int = 30, ysize: int = 30, scale: int = 15, visual: bool = True):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        self.visual = visual

        if self.visual:
            pygame.init()
            self.screen = pygame.display.set_mode((xsize * scale, ysize * scale))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Snake AI Training")

    def __del__(self):
        if self.visual:
            pygame.quit()

    def block(self, obj):
        return (obj.x * self.scale, obj.y * self.scale, self.scale, self.scale)

    def translate_action_to_vector(self, action: int, current_v: Vector) -> Vector:
        if action == 1:
            return current_v
        elif action == 0:
            return Vector(-current_v.y, current_v.x)
        elif action == 2:
            return Vector(current_v.y, -current_v.x)
        return current_v

    # ========== INPUT METHODS ==========
    
    def get_game_state(self, snake: 'Snake', food: 'Food', input_mode: int = 11) -> List[float]:
        if input_mode == 24:
            return self._get_state_24(snake, food)
        else:
            return self._get_state_11(snake, food)
    
    def _get_state_11(self, snake: 'Snake', food: 'Food') -> List[float]:
        """Original 11-input state (backward compatible)."""
        head = snake.p
        v = snake.v
        
        state = [
            not (head + v).within(self.grid),
            not (head + Vector(-v.y, v.x)).within(self.grid),
            not (head + Vector(v.y, -v.x)).within(self.grid),
            food.p.y < head.y,
            food.p.y > head.y,
            food.p.x < head.x,
            food.p.x > head.x,
            abs(food.p.x - head.x) / self.grid.x,
            abs(food.p.y - head.y) / self.grid.y,
            len(snake.body) / (self.grid.x * self.grid.y),
            min(snake.moves_since_last_food / 100.0, 1.0)
        ]
        
        return [float(x) for x in state]
    
    def _get_state_24(self, snake: 'Snake', food: 'Food') -> List[float]:
        """
        Udvidet 24-input state med fuld vision.
        
        8 retninger × 3 features:
            - Normaliseret afstand til væg (0 = tæt, 1 = langt)
            - Normaliseret afstand til hale (0 = tæt, 1 = ingen/langt)
            - Mad i denne retning (0 eller 1)
        """
        head = snake.p
        v = snake.v
        
        directions = [
            Vector(v.x, v.y),                                     # Frem
            Vector(-v.y, v.x),                                    # Venstre
            Vector(v.y, -v.x),                                    # Højre
            Vector(-v.x, -v.y),                                   # Bagud
            self._normalize_diagonal(v.x - v.y, v.y + v.x),       # Frem-venstre
            self._normalize_diagonal(v.x + v.y, v.y - v.x),       # Frem-højre
            self._normalize_diagonal(-v.x - v.y, -v.y + v.x),     # Bag-venstre
            self._normalize_diagonal(-v.x + v.y, -v.y - v.x),     # Bag-højre
        ]
        
        body_set = set((seg.x, seg.y) for seg in snake.body)
        max_dist = float(max(self.grid.x, self.grid.y))
        
        state = []
        for d in directions:
            wall_dist = self._distance_to_wall(head, d)
            body_dist = self._distance_to_body(head, d, body_set)
            food_visible = self._food_in_direction(head, d, food.p)
            
            state.append(wall_dist / max_dist)
            state.append(body_dist / max_dist if body_dist < float('inf') else 1.0)
            state.append(1.0 if food_visible else 0.0)
        
        return state
    
    def _normalize_diagonal(self, x: int, y: int) -> Vector:
        norm_x = 0 if x == 0 else (1 if x > 0 else -1)
        norm_y = 0 if y == 0 else (1 if y > 0 else -1)
        return Vector(norm_x, norm_y)
    
    def _distance_to_wall(self, start: Vector, direction: Vector) -> int:
        pos = start
        distance = 0
        while True:
            pos = pos + direction
            distance += 1
            if not pos.within(self.grid):
                return distance
            if distance > self.grid.x + self.grid.y:
                return distance
    
    def _distance_to_body(self, start: Vector, direction: Vector, body_set: Set) -> float:
        pos = start
        distance = 0
        while True:
            pos = pos + direction
            distance += 1
            if not pos.within(self.grid):
                return float('inf')
            if (pos.x, pos.y) in body_set:
                return distance
            if distance > self.grid.x + self.grid.y:
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

    # ========== GAME RUNNING METHODS ==========

    def run_headless(self, ai_model: NeuralNetwork, max_moves: int = 200, 
                     fitness_config: dict = None, input_mode: int = 11) -> float:
        result = self.run_headless_detailed(ai_model, max_moves, fitness_config, input_mode)
        return result.fitness

    def run_headless_detailed(self, ai_model: NeuralNetwork, max_moves: int = 200, 
                               fitness_config: dict = None, input_mode: int = 11) -> GameResult:
        cfg = fitness_config or {
            "score_multiplier": 5000,
            "score_exponent": 2,
            "move_bonus": 2,
            "proximity_multiplier": 50,
            "loop_death_factor": 30,
            "moves_per_food": 50,
        }
        
        snake = Snake(game=self)
        food = Food(game=self, snake=snake)
        snake.v = Vector(1, 0)

        total_moves = 0
        min_distance_to_food = float('inf')
        death_reason = "max_moves"
        
        # Dynamisk max_moves parameter
        moves_per_food = cfg.get("moves_per_food", 50)
        
        while True:
            # Beregn dynamisk max moves: base + bonus per mad spist
            dynamic_max_moves = max_moves + (snake.score * moves_per_food)
            
            if total_moves >= dynamic_max_moves:
                death_reason = "max_moves"
                break
            
            distance = snake.p.manhattan_distance(food.p)
            min_distance_to_food = min(min_distance_to_food, distance)
            
            state = self.get_game_state(snake, food, input_mode)
            action = ai_model.predict(state)
            snake.v = self.translate_action_to_vector(action, snake.v)

            snake.move()
            total_moves += 1
            snake.moves_since_last_food += 1
            
            if not snake.p.within(self.grid):
                death_reason = "wall"
                break
            
            if snake.cross_own_tail:
                death_reason = "tail"
                break
            
            if snake.p == food.p:
                snake.add_score()
                food = Food(game=self, snake=snake)
                snake.moves_since_last_food = 0
                min_distance_to_food = float('inf')
                
            if snake.moves_since_last_food > (len(snake.body) * cfg["loop_death_factor"]): 
                death_reason = "loop"
                break 

        # Fitness beregning
        if snake.score > 0:
            fitness = (snake.score ** cfg["score_exponent"]) * cfg["score_multiplier"]
            fitness += total_moves * cfg["move_bonus"]
        else:
            max_distance = self.grid.x + self.grid.y
            proximity_bonus = (max_distance - min_distance_to_food) * cfg["proximity_multiplier"]
            fitness = total_moves + proximity_bonus

        return GameResult(
            score=snake.score,
            moves=total_moves,
            fitness=fitness,
            death_reason=death_reason
        )
    
    def run(self):
        """Kør spil med tastatur-kontrol."""
        if not self.visual:
            print("Fejl: Spillet er ikke i visuel tilstand.")
            return
             
        running = True
        snake = Snake(game=self)
        food = Food(game=self, snake=snake)
        snake.v = Vector(1, 0)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        snake.v = Vector(-1, 0)
                    if event.key == pygame.K_RIGHT:
                        snake.v = Vector(1, 0)
                    if event.key == pygame.K_UP:
                        snake.v = Vector(0, -1)
                    if event.key == pygame.K_DOWN:
                        snake.v = Vector(0, 1)

            snake.move()
            if not snake.p.within(self.grid) or snake.cross_own_tail:
                running = False
            if snake.p == food.p:
                snake.add_score()
                food = Food(game=self, snake=snake)

            self.screen.fill('black')
            for i, p in enumerate(snake.body):
                pygame.draw.rect(self.screen, (0, max(128, 255 - i * 8), 0), self.block(p))
            pygame.draw.rect(self.screen, (255, 0, 0), self.block(food.p))
            pygame.display.flip()
            self.clock.tick(10)

        print(f'Final Score: {snake.score}')


class Food:
    def __init__(self, game: SnakeGame, snake: 'Snake' = None):
        self.game = game
        self.p = self._spawn_food(snake)
    
    def _spawn_food(self, snake: 'Snake' = None) -> Vector:
        while True:
            pos = Vector.random_within(self.game.grid)
            if snake is None or pos not in snake.body:
                return pos


class Snake:
    def __init__(self, *, game: 'SnakeGame'):
        self.game = game
        self.score = 0
        self.v = Vector(0, 0)
        self.body = deque()
        center = Vector(self.game.grid.x // 2, self.game.grid.y // 2)
        self.body.append(center)
        self.moves_since_last_food = 0 
        self._grow = False

    def move(self):
        self.p = self.p + self.v

    @property
    def cross_own_tail(self):
        return self.p in list(self.body)[1:]

    @property
    def p(self):
        return self.body[0]

    @p.setter
    def p(self, value):
        self.body.appendleft(value)
        if not self._grow:
            self.body.pop()
        else:
            self._grow = False 

    def add_score(self):
        self.score += 1
        self._grow = True 


if __name__ == '__main__':
    print("Filen er sat op til AI-træning. Kør fra Jupyter Notebook.")