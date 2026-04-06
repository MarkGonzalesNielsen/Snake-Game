"""
Neural Network til Snake AI.
Konfigurerbar hidden_nodes via parameter.
"""

import numpy as np
from typing import List

# Defaults
DEFAULT_INPUT_NODES = 11
DEFAULT_HIDDEN_NODES = 8
DEFAULT_OUTPUT_NODES = 3

# Backward compatibility
INPUT_NODES = DEFAULT_INPUT_NODES
HIDDEN_NODES = DEFAULT_HIDDEN_NODES
OUTPUT_NODES = DEFAULT_OUTPUT_NODES


class NeuralNetwork:
    def __init__(
        self, 
        weights_array: np.ndarray | None = None,
        hidden_nodes: int = DEFAULT_HIDDEN_NODES,
        input_nodes: int = DEFAULT_INPUT_NODES,
        output_nodes: int = DEFAULT_OUTPUT_NODES,
        activation: str = "softmax"
    ):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.activation = activation
        
        self.total_weights = (input_nodes * hidden_nodes) + (hidden_nodes * output_nodes)
        
        if weights_array is None:
            weights_array = np.random.uniform(-1.0, 1.0, size=self.total_weights)
        
        w1_size = input_nodes * hidden_nodes
        self.W1 = weights_array[:w1_size].reshape((input_nodes, hidden_nodes))
        self.W2 = weights_array[w1_size:].reshape((hidden_nodes, output_nodes))
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def predict(self, input_data: List[float]) -> int:
        X = np.array(input_data).reshape(1, self.input_nodes)
        
        Z1 = np.dot(X, self.W1)
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2)
        
        if self.activation == "softmax":
            A2 = self.softmax(Z2)
        else:
            A2 = self.sigmoid(Z2)
        
        return int(np.argmax(A2))
    
    @property
    def weights_flat(self) -> np.ndarray:
        return np.concatenate((self.W1.flatten(), self.W2.flatten()))
    
    @classmethod
    def from_config(cls, config: dict, weights_array: np.ndarray | None = None):
        """Opret NN fra config dict."""
        return cls(
            weights_array=weights_array,
            hidden_nodes=config.get("hidden_nodes", DEFAULT_HIDDEN_NODES),
            input_nodes=config.get("input_nodes", DEFAULT_INPUT_NODES),
            output_nodes=config.get("output_nodes", DEFAULT_OUTPUT_NODES),
            activation=config.get("activation", "softmax"),
        )