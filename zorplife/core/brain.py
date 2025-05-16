import numpy as np
from enum import Enum, auto
from typing import NamedTuple, Optional, Tuple, List, Dict, Any
import uuid

class ActionType(Enum):
    """Enumeration of possible actions a Zorp can take."""
    MOVE = auto()
    EAT = auto()
    REPRODUCE = auto()
    EMIT_SIGNAL = auto()
    REST = auto()
    PICK_UP = auto()

class ActionChoice(NamedTuple):
    """
    Represents the chosen action and its parameters.

    Attributes:
        action_type: The type of action chosen.
        move_delta: Tuple (dx, dy) if action_type is MOVE.
        signal_vector: Numpy array of floats if action_type is EMIT_SIGNAL.
        partner_id: UUID of the partner Zorp if action_type is REPRODUCE (sexual reproduction).
    """
    action_type: ActionType
    move_delta: Optional[Tuple[int, int]] = None
    signal_vector: Optional[np.ndarray] = None
    partner_id: Optional[uuid.UUID] = None

class ZorpBrain:
    """
    A simple Multi-Layer Perceptron (MLP) brain for a Zorp.
    It has one hidden layer.
    """
    def __init__(
        self,
        genome: Dict[str, List[np.ndarray]],
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        """
        Initializes the ZorpBrain from a genome.

        Args:
            genome: A dictionary containing weights and biases.
                    Expected structure: {"weights": [W1, W2], "biases": [B1, B2]}
                    W1: input_size x hidden_size
                    B1: hidden_size
                    W2: hidden_size x output_size
                    B2: output_size
            input_size: Number of input neurons.
            hidden_size: Number of neurons in the hidden layer.
            output_size: Number of output neurons.

        Raises:
            ValueError: If genome structure or dimensions are incorrect.
        """
        if not isinstance(genome, dict) or "weights" not in genome or "biases" not in genome:
            raise ValueError("Genome must be a dict with 'weights' and 'biases'.")
        if len(genome["weights"]) != 2 or len(genome["biases"]) != 2:
            raise ValueError("Genome must contain 2 weight matrices and 2 bias vectors.")

        self.W1: np.ndarray = genome["weights"][0]
        self.B1: np.ndarray = genome["biases"][0]
        self.W2: np.ndarray = genome["weights"][1]
        self.B2: np.ndarray = genome["biases"][1]

        if self.W1.shape != (input_size, hidden_size):
            raise ValueError(
                f"W1 shape mismatch. Expected {(input_size, hidden_size)}, got {self.W1.shape}"
            )
        if self.B1.shape != (hidden_size,):
            raise ValueError(
                f"B1 shape mismatch. Expected {(hidden_size,)}, got {self.B1.shape}"
            )
        if self.W2.shape != (hidden_size, output_size):
            raise ValueError(
                f"W2 shape mismatch. Expected {(hidden_size, output_size)}, got {self.W2.shape}"
            )
        if self.B2.shape != (output_size,):
            raise ValueError(
                f"B2 shape mismatch. Expected {(output_size,)}, got {self.B2.shape}"
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)

    def forward(self, perception_vector: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the neural network.

        Args:
            perception_vector: A 1D NumPy array representing the Zorp's sensory input.
                               Shape must be (input_size,).

        Returns:
            A 1D NumPy array representing the raw output of the network.
            Shape will be (output_size,).
            The first 5 elements are action logits.
            The next 2 elements are move parameters (pre-tanh).
            The last 4 elements are signal parameters (pre-tanh).

        Raises:
            ValueError: If perception_vector shape is incorrect.
        """
        if perception_vector.shape != (self.input_size,):
            raise ValueError(
                f"Perception vector shape mismatch. Expected {(self.input_size,)}, got {perception_vector.shape}"
            )

        # Input to Hidden layer
        hidden_layer_input: np.ndarray = perception_vector @ self.W1 + self.B1
        hidden_layer_output: np.ndarray = self._tanh(hidden_layer_input)

        # Hidden to Output layer
        output_layer_input: np.ndarray = hidden_layer_output @ self.W2 + self.B2
        
        # Apply activations to specific parts of the output
        # Action logits (first 5) - no activation here, softmax applied by consumer
        # Move parameters (next 2) - tanh
        # Signal parameters (last 4) - tanh
        
        processed_output = np.copy(output_layer_input)
        if self.output_size >= 7: # 5 action logits + 2 move params
            processed_output[5:7] = self._tanh(output_layer_input[5:7])
        if self.output_size == 11: # 5 action logits + 2 move params + 4 signal params
             processed_output[7:11] = self._tanh(output_layer_input[7:11])
        elif self.output_size > 7 and self.output_size < 11 :
             # This case means something is wrong with OUTPUT_SIZE configuration
             # or the intended output structure.
             # For now, let's assume if it's not exactly 11, and greater than 7,
             # the remaining are signals, but this might need refinement
             # based on how output_size is configured with action sets.
             # Given Stage 1 details, output_size should be 11.
             pass


        return processed_output 