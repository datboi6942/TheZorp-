import uuid
import numpy as np
from typing import Tuple, Dict, List, Any, TYPE_CHECKING, Optional

from .brain import ZorpBrain, ActionType, ActionChoice # Assuming brain.py is in the same directory
# from ..config import GameConfig # Would be ideal for default sizes

if TYPE_CHECKING:
    from ..world.world import ZorpWorld # Forward reference for type hinting

# Default brain dimensions - these could come from GameConfig
DEFAULT_INPUT_SIZE = 8  # 1 entropy + 3 resources + 4 signals
DEFAULT_HIDDEN_SIZE = 16 # Configurable width
DEFAULT_OUTPUT_SIZE = 11 # 5 action_type_logits + 2 move_params + 4 signal_params

HUNGER_THRESHOLD_PERCENT = 0.5 # Zorp seeks food if energy is below 50% of max (assuming max is initial energy)
MAX_ENERGY = 100.0 # Assuming this is the default starting energy, can be configured

class Zorp:
    """
    Represents an agent in the ZorpLife simulation.
    """
    def __init__(
        self,
        position: Tuple[int, int],
        energy: float = 100.0,
        genome: Optional[Dict[str, List[np.ndarray]]] = None,
        age: int = 0,
        lineage_id: Optional[str] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        output_size: int = DEFAULT_OUTPUT_SIZE,
    ):
        """
        Initializes a Zorp instance.

        Args:
            position: The (x, y) coordinates of the Zorp.
            energy: The current energy level of the Zorp.
            genome: The genetic makeup of the Zorp, defining its brain's weights and biases.
                    If None, a random genome is generated.
            age: The age of the Zorp in ticks.
            lineage_id: An identifier for tracking family trees.
            input_size: The number of inputs for the Zorp's brain.
            hidden_size: The number of hidden neurons in the Zorp's brain.
            output_size: The number of outputs from the Zorp's brain.
        """
        self.id: str = str(uuid.uuid4())
        self.position: Tuple[int, int] = position
        self.energy: float = energy
        self.age: int = age
        self.alive: bool = True
        self.lineage_id: str = lineage_id if lineage_id else self.id
        self.max_energy: float = MAX_ENERGY # Store max energy for threshold calculation

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if genome is None:
            self.genome = self._generate_random_genome()
        else:
            self.genome = genome

        self.brain: ZorpBrain = ZorpBrain(
            genome=self.genome,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
        )
        self.current_signal: np.ndarray = np.zeros(4) # float[4] for emitted signal

    def _generate_random_genome(self) -> Dict[str, List[np.ndarray]]:
        """Generates a random genome (weights and biases) for the ZorpBrain."""
        W1 = np.random.randn(self.input_size, self.hidden_size).astype(np.float32) * 0.1
        B1 = np.zeros((self.hidden_size,), dtype=np.float32)
        W2 = np.random.randn(self.hidden_size, self.output_size).astype(np.float32) * 0.1
        B2 = np.zeros((self.output_size,), dtype=np.float32)
        return {"weights": [W1, W2], "biases": [B1, B2]}

    def perceive(self, world: 'ZorpWorld') -> np.ndarray:
        """
        Gathers sensory information from the environment.
        This is a placeholder implementation.

        Args:
            world: The ZorpWorld instance providing environmental data.

        Returns:
            A NumPy array representing the Zorp's perception vector.
            Expected elements: [local_entropy, resource1, resource2, resource3, sig1, sig2, sig3, sig4]
        """
        perception_vector = np.zeros(self.input_size, dtype=np.float32)
        
        # Placeholder for local_entropy
        perception_vector[0] = 0.0 # world.get_local_entropy(self.position)
        
        # Placeholder for resource_presence (e.g., plant, mineral, water)
        # For now, 3 elements for resources. Actual implementation in Stage 5.
        perception_vector[1:4] = world.get_resources_at(self.position) # This returns np.zeros(3) currently
        
        # Wall sensors (N, E, S, W)
        # This will use perception_vector[4], [5], [6], [7]
        wall_sensors = world.get_wall_sensor_vector(self.position)
        perception_vector[4:8] = wall_sensors
        
        return perception_vector

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax function."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def think(self, world: 'ZorpWorld') -> ActionChoice:
        """
        Uses the ZorpBrain to decide on an action based on perception.

        Args:
            world: The ZorpWorld instance to perceive from.

        Returns:
            An ActionChoice object representing the Zorp's decided action.
        """
        perception_vector = self.perceive(world)
        raw_nn_output = self.brain.forward(perception_vector)

        action_logits = raw_nn_output[0:5]
        action_probabilities = self._softmax(action_logits)
        
        # Choose action stochastically based on probabilities
        action_type_idx = np.random.choice(len(ActionType), p=action_probabilities)
        chosen_action_type = list(ActionType)[action_type_idx]

        move_delta: Optional[Tuple[int, int]] = None
        signal_vector: Optional[np.ndarray] = None

        if chosen_action_type == ActionType.MOVE:
            # move_params are raw_nn_output[5:7], already tanh'd in brain's forward pass
            move_params = raw_nn_output[5:7]
            # Discretize to -1, 0, or 1
            dx = int(np.round(move_params[0]))
            dy = int(np.round(move_params[1]))
            # Ensure they are within bounds if necessary, e.g. sum of abs values <= 1 for pure cardinal/diagonal
            # For now, direct rounding is fine. Can refine to ensure valid single-step moves.
            if dx == 0 and dy == 0:
                 # If move results in no change, force a random 1-step move.
                 # This ensures that if the NN decides to move, it actually moves.
                 print(f"[Zorp {self.id} think] NN chose MOVE but delta was (0,0). Forcing random 1-step move.")
                 possible_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                 chosen_move_idx = np.random.choice(len(possible_moves))
                 dx, dy = possible_moves[chosen_move_idx]
            move_delta = (dx, dy)
        
        elif chosen_action_type == ActionType.EMIT_SIGNAL:
            # signal_params are raw_nn_output[7:11], already tanh'd in brain's forward pass
            signal_params = raw_nn_output[7:11]
            signal_vector = signal_params # These are floats in [-1, 1]
            self.current_signal = signal_vector # Update Zorp's own last emitted signal

        result_action_choice = ActionChoice(
            action_type=chosen_action_type,
            move_delta=move_delta,
            signal_vector=signal_vector
        )
        print(f"[Zorp {self.id} think] NN chose action: {chosen_action_type}, Details: {result_action_choice}")
        return result_action_choice

    def decide_action(self, world: 'ZorpWorld') -> ActionChoice:
        """
        Decides the Zorp's next action based on its state (e.g., hunger) or brain.
        This is the main decision-making entry point called by ZorpWorld.tick().
        """
        hunger_trigger_value = self.max_energy * HUNGER_THRESHOLD_PERCENT
        print(f"[Zorp {self.id} decide_action] Energy: {self.energy:.1f}/{self.max_energy:.1f}, Hunger Threshold: < {hunger_trigger_value:.1f}")

        # Hybrid Behavior:
        # 1. Hardcoded: Seek food if hungry
        if self.energy < hunger_trigger_value:
            print(f"[Zorp {self.id} decide_action] Hungry. Calling seek_food().")
            action = self.seek_food(world)
        # 2. Emergent: Otherwise, use the neural network
        else:
            print(f"[Zorp {self.id} decide_action] Not hungry. Calling think().")
            action = self.think(world)
        
        print(f"[Zorp {self.id} decide_action] Chosen action: {action.action_type}, Details: {action}")
        return action

    def passive_update(self, world: 'ZorpWorld') -> None:
        """
        Handles passive updates to the Zorp's state each tick (e.g., aging, base metabolism).
        Called by ZorpWorld.tick() after an action has been chosen and applied.

        Args:
            world: The ZorpWorld instance.
        """
        if not self.alive:
            return

        # Basic energy consumption for existing (cost of living)
        self.energy -= 0.5 # Reduced base cost, as actions now have costs via ZorpWorld._apply_action
        if self.energy <= 0:
            self.alive = False
            # print(f"Zorp {self.id} starved during passive_update.")
            return
        
        self.age += 1

    def seek_food(self, world: 'ZorpWorld') -> ActionChoice:
        """
        Hardcoded behavior to find and move towards the nearest food source, or eat if already on food.

        Args:
            world: The ZorpWorld instance.

        Returns:
            An ActionChoice (EAT, MOVE, or REST).
        """
        print(f"[Zorp {self.id} seek_food] Called. Position: {self.position}, Energy: {self.energy:.1f}")
        # 1. Check current tile for food
        current_resources = world.get_resources_at(self.position)
        print(f"[Zorp {self.id} seek_food] Resources at current pos {self.position}: {current_resources}")
        if current_resources[0] > 0: # Index 0 is food_metric (energy value)
            print(f"[Zorp {self.id} seek_food] Food on current tile. Choosing EAT.")
            return ActionChoice(action_type=ActionType.EAT)

        # 2. Scan immediate neighbors for food if not eating on current tile
        print(f"[Zorp {self.id} seek_food] No food on current tile. Scanning neighbors.")
        for dx_offset in [-1, 0, 1]:
            for dy_offset in [-1, 0, 1]:
                if dx_offset == 0 and dy_offset == 0:
                    continue # Skip current tile
                
                check_pos = (self.position[0] + dx_offset, self.position[1] + dy_offset)
                
                if not world.is_within_bounds(check_pos):
                    # print(f"[Zorp {self.id} seek_food] Neighbor {check_pos} is out of bounds.")
                    continue

                resources_at_check_pos = world.get_resources_at(check_pos)
                passable = world.is_tile_passable_for_zorp(check_pos)
                print(f"[Zorp {self.id} seek_food] Checking neighbor {check_pos}: Resources={resources_at_check_pos}, Passable={passable}")

                if resources_at_check_pos[0] > 0: # Food found
                    if passable:
                        print(f"[Zorp {self.id} seek_food] Food found at {check_pos}. Choosing MOVE.")
                        return ActionChoice(action_type=ActionType.MOVE, move_delta=(dx_offset, dy_offset))
                    else:
                        print(f"[Zorp {self.id} seek_food] Food at {check_pos} but not passable.")
        
        # 3. If no food on current tile or adjacent passable tiles, perform a small random walk or REST
        print(f"[Zorp {self.id} seek_food] No immediate food found. Resorting to random move/rest.")
        rand_dx = np.random.randint(-1, 2)
        rand_dy = np.random.randint(-1, 2)
        
        if rand_dx == 0 and rand_dy == 0: # If random choice is to stay, REST
            # print(f"Zorp {self.id} chose random REST while seeking food.")
            return ActionChoice(action_type=ActionType.REST)

        next_pos_candidate = (self.position[0] + rand_dx, self.position[1] + rand_dy)
        if world.is_tile_passable_for_zorp(next_pos_candidate):
            # print(f"Zorp {self.id} random move to {next_pos_candidate} while seeking food.")
            return ActionChoice(action_type=ActionType.MOVE, move_delta=(rand_dx, rand_dy))
        else: # If random move is blocked, just rest
            # print(f"Zorp {self.id} random move blocked, chose REST while seeking food.")
            return ActionChoice(action_type=ActionType.REST)

    def __repr__(self) -> str:
        return f"Zorp(id={self.id}, pos={self.position}, energy={self.energy:.1f}, age={self.age}, alive={self.alive})"

# Example usage (for testing, not part of the class itself)
if __name__ == '__main__':
    # This part would require a mock ZorpWorld or be part of a larger test setup
    class MockZorpWorld:
        pass # Add methods if perceive() needs them for a standalone test

    world_instance = MockZorpWorld()

    # Create a Zorp with a default random genome
    zorp = Zorp(position=(5, 5))
    print(f"Created Zorp: {zorp}")
    print(f"Zorp genome: {zorp.genome['weights'][0].shape}, {zorp.genome['biases'][0].shape}, ...")

    # Simulate perception and thinking
    try:
        action_choice = zorp.think(world_instance)
        print(f"Zorp chose action: {action_choice}")
        if action_choice.action_type == ActionType.MOVE:
            print(f"Move details: {action_choice.move_delta}")
        elif action_choice.action_type == ActionType.EMIT_SIGNAL:
            print(f"Signal details: {action_choice.signal_vector}")
    except Exception as e:
        print(f"Error during think: {e}")

    zorp.update(world_instance)
    print(f"Zorp after update: {zorp}") 