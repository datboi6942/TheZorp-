from typing import List, Tuple, Dict, Set, Optional
import numpy as np

from ..core.zorp import Zorp, DEFAULT_INPUT_SIZE, DEFAULT_OUTPUT_SIZE
from ..core.brain import ActionType, ActionChoice
from ..config import GameConfig # To get map dimensions, etc.
from zorplife.world.tiles import ResourceType

# Basic energy costs for actions (can be moved to GameConfig)
ENERGY_COST_MOVE = 1.0
ENERGY_COST_EAT_ATTEMPT = 0.5
ENERGY_COST_REPRODUCE_ATTEMPT = 5.0 # Higher cost even if not successful yet
ENERGY_COST_EMIT_SIGNAL = 0.2
ENERGY_COST_REST = 0.0 # Resting might even regain a tiny bit or have a very low cost
ENERGY_COST_THINK = 0.1 # Base cost for thinking per tick

class ZorpWorld:
    """
    Manages the simulation world, including the grid, Zorps, and environmental factors.

    Args:
        config: The game configuration object.
        map_data: The tile grid (np.ndarray of Tile) for the world.
    """
    def __init__(self, config: GameConfig, map_data: np.ndarray) -> None:
        """
        Initializes the ZorpWorld.

        Args:
            config: The game configuration object.
            map_data: The tile grid (np.ndarray of Tile) for the world.
        """
        self.width: int = config.map_width
        self.height: int = config.map_height
        self.config: GameConfig = config
        self.map_data: np.ndarray = map_data

        self.all_zorps: List[Zorp] = [] # Master list of all Zorps
        # For faster spatial lookups, maps (x, y) to a list of Zorp IDs or Zorp objects
        # This helps in finding neighbors or objects at a specific location.
        self.zorp_spatial_map: Dict[Tuple[int, int], List[Zorp]] = {}

        # Placeholder for resource maps (to be developed in Stage 5)
        # self.entropy_map = np.zeros((self.width, self.height), dtype=float)
        # self.resource_maps = {
        # "plant_matter": np.zeros((self.width, self.height), dtype=float),
        # "minerals": np.zeros((self.width, self.height), dtype=float),
        # "water": np.zeros((self.width, self.height), dtype=float),
        # }

    def add_zorp(self, zorp: Zorp) -> None:
        """
        Adds a Zorp to the world.

        Args:
            zorp: The Zorp instance to add.
        """
        if not self.is_within_bounds(zorp.position):
            raise ValueError(f"Zorp position {zorp.position} is out of bounds.")
        
        self.all_zorps.append(zorp)
        # Update spatial map
        pos_tuple = tuple(zorp.position)
        if pos_tuple not in self.zorp_spatial_map:
            self.zorp_spatial_map[pos_tuple] = []
        self.zorp_spatial_map[pos_tuple].append(zorp)

    def remove_zorp(self, zorp: Zorp) -> None:
        """
        Removes a Zorp from the world (e.g., when it dies).

        Args:
            zorp: The Zorp instance to remove.
        """
        if zorp in self.all_zorps:
            self.all_zorps.remove(zorp)
        
        pos_tuple = tuple(zorp.position)
        if pos_tuple in self.zorp_spatial_map and zorp in self.zorp_spatial_map[pos_tuple]:
            self.zorp_spatial_map[pos_tuple].remove(zorp)
            if not self.zorp_spatial_map[pos_tuple]: # Clean up empty list
                del self.zorp_spatial_map[pos_tuple]

    def _update_zorp_spatial_map(self, zorp: Zorp, old_position: Tuple[int, int]) -> None:
        """Updates the Zorp's position in the spatial map."""
        old_pos_tuple = tuple(old_position)
        if old_pos_tuple in self.zorp_spatial_map and zorp in self.zorp_spatial_map[old_pos_tuple]:
            self.zorp_spatial_map[old_pos_tuple].remove(zorp)
            if not self.zorp_spatial_map[old_pos_tuple]:
                del self.zorp_spatial_map[old_pos_tuple]
        
        new_pos_tuple = tuple(zorp.position)
        if new_pos_tuple not in self.zorp_spatial_map:
            self.zorp_spatial_map[new_pos_tuple] = []
        self.zorp_spatial_map[new_pos_tuple].append(zorp)

    def is_within_bounds(self, position: Tuple[int, int]) -> bool:
        """Checks if a given position is within the world boundaries."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def tick(self) -> None:
        """Advances the world state by one simulation step."""
        zorps_to_remove: List[Zorp] = []
        for zorp in self.all_zorps:
            if not zorp.alive:
                zorps_to_remove.append(zorp)
                continue

            # Zorp perceives, thinks, and decides on an action
            action_choice = zorp.think(self) # 'self' is the ZorpWorld instance
            print(f"DEBUG ZORPWORLD TICK: Zorp {zorp.id} E_before_action: {zorp.energy:.1f} Action: {action_choice.action_type}")
            
            # Apply the chosen action
            self._apply_action(zorp, action_choice)
            print(f"DEBUG ZORPWORLD TICK: Zorp {zorp.id} E_after_action: {zorp.energy:.1f} Alive_after_action: {zorp.alive}")
            
            # Zorp's internal state update (aging, passive energy loss)
            zorp.update(self) # This already deducts 1 energy for living
            print(f"DEBUG ZORPWORLD TICK: Zorp {zorp.id} E_after_update: {zorp.energy:.1f} Alive_after_update: {zorp.alive}")

            if not zorp.alive: # Check again if action or update killed it
                zorps_to_remove.append(zorp)

        for zorp in zorps_to_remove:
            self.remove_zorp(zorp)
            # print(f"Zorp {zorp.id} removed from world.")

    def _apply_action(self, zorp: Zorp, action: ActionChoice) -> None:
        """Applies the Zorp's chosen action to the world and the Zorp itself."""
        zorp.energy -= ENERGY_COST_THINK # Cost for thinking

        if action.action_type == ActionType.MOVE:
            if action.move_delta:
                old_position = zorp.position
                dx, dy = action.move_delta
                new_position = (old_position[0] + dx, old_position[1] + dy)

                if self.is_within_bounds(new_position):
                    zorp.position = new_position
                    self._update_zorp_spatial_map(zorp, old_position)
                    zorp.energy -= ENERGY_COST_MOVE
                else:
                    # Bumped into wall, penalize slightly or do nothing
                    zorp.energy -= ENERGY_COST_MOVE * 0.5 # Lesser penalty for failed move
            else:
                # This case should ideally not happen if MOVE implies a move_delta
                # If it does, treat as a REST or a very short action
                zorp.energy -= ENERGY_COST_REST 

        elif action.action_type == ActionType.EAT:
            # Cost to attempt eating, regardless of success
            zorp.energy -= ENERGY_COST_EAT_ATTEMPT
            # print(f"DEBUG: Zorp {zorp.id} attempts EAT. E_cost: {Zorp.ENERGY_COST_EAT_ATTEMPT:.1f}, E_rem: {zorp.energy:.1f}")
            
            # Check tile for food
            tile_x, tile_y = zorp.position
            current_tile_enum = self.map_data[tile_y, tile_x]
            tile_meta = current_tile_enum.metadata

            # Check if the tile's resource type is something the Zorp can eat (e.g., PLANT_MATTER, APPLES, MUSHROOMS)
            # and if the tile has a defined energy_value > 0
            if tile_meta.energy_value > 0 and tile_meta.resource_type in [ResourceType.PLANT_MATTER, ResourceType.APPLES, ResourceType.MUSHROOMS, ResourceType.ORGANIC_MATTER]:
                energy_gained = tile_meta.energy_value
                zorp.energy += energy_gained
                # print(f"DEBUG: Zorp {zorp.id} ATE {current_tile_enum.name}, gained {energy_gained:.1f} E. Total E: {zorp.energy:.1f}")
                
                # Optional: Tile depletion/change logic can go here
                # For example, change GRASSLAND to DIRT for a while, or reduce a resource counter on the tile.
                # For now, food is infinitely available on the tile.
            # else:
                # print(f"DEBUG: Zorp {zorp.id} tried to EAT {current_tile_enum.name} but it provides no/unsuitable energy.")

        elif action.action_type == ActionType.REPRODUCE:
            # Placeholder: Zorp attempts to reproduce. Real logic in Stage 2.
            zorp.energy -= ENERGY_COST_REPRODUCE_ATTEMPT
            # TODO: Implement reproduction logic

        elif action.action_type == ActionType.EMIT_SIGNAL:
            # Placeholder: Zorp emits a signal. Real logic in Stage 3.
            zorp.energy -= ENERGY_COST_EMIT_SIGNAL
            if action.signal_vector is not None:
                zorp.current_signal = action.signal_vector
            # TODO: Make signal available to other Zorps via perception

        elif action.action_type == ActionType.REST:
            zorp.energy -= ENERGY_COST_REST
            # Potentially small energy gain or recovery from fatigue in future

        if zorp.energy <= 0:
            zorp.alive = False

    # --- Perception Methods (Placeholders for Zorp.perceive) ---
    # These methods will be called by individual Zorps during their perceive phase.
    # For Stage 1, they return default values.

    def get_local_entropy(self, position: Tuple[int, int]) -> float:
        """Placeholder for getting local entropy at a position."""
        # TODO: Implement in Stage 5
        return 0.0 # Example: normalized entropy value

    def get_resources_at(self, position: Tuple[int, int]) -> np.ndarray:
        """
        Placeholder for getting resource levels at a position.
        Returns a 3-element array for [plant, mineral, water].
        """
        # TODO: Implement in Stage 5
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_average_signal_near(self, position: Tuple[int, int], radius: int = 5) -> np.ndarray:
        """
        Placeholder for getting average signal from nearby Zorps.
        Returns a 4-element array for the signal vector.
        """
        # TODO: Implement in Stage 3
        # For now, returns no signal.
        # This would involve iterating self.zorp_spatial_map or self.all_zorps
        # to find Zorps within `radius` of `position` and averaging their `current_signal`.
        return np.zeros(4, dtype=np.float32) 

    def get_wall_sensor_vector(self, position: Tuple[int, int]) -> np.ndarray:
        """
        Checks for world boundaries in adjacent cells (N, E, S, W).

        Args:
            position: The (x, y) coordinates of the Zorp.

        Returns:
            A NumPy array of shape (4,) with sensor_values (0.0 or 1.0):
            - sensor_values[0] (North): 1.0 if moving North (y+1) is out of bounds, else 0.0.
            - sensor_values[1] (East):  1.0 if moving East (x+1) is out of bounds, else 0.0.
            - sensor_values[2] (South): 1.0 if moving South (y-1) is out of bounds, else 0.0.
            - sensor_values[3] (West):  1.0 if moving West (x-1) is out of bounds, else 0.0.
        """
        x, y = position
        sensor_values = np.zeros(4, dtype=np.float32)

        # Check North (y+1)
        if not self.is_within_bounds((x, y + 1)):
            sensor_values[0] = 1.0
        # Check East (x+1)
        if not self.is_within_bounds((x + 1, y)):
            sensor_values[1] = 1.0
        # Check South (y-1)
        if not self.is_within_bounds((x, y - 1)):
            sensor_values[2] = 1.0
        # Check West (x-1)
        if not self.is_within_bounds((x - 1, y)):
            sensor_values[3] = 1.0
            
        return sensor_values

    def get_zorps_at(self, position: Tuple[int,int]) -> List[Zorp]:
        """Returns a list of Zorps at a given position."""
        return self.zorp_spatial_map.get(tuple(position), []) 