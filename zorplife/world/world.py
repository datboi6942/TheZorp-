from typing import List, Tuple, Dict, Set, Optional
import numpy as np

from ..core.zorp import Zorp, DEFAULT_INPUT_SIZE, DEFAULT_OUTPUT_SIZE
from ..core.brain import ActionType, ActionChoice
from ..config import GameConfig # To get map dimensions, etc.
from zorplife.world.tiles import ResourceType, Tile

# Basic energy costs for actions (can be moved to GameConfig)
ENERGY_COST_MOVE = 1.0
ENERGY_COST_EAT_ATTEMPT = 0.5
ENERGY_COST_REPRODUCE_ATTEMPT = 2.0 # Cost for a failed attempt or if not eligible
ENERGY_COST_EMIT_SIGNAL = 0.2
ENERGY_COST_REST = 0.0 # Resting might even regain a tiny bit or have a very low cost
ENERGY_COST_THINK = 0.1 # Base cost for thinking per tick

# Reproduction related constants
REPRODUCTION_MIN_ENERGY_THRESHOLD_FACTOR = 0.8 # Zorp needs 80% of its max_energy to be eligible
REPRODUCTION_ACTUAL_COST_FACTOR = 0.5          # Parent loses 50% of its max_energy upon successful reproduction
REPRODUCTION_CHILD_INITIAL_ENERGY_FACTOR = 0.25  # Child starts with 25% of parent's max_energy
REGROW_TICKS = 50

EDIBLE_RESOURCE_TYPES = [ResourceType.PLANT_MATTER, ResourceType.APPLES, ResourceType.MUSHROOMS, ResourceType.ORGANIC_MATTER]

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

        # Per-tile resource energy map
        self.resource_energy_map: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        for y_idx in range(self.height): # Renamed y to y_idx to avoid clash
            for x_idx in range(self.width): # Renamed x to x_idx
                self.resource_energy_map[y_idx, x_idx] = self.map_data[y_idx, x_idx].metadata.energy_value

        self.resource_regrow_timer: np.ndarray = np.zeros((self.height, self.width), dtype=np.int32)
        
        # For tracking visually changed tiles
        self.consumed_tile_original_type: Dict[Tuple[int, int], Tile] = {}
        self.map_visuals_dirty: bool = False

        # Logging timer variables
        self.current_game_time: float = 0.0
        self.log_timer: float = 0.0
        self.log_interval: float = 2.0 # Log every 2 seconds

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

    def is_tile_passable_for_zorp(self, position: Tuple[int, int]) -> bool:
        """Checks if a tile at a given position is passable for a Zorp."""
        if not self.is_within_bounds(position):
            return False
        # Assuming self.map_data stores Tile enums
        tile_enum: Tile = self.map_data[position[1], position[0]]
        return tile_enum.metadata.passable

    def tick(self, dt: float) -> None:
        """Advances the world state by one simulation step."""
        self.current_game_time += dt
        allow_logging_this_tick = False
        if self.current_game_time >= self.log_timer + self.log_interval:
            allow_logging_this_tick = True
            self.log_timer = self.current_game_time
            if allow_logging_this_tick:
                 print(f"[ZorpWorld tick] --- Logging interval reached at game time {self.current_game_time:.2f}s ---")

        zorps_to_remove: List[Zorp] = []
        for zorp in self.all_zorps:
            if not zorp.alive:
                zorps_to_remove.append(zorp)
                continue

            # Zorp decides action (can be hunger-driven or NN-driven)
            action_choice = zorp.decide_action(self, allow_logging_this_tick) # Pass logging flag
            
            if allow_logging_this_tick:
                print(f"[ZorpWorld tick] Zorp {zorp.id} (E: {zorp.energy:.1f}) decided action: {action_choice.action_type}, Details: {action_choice}")
            
            # Apply the chosen action
            self._apply_action(zorp, action_choice, allow_logging_this_tick) # Pass logging flag
            
            # Zorp's internal passive state update (aging, base metabolism cost of living)
            zorp.passive_update(self) 

            if not zorp.alive: # Check again if action or update killed it
                zorps_to_remove.append(zorp)

        for zorp_to_remove in zorps_to_remove: # Renamed zorp to zorp_to_remove
            if allow_logging_this_tick: # Also gate this log
                print(f"[ZorpWorld tick] Zorp {zorp_to_remove.id} removed from world.")
            self.remove_zorp(zorp_to_remove)
            # print(f"Zorp {zorp.id} removed from world.") # Original unconditional print

        # Regrow resources
        for y_coord in range(self.height): # Renamed y to y_coord
            for x_coord in range(self.width): # Renamed x to x_coord
                if self.resource_regrow_timer[y_coord, x_coord] > 0:
                    self.resource_regrow_timer[y_coord, x_coord] -= 1
                    if self.resource_regrow_timer[y_coord, x_coord] == 0:
                        # Restore original energy value from tile metadata
                        original_tile_type_for_energy = self.map_data[y_coord, x_coord]
                        # If it was a consumed tile, its current type in map_data is the "empty" version.
                        # We need the *original* type's metadata for energy.
                        if (x_coord, y_coord) in self.consumed_tile_original_type:
                            original_tile_type_for_energy = self.consumed_tile_original_type[(x_coord, y_coord)]
                        
                        self.resource_energy_map[y_coord, x_coord] = original_tile_type_for_energy.metadata.energy_value
                        
                        if allow_logging_this_tick: # Gate this log too
                            log_msg = f"[ZorpWorld tick] Resource at ({x_coord},{y_coord}) energy regrew to {self.resource_energy_map[y_coord, x_coord]:.1f}."
                            if (x_coord, y_coord) in self.consumed_tile_original_type:
                                log_msg += f" Tile type changing back to {original_tile_type_for_energy.name}."
                            print(log_msg)

                        # Change tile type back if it was a consumed special resource
                        if (x_coord, y_coord) in self.consumed_tile_original_type:
                            self.map_data[y_coord, x_coord] = self.consumed_tile_original_type.pop((x_coord, y_coord))
                            self.map_visuals_dirty = True
                            print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: Regrowth at ({x_coord},{y_coord}) to {self.map_data[y_coord, x_coord].name}") # UNGATED DEBUG
                            if allow_logging_this_tick:
                                print(f"[ZorpWorld tick] Tile at ({x_coord},{y_coord}) visually restored to {self.map_data[y_coord, x_coord].name}.")

    def _apply_action(self, zorp: Zorp, action: ActionChoice, allow_log: bool) -> None:
        """Applies the Zorp's chosen action to the world and the Zorp itself."""
        zorp.energy -= ENERGY_COST_THINK # Cost for thinking

        if action.action_type == ActionType.MOVE:
            if action.move_delta:
                old_position = zorp.position
                dx, dy = action.move_delta
                new_position = (old_position[0] + dx, old_position[1] + dy)

                if self.is_within_bounds(new_position) and self.is_tile_passable_for_zorp(new_position): # Added passability check
                    # Check if another Zorp is in the target position
                    # For now, allow stacking. Could be changed later:
                    # if not self.get_zorps_at(new_position): 
                    zorp.position = new_position
                    self._update_zorp_spatial_map(zorp, old_position)
                    zorp.energy -= ENERGY_COST_MOVE
                    # else: # Target tile occupied
                        # zorp.energy -= ENERGY_COST_MOVE * 0.25 # Small penalty for trying to move into occupied tile
                else:
                    # Bumped into wall or impassable terrain
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
            current_tile_enum: Tile = self.map_data[tile_y, tile_x] # Added type hint
            tile_meta = current_tile_enum.metadata

            # DEBUGGING FOR APPLES/MUSHROOMS - Ungated log for EAT attempt
            if tile_meta.resource_type == ResourceType.APPLES or tile_meta.resource_type == ResourceType.MUSHROOMS:
                print(f"[DEBUG _apply_action EAT Attempt] Zorp: {zorp.id} at Pos: ({tile_x},{tile_y}), Tile: {current_tile_enum.name}, MetaEnergy: {tile_meta.energy_value}, MapEnergy: {self.resource_energy_map[tile_y, tile_x]}, EdibleListCheck: {tile_meta.resource_type in EDIBLE_RESOURCE_TYPES}, ZorpEnergy: {zorp.energy:.1f}")

            # Check if the tile's resource type is something the Zorp can eat
            # and if the tile has a defined energy_value > 0
            if tile_meta.energy_value > 0 and tile_meta.resource_type in EDIBLE_RESOURCE_TYPES and self.resource_energy_map[tile_y, tile_x] > 0:
                energy_gained = self.resource_energy_map[tile_y, tile_x]
                zorp.energy += energy_gained
                zorp.energy = min(zorp.energy, zorp.max_energy)
                
                # DEBUGGING FOR APPLES/MUSHROOMS - Ungated log for EAT success
                if current_tile_enum.metadata.resource_type == ResourceType.APPLES or current_tile_enum.metadata.resource_type == ResourceType.MUSHROOMS:
                    print(f"[DEBUG _apply_action EAT Success] Zorp: {zorp.id} ATE {current_tile_enum.name}. Gained: {energy_gained:.1f}. New Zorp E: {zorp.energy:.1f}. New MapEnergy for tile: 0.0")

                # Original conditional log for success (can be kept or commented if new debug is sufficient)
                if allow_log:
                    print(f"[ZorpWorld _apply_action EAT] Zorp {zorp.id} ATE {current_tile_enum.name}, gained {energy_gained:.1f} E. Total E: {zorp.energy:.1f}")
                
                self.resource_energy_map[tile_y, tile_x] = 0.0
                self.resource_regrow_timer[tile_y, tile_x] = REGROW_TICKS
                
                # Visual change for consumed special resources
                if current_tile_enum == Tile.MUSHROOM_PATCH:
                    self.consumed_tile_original_type[(tile_x, tile_y)] = Tile.MUSHROOM_PATCH
                    self.map_data[tile_y, tile_x] = Tile.FOREST_FLOOR # Change to base tile
                    self.map_visuals_dirty = True
                    print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: MUSHROOM_PATCH consumed at ({tile_x},{tile_y})") # UNGATED DEBUG
                    if allow_log: 
                         print(f"[ZorpWorld _apply_action EAT] MUSHROOM_PATCH at ({tile_x},{tile_y}) consumed, changed to FOREST_FLOOR.")
                elif current_tile_enum == Tile.APPLE_TREE_LEAVES:
                    self.consumed_tile_original_type[(tile_x, tile_y)] = Tile.APPLE_TREE_LEAVES
                    self.map_data[tile_y, tile_x] = Tile.TREE_CANOPY # Change to non-fruiting canopy
                    self.map_visuals_dirty = True
                    print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: APPLE_TREE_LEAVES consumed at ({tile_x},{tile_y})") # UNGATED DEBUG
                    if allow_log: 
                        print(f"[ZorpWorld _apply_action EAT] APPLE_TREE_LEAVES at ({tile_x},{tile_y}) consumed, changed to TREE_CANOPY.")
                
                # Original conditional log for depletion (can be kept or commented)
                if allow_log:
                    print(f"[ZorpWorld _apply_action EAT] Tile {current_tile_enum.name} at ({tile_x},{tile_y}) depleted. Regrowth in {REGROW_TICKS} ticks.")

            else:
                # DEBUGGING FOR APPLES/MUSHROOMS - Ungated log for EAT fail
                if current_tile_enum.metadata.resource_type == ResourceType.APPLES or current_tile_enum.metadata.resource_type == ResourceType.MUSHROOMS:
                     print(f"[DEBUG _apply_action EAT Fail] Zorp: {zorp.id} at ({tile_x},{tile_y}) on {current_tile_enum.name}. Conditions: MetaEnergyOK={tile_meta.energy_value > 0}, EdibleListOK={tile_meta.resource_type in EDIBLE_RESOURCE_TYPES}, MapEnergyOK={self.resource_energy_map[tile_y, tile_x] > 0}. ZorpEnergy: {zorp.energy:.1f}")
                
                # Original conditional log for failure
                if allow_log:
                    print(f"[ZorpWorld _apply_action EAT] Zorp {zorp.id} tried to EAT {current_tile_enum.name} but it provides no/unsuitable energy.")

        elif action.action_type == ActionType.REPRODUCE:
            parent_max_energy = zorp.max_energy
            min_energy_to_reproduce = parent_max_energy * REPRODUCTION_MIN_ENERGY_THRESHOLD_FACTOR
            actual_reproduction_cost = parent_max_energy * REPRODUCTION_ACTUAL_COST_FACTOR
            child_initial_energy = parent_max_energy * REPRODUCTION_CHILD_INITIAL_ENERGY_FACTOR

            if zorp.energy >= min_energy_to_reproduce:
                spawn_location_found = False
                best_spawn_pos: Optional[Tuple[int, int]] = None
                deltas = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
                np.random.shuffle(deltas)

                for dx, dy in deltas:
                    check_pos = (zorp.position[0] + dx, zorp.position[1] + dy)
                    if (self.is_within_bounds(check_pos) and
                        self.is_tile_passable_for_zorp(check_pos) and
                        not self.get_zorps_at(check_pos)):
                        best_spawn_pos = check_pos
                        spawn_location_found = True
                        break
                
                if spawn_location_found and best_spawn_pos is not None:
                    zorp.energy -= actual_reproduction_cost
                    new_zorp = Zorp(position=best_spawn_pos, energy=child_initial_energy)
                    self.add_zorp(new_zorp)
                    if allow_log:
                        print(f"[ZorpWorld _apply_action REPRODUCE] Zorp {zorp.id} REPRODUCED. New Zorp {new_zorp.id} at {new_zorp.position} with E:{new_zorp.energy:.1f}. Parent E after: {zorp.energy:.1f}")
                else:
                    zorp.energy -= ENERGY_COST_REPRODUCE_ATTEMPT 
                    if allow_log:
                        print(f"[ZorpWorld _apply_action REPRODUCE] Zorp {zorp.id} FAILED to reproduce (no valid spawn location). Cost: {ENERGY_COST_REPRODUCE_ATTEMPT:.1f}. Parent E after: {zorp.energy:.1f}")
            else:
                zorp.energy -= ENERGY_COST_REPRODUCE_ATTEMPT
                if allow_log:
                    print(f"[ZorpWorld _apply_action REPRODUCE] Zorp {zorp.id} FAILED to reproduce (not enough energy: {zorp.energy:.1f}/{min_energy_to_reproduce:.1f}). Cost: {ENERGY_COST_REPRODUCE_ATTEMPT:.1f}. Parent E after: {zorp.energy:.1f}")

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
        Gets resource levels at a position.
        Returns a 3-element array for [food_metric, water_metric, other_metric].
        - food_metric: energy value from edible tile, or 0.
        - water_metric: 1.0 if water tile, 0.0 otherwise.
        - other_metric: currently 0.0.
        """
        if not self.is_within_bounds(position):
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        tile_enum: Tile = self.map_data[position[1], position[0]]
        tile_meta = tile_enum.metadata

        # DEBUGGING FOR APPLES/MUSHROOMS - Ungated
        if tile_meta.resource_type == ResourceType.APPLES or tile_meta.resource_type == ResourceType.MUSHROOMS:
            print(f"[DEBUG get_resources_at] Pos: {position}, Tile: {tile_enum.name}, MetaEnergy: {tile_meta.energy_value}, MapEnergy: {self.resource_energy_map[position[1], position[0]]}, EdibleListCheck: {tile_meta.resource_type in EDIBLE_RESOURCE_TYPES}")

        food_metric = 0.0
        if tile_meta.resource_type in EDIBLE_RESOURCE_TYPES and self.resource_energy_map[position[1], position[0]] > 0:
            food_metric = self.resource_energy_map[position[1], position[0]]
        
        water_metric = 0.0
        if tile_meta.resource_type == ResourceType.WATER:
            water_metric = 1.0
            
        other_metric = 0.0 # Placeholder for other resources like minerals, etc.

        return np.array([food_metric, water_metric, other_metric], dtype=np.float32)

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