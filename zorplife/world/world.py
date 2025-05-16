from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import uuid

from ..core.zorp import Zorp, DEFAULT_INPUT_SIZE, DEFAULT_OUTPUT_SIZE
from ..core.brain import ActionType, ActionChoice
from ..config import GameConfig # To get map dimensions, etc.
from zorplife.world.tiles import ResourceType, Tile
from zorplife.world.mapgen import MapGenerator

# Basic energy costs for actions (can be moved to GameConfig)
ENERGY_COST_MOVE = 0.1
ENERGY_COST_EAT_ATTEMPT = 0.5
ENERGY_COST_REPRODUCE_ATTEMPT = 2.0 # Cost for a failed attempt or if not eligible
ENERGY_COST_EMIT_SIGNAL = 0.2
ENERGY_COST_REST = 0.0 # Resting might even regain a tiny bit or have a very low cost
ENERGY_COST_THINK = 0.05 # Base cost for thinking per tick

# Reproduction related constants
REPRODUCTION_MIN_ENERGY_THRESHOLD_FACTOR = 0.3  # Zorp needs 30% of its max_energy to be eligible
REPRODUCTION_ACTUAL_COST_FACTOR = 0.1           # Parent loses 10% of its max_energy upon successful reproduction
REPRODUCTION_CHILD_INITIAL_ENERGY_FACTOR = 0.7  # Child starts with 70% of parent's max_energy
REGROW_TICKS = 100  # Much slower regrowth

EDIBLE_RESOURCE_TYPES = [ResourceType.PLANT_MATTER, ResourceType.APPLES, ResourceType.MUSHROOMS, ResourceType.ORGANIC_MATTER]

class ZorpWorld:
    """
    Manages the simulation world, including the grid, Zorps, and environmental factors.

    Args:
        config: The game configuration object.
        map_data: The tile grid (np.ndarray of Tile) for the world.
        map_generator: The MapGenerator instance used to generate the map (for biome/resource counts).
    """
    def __init__(self, config: GameConfig, map_data: np.ndarray, map_generator: Optional[MapGenerator] = None) -> None:
        """
        Initializes the ZorpWorld.

        Args:
            config: The game configuration object.
            map_data: The tile grid (np.ndarray of Tile) for the world.
            map_generator: The MapGenerator instance used to generate the map (for biome/resource counts).
        """
        self.width: int = config.map_width
        self.height: int = config.map_height
        self.config: GameConfig = config
        self.map_data: np.ndarray = map_data
        self.map_generator: Optional[MapGenerator] = map_generator

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

        # Track last reproduction time for each Zorp
        self.last_reproduction_time: Dict[uuid.UUID, float] = {}

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
                print(f"[ZorpWorld tick] Zorp {zorp.id} (E: {zorp.energy.hunger:.1f}) decided action: {action_choice.action_type}, Details: {action_choice}")
            
            # Apply the chosen action
            self._apply_action(zorp, action_choice, allow_logging_this_tick) # Pass logging flag
            
            # Zorp's internal passive state update (aging, base metabolism cost of living)
            zorp.passive_update(dt) 

            if not zorp.alive: # Check again if action or update killed it
                zorps_to_remove.append(zorp)

        for zorp_to_remove in zorps_to_remove: # Renamed zorp to zorp_to_remove
            if allow_logging_this_tick: # Also gate this log
                print(f"[ZorpWorld tick] Zorp {zorp_to_remove.id} removed from world.")
            self.remove_zorp(zorp_to_remove)
            # print(f"Zorp {zorp.id} removed from world.") # Original unconditional print

        # Regrow resources
        for y_coord in range(self.height):
            for x_coord in range(self.width):
                if self.resource_regrow_timer[y_coord, x_coord] > 0:
                    # DEBUG: Check timer value BEFORE decrementing
                    # print(f"[DEBUG TIMER] Before decrement: ({x_coord},{y_coord}) Timer: {self.resource_regrow_timer[y_coord, x_coord]}")
                    self.resource_regrow_timer[y_coord, x_coord] -= 1
                    # DEBUG: Check timer value AFTER decrementing
                    # print(f"[DEBUG TIMER] After decrement: ({x_coord},{y_coord}) Timer: {self.resource_regrow_timer[y_coord, x_coord]}")

                    if self.resource_regrow_timer[y_coord, x_coord] == 0:
                        # DEBUG: Confirm this block is reached when timer is zero
                        # print(f"[DEBUG TIMER ZERO] Timer reached zero for ({x_coord},{y_coord})")
                        original_tile_type_for_energy = self.map_data[y_coord, x_coord]
                        if (x_coord, y_coord) in self.consumed_tile_original_type:
                            original_tile_type_for_energy = self.consumed_tile_original_type[(x_coord, y_coord)]
                        self.resource_energy_map[y_coord, x_coord] = original_tile_type_for_energy.metadata.energy_value
                        self.map_visuals_dirty = True # Ensure visual update on regrowth for all types
                        print(f"[DEBUG REGEN] Resource at ({x_coord},{y_coord}) regrew to {self.resource_energy_map[y_coord, x_coord]:.1f} (Tile: {original_tile_type_for_energy.name})")
                        if allow_logging_this_tick:
                            log_msg = f"[ZorpWorld tick] Resource at ({x_coord},{y_coord}) energy regrew to {self.resource_energy_map[y_coord, x_coord]:.1f}."
                            if (x_coord, y_coord) in self.consumed_tile_original_type:
                                log_msg += f" Tile type changing back to {original_tile_type_for_energy.name}."
                            print(log_msg)
                        if (x_coord, y_coord) in self.consumed_tile_original_type:
                            self.map_data[y_coord, x_coord] = self.consumed_tile_original_type.pop((x_coord, y_coord))
                            self.map_visuals_dirty = True
                            print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: Regrowth at ({x_coord},{y_coord}) to {self.map_data[y_coord, x_coord].name}")
                            if allow_logging_this_tick:
                                print(f"[ZorpWorld tick] Tile at ({x_coord},{y_coord}) visually restored to {self.map_data[y_coord, x_coord].name}.")
                # DEBUG: If timer is not > 0, print its value to see if it's negative or already zero
                else:
                    if self.resource_regrow_timer[y_coord, x_coord] != 0: # Only print if it's not naturally zero (i.e. never consumed)
                        print(f"[DEBUG TIMER] Timer for ({x_coord},{y_coord}) is {self.resource_regrow_timer[y_coord, x_coord]} (not > 0, and not naturally 0)")

    def _apply_action(self, zorp: Zorp, action: ActionChoice, allow_log: bool) -> None:
        """Applies the Zorp's chosen action to the world and the Zorp itself."""
        zorp.energy.hunger -= ENERGY_COST_THINK

        if action.action_type == ActionType.MOVE:
            if action.move_delta:
                old_position = zorp.position
                dx, dy = action.move_delta
                new_position = (old_position[0] + dx, old_position[1] + dy)

                target_tile_passable = False
                target_tile_name = "Out_Of_Bounds"
                
                if self.is_within_bounds(new_position):
                    target_tile_enum: Tile = self.map_data[new_position[1], new_position[0]]
                    target_tile_name = target_tile_enum.name
                    target_tile_passable = target_tile_enum.metadata.passable

                log_msg_prefix = f"[ZorpWorld MOVE] Zorp {zorp.id} from {old_position} to {new_position} (Target: {target_tile_name}, Passable: {target_tile_passable}):"

                if self.is_within_bounds(new_position) and target_tile_passable: # Use pre-fetched target_tile_passable
                    zorp.position = new_position
                    self._update_zorp_spatial_map(zorp, old_position)
                    zorp.energy.hunger -= ENERGY_COST_MOVE
                    if allow_log: print(f"{log_msg_prefix} Success.")
                else:
                    # Bumped into wall or impassable terrain
                    zorp.energy.hunger -= ENERGY_COST_MOVE * 0.5 # Lesser penalty for failed move
                    if allow_log: print(f"{log_msg_prefix} Failed (Blocked or OOB).")
            else:
                # This case should ideally not happen if MOVE implies a move_delta
                # If it does, treat as a REST or a very short action
                zorp.energy.hunger -= ENERGY_COST_REST
                print(f"[ZorpWorld DEBUG] Zorp {zorp.id} MOVE action with no move_delta (treated as REST)")

        elif action.action_type == ActionType.EAT:
            # Try to eat from inventory first
            successfully_ate_from_inventory = zorp.attempt_eat_from_inventory()

            if successfully_ate_from_inventory:
                if allow_log:
                    print(f"[ZorpWorld _apply_action] Zorp {zorp.id} ATE FROM INVENTORY. New hunger: {zorp.energy.hunger:.1f}")
            else:
                zorp.energy.hunger -= ENERGY_COST_EAT_ATTEMPT
                tile_x, tile_y = zorp.position
                current_tile_enum: Tile = self.map_data[tile_y, tile_x]
                tile_meta = current_tile_enum.metadata
                if tile_meta.energy_value > 0 and tile_meta.resource_type in EDIBLE_RESOURCE_TYPES and self.resource_energy_map[tile_y, tile_x] > 0:
                    print(f"[DEBUG CONSUME] Zorp {zorp.id} ate {current_tile_enum.name} at ({tile_x},{tile_y}), depleting resource. Energy before: {zorp.energy.hunger:.1f}")
                    energy_gained = self.resource_energy_map[tile_y, tile_x]
                    zorp.energy.hunger += energy_gained # Gain energy (cost was already applied before this block)
                    zorp.energy.hunger = min(zorp.energy.hunger, zorp.max_energy)
                    
                    # DEBUGGING FOR APPLES/MUSHROOMS - Ungated log for EAT success (from world)
                    if current_tile_enum.metadata.resource_type == ResourceType.APPLES or current_tile_enum.metadata.resource_type == ResourceType.MUSHROOMS:
                        print(f"[DEBUG _apply_action EAT World Success] Zorp: {zorp.id} ATE {current_tile_enum.name}. Gained: {energy_gained:.1f}. New Zorp E: {zorp.energy.hunger:.1f}. New MapEnergy for tile: 0.0")

                    if allow_log:
                        print(f"[ZorpWorld _apply_action EAT] Zorp {zorp.id} ATE FROM WORLD TILE {current_tile_enum.name}, gained {energy_gained:.1f} E. Total E: {zorp.energy.hunger:.1f}")
                    
                    self.resource_energy_map[tile_y, tile_x] = 0.0
                    self.resource_regrow_timer[tile_y, tile_x] = REGROW_TICKS
                    self.map_visuals_dirty = True # Ensure visual update on consumption
                    
                    if current_tile_enum == Tile.MUSHROOM_PATCH:
                        self.consumed_tile_original_type[(tile_x, tile_y)] = Tile.MUSHROOM_PATCH
                        self.map_data[tile_y, tile_x] = Tile.FOREST_FLOOR
                        self.map_visuals_dirty = True
                        print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: MUSHROOM_PATCH consumed at ({tile_x},{tile_y})")
                        if allow_log: 
                             print(f"[ZorpWorld _apply_action EAT] MUSHROOM_PATCH at ({tile_x},{tile_y}) consumed, changed to FOREST_FLOOR.")
                    elif current_tile_enum == Tile.APPLE_TREE_LEAVES:
                        self.consumed_tile_original_type[(tile_x, tile_y)] = Tile.APPLE_TREE_LEAVES
                        self.map_data[tile_y, tile_x] = Tile.TREE_CANOPY
                        self.map_visuals_dirty = True
                        print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: APPLE_TREE_LEAVES consumed at ({tile_x},{tile_y})")
                        if allow_log: 
                            print(f"[ZorpWorld _apply_action EAT] APPLE_TREE_LEAVES at ({tile_x},{tile_y}) consumed, changed to TREE_CANOPY.")
                    
                    if allow_log:
                        print(f"[ZorpWorld _apply_action EAT] Tile {current_tile_enum.name} at ({tile_x},{tile_y}) depleted by world eat. Regrowth in {REGROW_TICKS} ticks.")
                else:
                    # Failed to eat from world tile (no energy, not edible, depleted)
                    # ENERGY_COST_EAT_ATTEMPT was already applied
                    # DEBUGGING FOR APPLES/MUSHROOMS - Ungated log for EAT fail (from world)
                    if current_tile_enum.metadata.resource_type == ResourceType.APPLES or current_tile_enum.metadata.resource_type == ResourceType.MUSHROOMS:
                         print(f"[DEBUG _apply_action EAT World Fail] Zorp: {zorp.id} at ({tile_x},{tile_y}) on {current_tile_enum.name}. Conditions: MetaEnergyOK={tile_meta.energy_value > 0}, EdibleListOK={tile_meta.resource_type in EDIBLE_RESOURCE_TYPES}, MapEnergyOK={self.resource_energy_map[tile_y, tile_x] > 0}. ZorpEnergy: {zorp.energy.hunger:.1f}")
                    
                    if allow_log:
                        print(f"[ZorpWorld _apply_action EAT] Zorp {zorp.id} FAILED to eat from world tile {current_tile_enum.name} (TileEnergyInMap: {self.resource_energy_map[tile_y, tile_x]:.1f}). Hunger: {zorp.energy.hunger:.1f}")

        elif action.action_type == ActionType.REPRODUCE:
            # --- SEXUAL REPRODUCTION LOGIC ---
            partner = None
            if hasattr(action, 'partner_id') and action.partner_id is not None:
                # Find the partner Zorp by UUID
                for candidate in self.all_zorps:
                    if candidate.id == action.partner_id:
                        partner = candidate
                        break
            if partner and partner.pending_reproduction_with == zorp.id and zorp.pending_reproduction_with == partner.id:
                # Both Zorps have mutually selected each other
                parent_max_energy = zorp.max_energy
                min_energy_to_reproduce = parent_max_energy * REPRODUCTION_MIN_ENERGY_THRESHOLD_FACTOR
                actual_reproduction_cost = parent_max_energy * REPRODUCTION_ACTUAL_COST_FACTOR
                child_initial_energy = parent_max_energy * REPRODUCTION_CHILD_INITIAL_ENERGY_FACTOR
                # Both must have enough energy
                if zorp.energy.hunger >= min_energy_to_reproduce and partner.energy.hunger >= min_energy_to_reproduce:
                    # Find a spawn location adjacent to either parent
                    spawn_location_found = False
                    best_spawn_pos: Optional[Tuple[int, int]] = None
                    deltas = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
                    np.random.shuffle(deltas)
                    for dx, dy in deltas:
                        for base_pos in [zorp.position, partner.position]:
                            check_pos = (base_pos[0] + dx, base_pos[1] + dy)
                            if (self.is_within_bounds(check_pos) and
                                self.is_tile_passable_for_zorp(check_pos) and
                                not self.get_zorps_at(check_pos)):
                                best_spawn_pos = check_pos
                                spawn_location_found = True
                                break
                        if spawn_location_found:
                            break
                    if spawn_location_found and best_spawn_pos is not None:
                        zorp.energy.hunger -= actual_reproduction_cost
                        partner.energy.hunger -= actual_reproduction_cost
                        new_zorp = Zorp(position=best_spawn_pos, world_ref=self, energy_component=None, age_component=None, genetics_component=None, inventory_component=None, genome_dict=None)
                        new_zorp.energy.hunger = child_initial_energy
                        if self._foraging_system:
                            self._foraging_system.add_new_zorp_inventory(new_zorp.id)
                        self.add_zorp(new_zorp)
                        self.last_reproduction_time[zorp.id] = self.current_game_time
                        self.last_reproduction_time[partner.id] = self.current_game_time
                        zorp.pending_reproduction_with = None
                        partner.pending_reproduction_with = None
                        if allow_log:
                            print(f"[ZorpWorld _apply_action SEXUAL REPRODUCE] Zorps {zorp.id} and {partner.id} reproduced. New Zorp {new_zorp.id} at {new_zorp.position} with E:{new_zorp.energy.hunger:.1f}. Parents E after: {zorp.energy.hunger:.1f}, {partner.energy.hunger:.1f}")
                    else:
                        if allow_log:
                            print(f"[ZorpWorld _apply_action SEXUAL REPRODUCE] Zorps {zorp.id} and {partner.id} could not find a spawn location.")
                else:
                    if allow_log:
                        print(f"[ZorpWorld _apply_action SEXUAL REPRODUCE] Zorps {zorp.id} and {partner.id} do not have enough energy to reproduce.")
            # If not mutual or not enough energy, do nothing (no asexual fallback)

        elif action.action_type == ActionType.EMIT_SIGNAL:
            # Placeholder: Zorp emits a signal. Real logic in Stage 3.
            zorp.energy.hunger -= ENERGY_COST_EMIT_SIGNAL
            if action.signal_vector is not None:
                zorp.current_signal = action.signal_vector
            # TODO: Make signal available to other Zorps via perception

        elif action.action_type == ActionType.REST:
            zorp.energy.hunger -= ENERGY_COST_REST
            # Potentially small energy gain or recovery from fatigue in future

        elif action.action_type == ActionType.PICK_UP:
            tile_x, tile_y = zorp.position
            current_tile_enum: Tile = self.map_data[tile_y, tile_x]
            tile_meta = current_tile_enum.metadata
            resource_on_tile = tile_meta.resource_type
            energy_on_tile = self.resource_energy_map[tile_y, tile_x]

            if allow_log:
                log_prefix = f"[ZorpWorld _apply_action PICK_UP Attempt] Zorp {zorp.id} at ({tile_x},{tile_y}) on {current_tile_enum.name} (Resource: {resource_on_tile.name if resource_on_tile else 'None'}, Energy: {energy_on_tile:.1f}). Zorp Inv: {zorp.inventory.carrying.name if zorp.inventory.carrying else 'None'}({zorp.inventory.amount})"
                print(log_prefix)

            can_pickup = (
                resource_on_tile in EDIBLE_RESOURCE_TYPES and 
                energy_on_tile > 0 and 
                (zorp.inventory.carrying is None or zorp.inventory.amount == 0) # Can only pick up if hands are empty
            )

            if can_pickup:
                print(f"[DEBUG CONSUME] Zorp {zorp.id} picked up {resource_on_tile.name} at ({tile_x},{tile_y}), depleting resource. Energy before: {zorp.energy.hunger:.1f}")
                zorp.inventory.carrying = resource_on_tile
                zorp.inventory.amount = 1 # Assuming picking up one unit
                
                # Deplete the resource from the world
                self.resource_energy_map[tile_y, tile_x] = 0.0
                self.resource_regrow_timer[tile_y, tile_x] = REGROW_TICKS
                self.map_visuals_dirty = True # Ensure visual update on consumption
                
                pickup_log_msg = f"Successfully picked up {resource_on_tile.name}. Inv: {zorp.inventory.carrying.name}({zorp.inventory.amount})."
                
                # Handle tile changes for specific resources (like MUSHROOM_PATCH, APPLE_TREE_LEAVES)
                if current_tile_enum == Tile.MUSHROOM_PATCH:
                    self.consumed_tile_original_type[(tile_x, tile_y)] = Tile.MUSHROOM_PATCH
                    self.map_data[tile_y, tile_x] = Tile.FOREST_FLOOR
                    self.map_visuals_dirty = True
                    pickup_log_msg += f" Tile changed to FOREST_FLOOR."
                    print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: MUSHROOM_PATCH picked up at ({tile_x},{tile_y})")
                elif current_tile_enum == Tile.APPLE_TREE_LEAVES:
                    self.consumed_tile_original_type[(tile_x, tile_y)] = Tile.APPLE_TREE_LEAVES
                    self.map_data[tile_y, tile_x] = Tile.TREE_CANOPY # The non-fruit bearing part of the tree
                    self.map_visuals_dirty = True
                    pickup_log_msg += f" Tile changed to TREE_CANOPY."
                    print(f"[!!! MAP_VISUALS_DIRTY SET TRUE !!!] Reason: APPLE_TREE_LEAVES picked up at ({tile_x},{tile_y})")
                # Add more specific tile changes if other resources also change the base tile when picked up
                
                if allow_log: print(f"{log_prefix} -> {pickup_log_msg}")
                zorp.current_action = None # Clear action after successful pickup
            else:
                fail_reason = ""
                if resource_on_tile not in EDIBLE_RESOURCE_TYPES: fail_reason += "Not an edible resource. "
                if energy_on_tile <= 0: fail_reason += "No energy on tile. "
                if not (zorp.inventory.carrying is None or zorp.inventory.amount == 0): fail_reason += "Inventory not empty. "
                if allow_log: print(f"{log_prefix} -> Failed. Reason: {fail_reason.strip()}")
                # No energy cost for failed pickup, as it's an opportunistic check

        if zorp.energy.hunger <= 0:
            zorp.alive = False

    # --- Perception Methods (Placeholders for Zorp.perceive) ---
    # These methods will be called by individual Zorps during their perceive phase.
    # For Stage 1, they return default values.

    def get_local_entropy(self, position: Tuple[int, int]) -> float:
        """Placeholder for getting local entropy at a position."""
        # TODO: Implement in Stage 5
        return 0.0 # Example: normalized entropy value

    def get_resources_at(self, position: Tuple[int, int], verbose: bool = False) -> np.ndarray:
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

        # DEBUGGING FOR APPLES/MUSHROOMS - Gated by verbose flag
        if verbose and (tile_meta.resource_type == ResourceType.APPLES or tile_meta.resource_type == ResourceType.MUSHROOMS):
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

    def get_last_reproduction_time(self, zorp_id: uuid.UUID) -> Optional[float]:
        """Returns the last reproduction time for a given Zorp, or None if never reproduced."""
        return self.last_reproduction_time.get(zorp_id) 

    def handle_zorp_death(self, zorp: Zorp) -> None:
        """Handles the death of a Zorp: removes it from the world and updates the map if needed.

        Args:
            zorp: The Zorp instance that died.
        """
        # Remove from world
        self.remove_zorp(zorp)
        # Optionally, convert the tile to ORGANIC_REMAINS
        tile_x, tile_y = zorp.position
        if 0 <= tile_y < self.height and 0 <= tile_x < self.width:
            from zorplife.world.tiles import Tile, ResourceType
            current_tile = self.map_data[tile_y, tile_x]
            # Only convert if not already remains
            if current_tile != Tile.ORGANIC_REMAINS:
                # Decrement counts for the tile being replaced
                if hasattr(self.map_generator, 'biome_counts') and current_tile in self.map_generator.biome_counts:
                    self.map_generator.biome_counts[current_tile] -= 1
                    if self.map_generator.biome_counts[current_tile] == 0:
                        del self.map_generator.biome_counts[current_tile]
                if hasattr(self.map_generator, 'resource_counts'):
                    old_resource = current_tile.metadata.resource_type
                    if old_resource != ResourceType.NONE and old_resource in self.map_generator.resource_counts:
                        self.map_generator.resource_counts[old_resource] -= 1
                        if self.map_generator.resource_counts[old_resource] == 0:
                            del self.map_generator.resource_counts[old_resource]
                # Set new tile and update counts
                self.map_data[tile_y, tile_x] = Tile.ORGANIC_REMAINS
                if hasattr(self.map_generator, 'biome_counts'):
                    self.map_generator.biome_counts[Tile.ORGANIC_REMAINS] = self.map_generator.biome_counts.get(Tile.ORGANIC_REMAINS, 0) + 1
                if hasattr(self.map_generator, 'resource_counts'):
                    from zorplife.world.tiles import ResourceType
                    self.map_generator.resource_counts[ResourceType.ORGANIC_MATTER] = self.map_generator.resource_counts.get(ResourceType.ORGANIC_MATTER, 0) + 1
                self.map_visuals_dirty = True
                print(f"[ZorpWorld _apply_action] Zorp {zorp.id} removed from world and tile at ({tile_x},{tile_y}) converted to ORGANIC_REMAINS.")