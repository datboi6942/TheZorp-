import uuid
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING, Set # Added Set
import random # For random lifespan
import numpy as np # For brain inputs/outputs

from ..agents.components import Position, Energy, Age, Genetics, Inventory
from ..world.tiles import Tile, ResourceType
# from ..config import GameConfig # Would be ideal for default sizes, if GameConfig is accessible here
from ..core.brain import ActionType, ActionChoice

# Default values for brain, if not overridden by Zorp's genome or __init__
DEFAULT_INPUT_SIZE = 10  # Example: proximity sensors, internal state
DEFAULT_HIDDEN_SIZE = 8
DEFAULT_OUTPUT_SIZE = 5 # Example: move_x, move_y, eat, sleep, emit_signal_strength

MAX_ENERGY = 100.0  # Default max energy
HUNGER_THRESHOLD_PERCENT = 0.5 # Eat if energy < 50% of max_energy

# Forward declaration for ZorpWorld
if TYPE_CHECKING:
    from .world import ZorpWorld # Corrected import path assuming world.py is in the same directory (core)


class Zorp:
    """Represents a Zorp agent in the simulation.

    Combines ECS-like components for state with its own methods for behavior.
    The genome now dictates behavioral parameters rather than a full neural network structure initially.
    """
    def __init__(
        self,
        position: Tuple[int, int],
        world_ref: Optional['ZorpWorld'] = None,
        energy_component: Optional[Energy] = None, # Renamed for clarity
        age_component: Optional[Age] = None,       # Renamed for clarity
        genetics_component: Optional[Genetics] = None, # Renamed for clarity
        inventory_component: Optional[Inventory] = None, # Renamed for clarity
        genome_dict: Optional[Dict[str, Any]] = None, # Renamed for clarity
        # Parameters for a potential neural network brain (can be part of genome_dict too)
        # input_size: int = DEFAULT_INPUT_SIZE,
        # hidden_size: int = DEFAULT_HIDDEN_SIZE,
        # output_size: int = DEFAULT_OUTPUT_SIZE,
    ):
        self.id: uuid.UUID = uuid.uuid4() # Changed from str to uuid.UUID
        self.position: Tuple[int, int] = position # Tile coordinates (x, y)
        
        # Initialize components
        self.energy: Energy = energy_component if energy_component is not None else Energy(hunger=random.uniform(50, 100), sleepiness=0.0)
        self.age: Age = age_component if age_component is not None else Age(current_age=0.0, lifespan=random.uniform(60, 100))
        self.genetics: Genetics = genetics_component if genetics_component is not None else Genetics()
        self.inventory: Inventory = inventory_component if inventory_component is not None else Inventory()
        
        self.alive: bool = True
        self.max_energy: float = MAX_ENERGY # Use defined constant
        self.world_ref: Optional['ZorpWorld'] = world_ref
        
        # Genome for behavioral parameters
        self.genome: Dict[str, Any] = genome_dict if genome_dict is not None else self._initialize_default_genome_parameters()

        # Behavioral state (initialized here, within __init__)
        self.current_action: Optional[str] = None # e.g., "eating", "sleeping", "wandering"
        self.action_target: Any = None # e.g., target Zorp for mating, tile for food
        self.action_progress: float = 0.0 # For actions that take time

        # Memory for food locations
        self.known_food_locations: Set[Tuple[int, int]] = set() # (x, y) coordinates

        # Communication related (placeholders for future development)
        self.known_words: Dict[str, Any] = {} # word: concept_representation
        self.last_communication_time: float = 0.0 # Game time of last communication

        # Sexual reproduction intent (None if not seeking, else partner's UUID)
        self.pending_reproduction_with: Optional[uuid.UUID] = None

        # If a neural network brain is to be used, it would be initialized here,
        # possibly using parameters from self.genome or dedicated __init__ args.
        # self.brain = ZorpBrain(input_size, hidden_size, output_size, genome_for_brain)
        # For now, behavior is rule-based using self.genome parameters.

    def _initialize_default_genome_parameters(self) -> Dict[str, Any]:
        """Initializes a basic genome with default behavioral parameters."""
        return {
            "metabolic_rate_hunger": random.uniform(0.015, 0.035),
            "metabolic_rate_sleep": random.uniform(0.05, 0.15),
            "reproduction_energy_threshold": 20.0,
            "min_reproduction_age": 5.0,  # Lowered for testing
            "max_reproduction_age": 100.0,
            "reproduction_cooldown": 3.0,  # Cooldown is currently bypassed in can_reproduce for debugging
            "hunger_threshold_for_eating": random.uniform(30.0, self.max_energy * HUNGER_THRESHOLD_PERCENT),
            "sleepiness_threshold_for_sleeping": random.uniform(70.0, 90.0),
            "preferred_food_types": [
                ResourceType.APPLES,
                ResourceType.MUSHROOMS
            ],
            "eating_energy_gain": 50.0,
            "perception_radius": random.uniform(3, 7),
            "mutation_rate": 0.05,
            "lifespan_min": 30.0,  # Increased
            "lifespan_max": 150.0, # Increased
            "combat_proficiency": { # Placeholder for future combat mechanics
                "attack_power": random.uniform(5.0, 15.0),
                "defense_value": random.uniform(3.0, 10.0),
            },
        }

    def passive_update(self, dt: float) -> None:
        """Updates the Zorp's age and metabolic needs based on genome parameters."""
        if not self.alive: return

        self.age.current_age += dt
        self.energy.hunger -= self.genome.get("metabolic_rate_hunger", 0.5) * dt
        self.energy.sleepiness += self.genome.get("metabolic_rate_sleep", 0.3) * dt
        self.energy.sleepiness = min(self.energy.sleepiness, 100.0) # Cap sleepiness

        if self.energy.hunger <= 0 or \
           self.age.current_age >= self.age.lifespan or \
           self.energy.sleepiness >= 100:
            self.die()

    def die(self) -> None:
        """Marks the Zorp as dead and notifies the world."""
        if not self.alive: return
        self.alive = False
        death_reason = "unknown"
        if self.energy.hunger <= 0: death_reason = "starvation"
        elif self.energy.sleepiness >= 100: death_reason = "exhaustion"
        elif self.age.current_age >= self.age.lifespan: death_reason = "old age"
        
        # Try to get a short genetic ID for logging, if the method exists
        genetic_id_short = "N/A"
        if hasattr(self.genetics, 'genetic_id') and self.genetics.genetic_id and hasattr(self.genetics.genetic_id, 'hex'):
            genetic_id_short = self.genetics.genetic_id.hex[:8]
        
        print(f"Zorp {self.id} ({genetic_id_short}) died of {death_reason} at ({self.position[0]},{self.position[1]}) age {self.age.current_age:.2f}.")
        if self.world_ref:
            self.world_ref.handle_zorp_death(self)

    def can_reproduce(self, current_game_time: float) -> bool:
        """Checks if the Zorp is currently able to reproduce based on genome and world state."""
        # Moved print upfront for better debugging if early exit
        print(f"[REPRODUCE DEBUG ENTRY] Zorp {self.id} attempting can_reproduce. Alive: {self.alive}, WorldRef: {'SET' if self.world_ref else 'NONE'}. Current Game Time: {current_game_time:.2f}")

        if not self.alive or not self.world_ref:
            print(f"[REPRODUCE DEBUG EXIT] Zorp {self.id} cannot reproduce: not alive or no world_ref.")
            return False
        
        # --- RE-ENABLE ACTUAL COOLDOWN LOGIC ---
        # cooldown_passed = True # DEBUG BYPASS
        # print(f"[REPRODUCE DEBUG COOLDOWN] Zorp {self.id}: Cooldown check bypassed, cooldown_passed forced to True for debugging.") # DEBUG BYPASS
        last_reproduction_time = self.world_ref.get_last_reproduction_time(self.id)
        cooldown_passed = True
        if last_reproduction_time is not None:
            reproduction_cooldown_duration = self.genome.get("reproduction_cooldown", 10.0)
            if current_game_time - last_reproduction_time < reproduction_cooldown_duration:
                cooldown_passed = False
                print(f"[REPRODUCE DEBUG COOLDOWN] Zorp {self.id}: Cooldown NOT passed. Last repro: {last_reproduction_time:.2f}, current: {current_game_time:.2f}, needed: {reproduction_cooldown_duration:.1f}")
            else:
                print(f"[REPRODUCE DEBUG COOLDOWN] Zorp {self.id}: Cooldown PASSED. Last repro: {last_reproduction_time:.2f}, current: {current_game_time:.2f}, needed: {reproduction_cooldown_duration:.1f}")
        else:
            print(f"[REPRODUCE DEBUG COOLDOWN] Zorp {self.id}: No last reproduction time found, cooldown PASSED by default.")
        # --- END COOLDOWN LOGIC ---

        # Match world logic: must have at least 30% of max energy
        energy_threshold = self.max_energy * 0.3
        energy_ok = self.energy.hunger >= energy_threshold
        
        min_age = self.genome.get("min_reproduction_age", 15.0)
        max_age = self.genome.get("max_reproduction_age", 80.0)
        age_ok = min_age <= self.age.current_age <= max_age
        
        can_repro = energy_ok and age_ok and cooldown_passed # cooldown_passed is now always true for this test
        
        # Original detailed print, now we can see if it's reached
        print(f"[REPRODUCE DEBUG EVAL] Zorp {self.id}: energy={self.energy.hunger:.1f} (actual_thresh={energy_threshold:.1f}), age={self.age.current_age:.1f} (range={min_age:.1f}-{max_age:.1f}), current_cooldown_passed_value={cooldown_passed}, energy_ok={energy_ok}, age_ok={age_ok}, FINAL can_reproduce_result={can_repro}")
        return can_repro

    def attempt_eat_from_inventory(self) -> bool:
        """Attempts to eat food from inventory. Returns True if successful."""
        if not self.alive:
            return False

        preferred_foods = self.genome.get("preferred_food_types", [])
        hunger_trigger = self.genome.get("hunger_threshold_for_eating", self.max_energy * 0.4)
        ate = False
        # Eat as long as we have food and are below the hunger threshold
        while self.inventory.carrying in preferred_foods and self.inventory.amount > 0 and self.energy.hunger < hunger_trigger:
            eaten_food_type = self.inventory.carrying
            energy_gain = self.genome.get("eating_energy_gain", 50.0)
            self.energy.hunger = min(self.max_energy, self.energy.hunger + energy_gain)
            self.inventory.amount -= 1
            ate = True
            print(f"[EAT] Zorp {self.id} ate {eaten_food_type.name if eaten_food_type else 'something'} from inventory. Hunger: {self.energy.hunger:.1f}")
            if self.inventory.amount == 0:
                self.inventory.carrying = None
        if ate:
            self.current_action = None  # Reset action after eating
        return ate

    def attempt_sleep(self) -> None:
        """Zorp sleeps to reduce sleepiness instantly."""
        if not self.alive: return
        self.energy.sleepiness = 0.0
        print(f"Zorp {self.id} slept. Sleepiness: {self.energy.sleepiness:.1f}")
        self.current_action = None # Reset action after sleeping

    def decide_action(self, world: 'ZorpWorld', allow_log: bool) -> ActionChoice:
        """Decides the Zorp's next action based on needs and genome, returns an ActionChoice for the world to apply.

        Args:
            world: The ZorpWorld instance.
            allow_log: Whether to print debug info.
        Returns:
            ActionChoice: The action the Zorp intends to take.
        """
        if not self.alive:
            self.current_action = "dead"
            return ActionChoice(action_type=ActionType.REST)

        # FORCE REPRODUCTION PRIORITY FOR DEBUGGING
        # Add a new debug log here to see what decide_action thinks about can_reproduce
        can_repro_result_for_decide_action = False
        if self.world_ref: # Check world_ref first to avoid error if it's None
            can_repro_result_for_decide_action = self.can_reproduce(self.world_ref.current_game_time)
        
        print(f"[DECIDE_ACTION REPRO CHECK] Zorp {self.id}: self.world_ref is {'SET' if self.world_ref else 'NONE'}. self.can_reproduce() returned: {can_repro_result_for_decide_action} at game time {self.world_ref.current_game_time if self.world_ref else 'N/A'}.")

        if self.world_ref and can_repro_result_for_decide_action: # Use the pre-calculated result
            self.current_action = "seeking_mate" # This might be misnomer for asexual
            print(f"[REPRODUCE DEBUG] Zorp {self.id} is eligible and is choosing to reproduce (from decide_action).")
            if allow_log:
                print(f"[Zorp {self.id} decide_action DEBUG] Action: ActionChoice(action_type=ActionType.REPRODUCE) (Reason: Eligible for reproduction)")
            # --- SEXUAL REPRODUCTION LOGIC ---
            # Scan adjacent tiles for eligible partner
            x, y = self.position
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not world.is_within_bounds((nx, ny)):
                        continue
                    for partner in world.get_zorps_at((nx, ny)):
                        if (
                            partner.id != self.id and
                            partner.can_reproduce(world.current_game_time) and
                            partner.pending_reproduction_with is None
                        ):
                            # Mutual intent: set both pending_reproduction_with
                            self.pending_reproduction_with = partner.id
                            partner.pending_reproduction_with = self.id
                            self.current_action = "seeking_mate"
                            if allow_log:
                                print(f"[REPRODUCE DEBUG] Zorp {self.id} and {partner.id} mutually selected for reproduction.")
                            return ActionChoice(action_type=ActionType.REPRODUCE, partner_id=partner.id)
            # If no partner found, do not reproduce this tick
            return ActionChoice(action_type=ActionType.REPRODUCE)

        # Priority 1: Sleep if exhausted
        if self.energy.sleepiness >= self.genome.get("sleepiness_threshold_for_sleeping", 80.0):
            self.current_action = "sleeping"
            action_to_take = ActionChoice(action_type=ActionType.REST)
            if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Exhausted)")
            return action_to_take

        # Priority 1.5: Critically Low Energy Handling
        emergency_rest_threshold = self.genome.get("emergency_rest_threshold_factor", 0.15) * self.max_energy
        is_critically_low_energy = self.energy.hunger < emergency_rest_threshold

        preferred_foods = self.genome.get("preferred_food_types", [])
        has_food_in_inventory = self.inventory.carrying in preferred_foods and self.inventory.amount > 0

        if is_critically_low_energy:
            if has_food_in_inventory:
                self.current_action = "eating_from_inventory_critical"
                action_to_take = ActionChoice(action_type=ActionType.EAT)
                if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Critically Low Energy {self.energy.hunger:.1f}/{emergency_rest_threshold:.1f} BUT has food)")
                return action_to_take
            else:
                self.current_action = "emergency_resting"
                action_to_take = ActionChoice(action_type=ActionType.REST)
                if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Critically Low Energy {self.energy.hunger:.1f}/{emergency_rest_threshold:.1f} AND no food)")
                return action_to_take

        # Priority 2: Eat if hungry and has food (standard hunger, not critical)
        hunger_trigger = self.genome.get("hunger_threshold_for_eating", self.max_energy * 0.4)
        if self.energy.hunger < hunger_trigger: # This implies not critically low, or critical case handled above
            if has_food_in_inventory:
                self.current_action = "eating_from_inventory"
                action_to_take = ActionChoice(action_type=ActionType.EAT)
                if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Hungry with Food)")
                return action_to_take
            else: # Hungry but no food in inventory
                # Check current tile for preferred food BEFORE deciding to move/forage
                current_tile_enum: Tile = world.map_data[self.position[1], self.position[0]]
                current_tile_resource_type = current_tile_enum.metadata.resource_type
                # Get food_metric, water_metric, other_metric from world.get_resources_at()
                # world.get_resources_at() uses self.resource_energy_map for food_metric.
                current_tile_resource_info = world.get_resources_at(self.position, verbose=False) 
                food_metric_on_tile = current_tile_resource_info[0]
                actual_energy_on_tile_from_map = world.resource_energy_map[self.position[1], self.position[0]]

                if allow_log:
                    log_msg = (
                        f"[Zorp {self.id} decide_action - Tile Check for PICK_UP] "
                        f"Pos: {self.position}, TileName: {current_tile_enum.name}, "
                        f"TileResource: {current_tile_resource_type.name if current_tile_resource_type else 'None'}, "
                        f"InPreferredFood: {current_tile_resource_type in preferred_foods}, "
                        f"FoodMetric (from get_resources_at): {food_metric_on_tile:.2f}, "
                        f"ActualEnergyInMap: {actual_energy_on_tile_from_map:.2f}"
                    )
                    print(log_msg)
                    if current_tile_resource_type in preferred_foods:
                        print(f"[FOOD RECOGNITION] Zorp {self.id} recognizes {current_tile_resource_type.name} as food on tile {self.position}.")

                if current_tile_resource_type in preferred_foods and food_metric_on_tile > 0: # food_metric from get_resources_at()
                    # If inventory is empty or carrying something different, try to pick up
                    if self.inventory.carrying is None or self.inventory.carrying != current_tile_resource_type:
                        self.current_action = "picking_up_food"
                        action_to_take = ActionChoice(action_type=ActionType.PICK_UP)
                        if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Hungry, no food in inv, found {current_tile_resource_type.name} on current tile {self.position} with food_metric > 0)")
                        return action_to_take
                    # If already carrying the same type, and it's stackable (not implemented yet), could pick up more
                    # For now, if carrying same type, will proceed to foraging/moving to find a new spot or eat existing

                # Priority 3: Forage (Move to find food)
                self.current_action = "foraging"
                
                # Step 1: Try to find food from memory
                known_food_direction = self.get_direction_to_nearest_known_food(world)

                if known_food_direction:
                    dx, dy = known_food_direction
                    target_x, target_y = self.position[0] + dx, self.position[1] + dy
                    # Check if the target tile is within bounds and passable
                    if world.is_within_bounds((target_x, target_y)):
                        target_tile_enum = world.map_data[target_y, target_x]
                        if target_tile_enum.metadata.passable:
                            print(f"[MEMORY USE] Zorp {self.id} moving toward remembered food at ({target_x},{target_y}) from memory. Memory size: {len(self.known_food_locations)}")
                            action_to_take = ActionChoice(action_type=ActionType.MOVE, move_delta=(dx, dy))
                            if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Foraging - moving towards KNOWN food at {target_x},{target_y} from memory)")
                            return action_to_take
                        else:
                            # Not passable, remove from memory
                            if (target_x, target_y) in self.known_food_locations:
                                self.known_food_locations.remove((target_x, target_y))
                                print(f"[MEMORY REMOVE] Zorp {self.id} removed unreachable food at ({target_x},{target_y}) (blocked). Memory size: {len(self.known_food_locations)}")
                    else:
                        # Out of bounds, remove from memory
                        if (target_x, target_y) in self.known_food_locations:
                            self.known_food_locations.remove((target_x, target_y))
                            print(f"[MEMORY REMOVE] Zorp {self.id} removed unreachable food at ({target_x},{target_y}) (OOB). Memory size: {len(self.known_food_locations)}")
                    # After removal, try again next tick
                    # Fall through to foraging scan if no valid memory move
                else:
                    # Step 2: No known food in memory or all known spots are now empty, actively scan surroundings
                    if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] No suitable food in memory. Actively scanning...")
                    food_direction = self.find_nearest_food_direction(world) # This will also update memory

                    if food_direction:
                        dx, dy = food_direction
                        action_to_take = ActionChoice(action_type=ActionType.MOVE, move_delta=(dx, dy))
                        if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Foraging - moving towards NEWLY SEEN food at {food_direction})")
                        return action_to_take
                    else:
                        # No food found by scanning either, fall back to original foraging behavior (rest or random move)
                        if random.random() < 0.20: # 20% chance to rest (more aggressive foraging)
                            action_to_take = ActionChoice(action_type=ActionType.REST)
                            if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Foraging - no food in sight or memory, decided to rest due to 20% chance)")
                            return action_to_take
                        else:
                            # Random exploratory move
                            dx = random.choice([-1, 0, 1])
                            dy = random.choice([-1, 0, 1])
                            if dx == 0 and dy == 0: # Avoid null move costing energy
                                action_to_take = ActionChoice(action_type=ActionType.REST)
                                if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Foraging - no food, null random move (0,0), defaulting to REST)")
                                return action_to_take
                            action_to_take = ActionChoice(action_type=ActionType.MOVE, move_delta=(dx, dy))
                            if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Foraging - no food in sight or memory, random exploration)")
                            return action_to_take
        
        # Priority 4: Wander if not sleepy or hungry enough to eat/forage
        self.current_action = "wandering"
        # Add a chance to rest instead of always moving
        if random.random() < 0.60: # Was 0.33: 60% chance to rest
            action_to_take = ActionChoice(action_type=ActionType.REST)
            if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Wandering, but decided to rest due to 60% chance)")
            return action_to_take

        # Wandering implies movement, generate a random move_delta
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0: # Avoid null move costing energy
            action_to_take = ActionChoice(action_type=ActionType.REST)
            if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Wandering, null move (0,0) chosen, defaulting to REST)")
            return action_to_take
            
        action_to_take = ActionChoice(action_type=ActionType.MOVE, move_delta=(dx, dy))
        if allow_log: print(f"[Zorp {self.id} decide_action DEBUG] Action: {action_to_take} (Reason: Wandering)")
        return action_to_take

    def execute_action(self, dt: float) -> None:
        """Executes the Zorp's current action."""
        if not self.alive or not self.current_action or not self.world_ref:
            return

        action_handlers = {
            "sleeping": self.attempt_sleep,
            "eating_from_inventory": self.attempt_eat_from_inventory,
            "foraging": lambda: None, # ForagingSystem will handle the actual foraging action. Zorp just signals intent.
            "wandering": lambda: self._wander(dt),
            "dropping_item": self.attempt_drop_item, # Added drop action handler
            # "seeking_mate": lambda: self._wander(dt), # Wander while seeking mate for now
            "dead": lambda: None
        }

        handler = action_handlers.get(self.current_action)
        if handler:
            handler()
        else:
            print(f"Warning: Zorp {self.id} has unknown action: {self.current_action}. Defaulting to wander.")
            self._wander(dt)

    def _wander(self, dt: float) -> None:
        """Simple random walk movement for one step."""
        if not self.world_ref or not self.world_ref.map_data or not self.world_ref.map_generator: return

        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        if dx == 0 and dy == 0: return # No movement

        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        if 0 <= new_x < self.world_ref.map_generator.width and \
           0 <= new_y < self.world_ref.map_generator.height:
            if self.world_ref.is_tile_passable(new_x, new_y):
                self.position = (new_x, new_y)

    def get_render_details(self) -> Dict[str, Any]:
        """Returns a dictionary of details needed for rendering."""
        return {
            "id": str(self.id), # Keep as string for dict keys in RenderSystem
            "position": self.position,
            "energy": self.energy.hunger,
            "max_energy": self.max_energy,
            "alive": self.alive,
            "carrying_item": self.inventory.carrying is not None and self.inventory.amount > 0,
            "carrying_type_name": self.inventory.carrying.name if (self.inventory.carrying and self.inventory.amount > 0) else None
        }

    # --- Placeholder communication methods ---
    def generate_sound_for_concept(self, concept: str) -> Any:
        # Actual sound generation/selection would go here
        return f"sound_for_{concept}" 

    def communicate(self, message_concept: str, target: Optional['Zorp'] = None) -> None:
        if not self.alive or not self.world_ref: return
        
        sound = self.generate_sound_for_concept(message_concept)
        # In a real implementation, this would interact with an audio system
        print(f"Zorp {self.id} communicates '{message_concept}' (sound: {sound})")
        self.last_communication_time = self.world_ref.current_game_time

    def __repr__(self) -> str:
        return (f"Zorp(id={str(self.id)[:8]}, pos={self.position}, "
                f"hunger={self.energy.hunger:.1f}, sleep={self.energy.sleepiness:.1f}, "
                f"age={self.age.current_age:.1f}/{self.age.lifespan:.0f}, "
                f"alive={self.alive}, inv_item={self.inventory.carrying.name if self.inventory.carrying else 'None'}({self.inventory.amount}), "
                f"action='{self.current_action}')")

    def attempt_drop_item(self) -> bool:
        """Attempts to drop the carried item onto the current tile.

        Returns:
            bool: True if the item was successfully dropped, False otherwise.
        """
        if not self.alive or not self.world_ref or not self.world_ref.map_generator:
            return False

        if self.inventory.carrying is None or self.inventory.amount == 0:
            print(f"Zorp {self.id} tried to drop an item but has an empty inventory.")
            self.current_action = None # Nothing to drop
            return False

        carried_resource_type = self.inventory.carrying
        tile_x, tile_y = self.position

        # Ensure position is valid before accessing map_data
        if not (0 <= tile_y < self.world_ref.map_generator.height and 0 <= tile_x < self.world_ref.map_generator.width):
            print(f"Zorp {self.id} at invalid position ({tile_x},{tile_y}) cannot drop item.")
            self.current_action = "wandering" # Or None
            return False
            
        current_tile_on_ground = self.world_ref.map_data[tile_y, tile_x]

        # Define which tiles are suitable for dropping items on
        droppable_on_tiles = [Tile.GRASS, Tile.DIRT]
        if current_tile_on_ground not in droppable_on_tiles:
            print(f"Zorp {self.id} cannot drop {carried_resource_type.name} on {current_tile_on_ground.name} at ({tile_x},{tile_y}). Tile not suitable.")
            # Maybe change action to wander to find a suitable spot
            self.current_action = "wandering" 
            return False

        # Mapping from ResourceType being carried to the Tile it becomes when dropped.
        # This simplifies things by reusing existing resource-providing tiles.
        # TODO: Consider creating specific "ITEM_ON_GROUND" tiles if needed.
        resource_to_tile_map: Dict[ResourceType, Optional[Tile]] = {
            ResourceType.APPLES: Tile.APPLE_TREE_LEAVES, # Changed FRUIT to APPLES, and target to APPLE_TREE_LEAVES (harvestable part)
            ResourceType.MUSHROOMS: Tile.MUSHROOM_PATCH,
            ResourceType.ORGANIC_MATTER: Tile.ORGANIC_REMAINS,
            # Other resources might not be 'droppable' by Zorps in this way
        }

        new_tile_to_place = resource_to_tile_map.get(carried_resource_type)

        if new_tile_to_place is None:
            print(f"Zorp {self.id} is carrying {carried_resource_type.name}, which has no defined dropped tile. Cannot drop.")
            self.inventory.carrying = None # Discard the item if it's undroppable? Or Zorp keeps carrying?
            self.inventory.amount = 0
            self.current_action = None
            return False
        
        # Proceed with dropping the item
        # 1. Decrement counts for the tile being replaced (current_tile_on_ground)
        old_resource_on_ground = current_tile_on_ground.metadata.resource_type
        if current_tile_on_ground in self.world_ref.map_generator.biome_counts:
            self.world_ref.map_generator.biome_counts[current_tile_on_ground] -= 1
            if self.world_ref.map_generator.biome_counts[current_tile_on_ground] == 0:
                del self.world_ref.map_generator.biome_counts[current_tile_on_ground]
        
        if old_resource_on_ground != ResourceType.NONE and old_resource_on_ground in self.world_ref.map_generator.resource_counts:
            self.world_ref.map_generator.resource_counts[old_resource_on_ground] -= 1
            if self.world_ref.map_generator.resource_counts[old_resource_on_ground] == 0:
                del self.world_ref.map_generator.resource_counts[old_resource_on_ground]

        # 2. Update map data to the new dropped item tile
        self.world_ref.map_data[tile_y, tile_x] = new_tile_to_place
        
        # 3. Increment counts for the new tile and its resource
        self.world_ref.map_generator.biome_counts[new_tile_to_place] = self.world_ref.map_generator.biome_counts.get(new_tile_to_place, 0) + 1
        new_resource_from_dropped_item = new_tile_to_place.metadata.resource_type
        if new_resource_from_dropped_item != ResourceType.NONE:
            self.world_ref.map_generator.resource_counts[new_resource_from_dropped_item] = self.world_ref.map_generator.resource_counts.get(new_resource_from_dropped_item, 0) + 1
            
        self.world_ref.map_visuals_dirty = True

        print(f"Zorp {self.id} dropped {carried_resource_type.name} at ({tile_x},{tile_y}). Tile changed to {new_tile_to_place.name}.")
        
        self.inventory.carrying = None
        self.inventory.amount = 0
        self.current_action = None # Reset action after successful drop
        return True

    def find_nearest_food_direction(self, world: 'ZorpWorld') -> Optional[Tuple[int, int]]:
        """Scans nearby tiles for preferred food, returns direction to the nearest one, AND updates food memory.

        Args:
            world: The ZorpWorld instance.

        Returns:
            Optional[Tuple[int, int]]: (dx, dy) direction to the nearest food, or None if no food is found.
        """
        if world.map_data is None or not self.alive:
            return None

        perception_radius = int(self.genome.get("perception_radius", 5)) # Default to 5 if not in genome
        preferred_foods = self.genome.get("preferred_food_types", [])
        current_x, current_y = self.position

        min_dist_sq = float('inf')
        best_target_pos: Optional[Tuple[int, int]] = None
        
        height, width = world.map_data.shape
        for r_offset in range(-perception_radius, perception_radius + 1):
            for c_offset in range(-perception_radius, perception_radius + 1):
                if r_offset == 0 and c_offset == 0:
                    continue # Skip current tile

                check_x, check_y = current_x + c_offset, current_y + r_offset

                if not (0 <= check_x < width and 0 <= check_y < height):
                    continue # Out of bounds

                tile_enum: Tile = world.map_data[check_y, check_x] # map_data is (row, col) which is (y, x)
                tile_resource_type = tile_enum.metadata.resource_type
                
                # Update memory if this is a preferred food type
                if tile_resource_type in preferred_foods:
                    if (check_x, check_y) not in self.known_food_locations:
                        self.known_food_locations.add((check_x, check_y))
                        print(f"[MEMORY ADD] Zorp {self.id} saw {tile_resource_type.name} at ({check_x},{check_y}). Memory size: {len(self.known_food_locations)}")


                # Check if this tile contains preferred food for immediate action
                if tile_resource_type in preferred_foods:
                    # Check actual energy on tile to ensure it's not depleted
                    # world.resource_energy_map is a numpy array, use array indexing
                    energy_on_tile = world.resource_energy_map[check_y, check_x]

                    if energy_on_tile > 0: # Only consider it if there's actual energy
                        dist_sq = c_offset**2 + r_offset**2
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_target_pos = (check_x, check_y)
        
        if best_target_pos:
            # Calculate direction vector (dx, dy)
            # Normalize (approximately) to unit steps if not already
            dx = np.sign(best_target_pos[0] - current_x)
            dy = np.sign(best_target_pos[1] - current_y)
            return int(dx), int(dy) # Ensure integer deltas
        
        return None

    def get_direction_to_nearest_known_food(self, world: 'ZorpWorld') -> Optional[Tuple[int, int]]:
        """Checks memory for the nearest valid food source and returns direction.
        Removes invalid/depleted food locations from memory.

        Args:
            world: The ZorpWorld instance.

        Returns:
            Optional[Tuple[int, int]]: (dx, dy) direction to the nearest known food, or None.
        """
        if not self.alive or not self.known_food_locations or world.map_data is None:
            return None

        preferred_foods = self.genome.get("preferred_food_types", [])
        current_x, current_y = self.position
        
        min_dist_sq = float('inf')
        best_target_pos: Optional[Tuple[int, int]] = None
        locations_to_remove: Set[Tuple[int, int]] = set()

        for food_x, food_y in list(self.known_food_locations): # Iterate over a copy for safe removal
            # Validate food location is still valid and has food
            if not (0 <= food_x < world.map_data.shape[1] and 0 <= food_y < world.map_data.shape[0]):
                locations_to_remove.add((food_x, food_y))
                continue

            tile_enum: Tile = world.map_data[food_y, food_x]
            tile_resource_type = tile_enum.metadata.resource_type
            energy_on_tile = world.resource_energy_map[food_y, food_x]

            if tile_resource_type not in preferred_foods or energy_on_tile <= 0:
                # print(f"[MEMORY REMOVE] Zorp {self.id} removing ({food_x},{food_y}) - {tile_resource_type.name if tile_resource_type else 'N/A'} E:{energy_on_tile:.1f}. Memory size: {len(self.known_food_locations)-1}")
                locations_to_remove.add((food_x, food_y))
                continue

            dist_sq = (food_x - current_x)**2 + (food_y - current_y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_target_pos = (food_x, food_y)

        for loc in locations_to_remove:
            if loc in self.known_food_locations:
                self.known_food_locations.remove(loc)
                print(f"[MEMORY REMOVE] Zorp {self.id} removed depleted/invalid food at {loc}. Memory size: {len(self.known_food_locations)}")
        
        if best_target_pos:
            dx = np.sign(best_target_pos[0] - current_x)
            dy = np.sign(best_target_pos[1] - current_y)
            return int(dx), int(dy)
            
        return None