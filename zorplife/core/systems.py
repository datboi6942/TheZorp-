import esper
import pyglet
from pyglet.graphics import Group # Changed from OrderedGroup
from typing import Any, TYPE_CHECKING, Optional, List, Dict, Tuple
import numpy as np # For type hint np.ndarray
from zorplife.ui.camera import Camera # Import Camera
from zorplife.ui.sprites import SpriteSheet # Import SpriteSheet
from zorplife.world.tiles import Tile, ResourceType # For type hint & ORGANIC_MATTER
from zorplife.agents.components import Position, Energy, Age, Genetics, AgentMarker # Agent components
import random
from pyglet import math # Import pyglet.math for Mat4 identity

# if TYPE_CHECKING:
#     # No longer needed as esper.World object is not used.
#     pass

if TYPE_CHECKING:
    from ..world.world import ZorpWorld # For type hinting world_ref
    # from ..core.zorp import Zorp # Zorp type is available via ZorpWorld.all_zorps

class System(esper.Processor):
    """Base class for all systems in the ECS.
    
    The esper module functions (e.g., esper.get_components) should be used directly
    by importing esper in the system's file.
    """
    # world: esper.World # Removed, esper v3 uses module level functions.

    def __init__(self) -> None:
        super().__init__()

class InputSystem(System):
    """Handles player input for camera control and other actions."""
    def __init__(self, window: pyglet.window.Window, camera: Camera) -> None:
        super().__init__()
        self.window = window
        self.camera = camera
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.push_handlers(self)
        
        # Register mouse scroll handler directly on the window
        # because InputSystem.process is driven by dt, not events directly for scroll
        self.window.on_mouse_scroll = self.handle_mouse_scroll

    def process(self, dt: float) -> None:
        # print(f"InputSystem.process FIRING, dt: {dt:.4f}") # DEBUG: Confirm system is processing - This can be removed now or kept for verbosity
        dx, dy = 0.0, 0.0

        # Debug print for key states - REMOVED
        # print(f"Keys - LEFT: {self.keys[pyglet.window.key.LEFT]}, RIGHT: {self.keys[pyglet.window.key.RIGHT]}, UP: {self.keys[pyglet.window.key.UP]}, DOWN: {self.keys[pyglet.window.key.DOWN]}, A: {self.keys[pyglet.window.key.A]}, D: {self.keys[pyglet.window.key.D]}, W: {self.keys[pyglet.window.key.W]}, S: {self.keys[pyglet.window.key.S]}")

        if self.keys[pyglet.window.key.LEFT] or self.keys[pyglet.window.key.A]:
            dx -= 1.0
            # print("LEFT/A pressed, dx now:", dx) # DEBUG - REMOVED
        if self.keys[pyglet.window.key.RIGHT] or self.keys[pyglet.window.key.D]:
            dx += 1.0
            # print("RIGHT/D pressed, dx now:", dx) # DEBUG - REMOVED
        if self.keys[pyglet.window.key.UP] or self.keys[pyglet.window.key.W]:
            dy += 1.0
            # print("UP/W pressed, dy now:", dy) # DEBUG - REMOVED
        if self.keys[pyglet.window.key.DOWN] or self.keys[pyglet.window.key.S]:
            dy -= 1.0
            # print("DOWN/S pressed, dy now:", dy) # DEBUG - REMOVED

        if dx != 0.0 or dy != 0.0:
            # print(f"Panning initiated with dx: {dx}, dy: {dy}") # DEBUG - REMOVED
            self.camera.pan(dx, dy, dt)
        # else: # DEBUG for when no keys are pressed - REMOVED
            # print("No pan movement detected (dx and dy are 0).") 

    def handle_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Handles mouse scroll events for camera zooming."""
        if scroll_y > 0:
            self.camera.zoom_in()
        elif scroll_y < 0:
            self.camera.zoom_out()

class MetabolismSystem(System):
    """Handles agent aging, energy consumption, and death."""
    def __init__(self, map_generator_ref: Any, tile_render_size: int) -> None: # Using Any for MapGenerator to avoid circular import if MapGenerator needs components
        super().__init__()
        self.map_generator = map_generator_ref
        self.tile_render_size = tile_render_size
        self.metabolic_rate_hunger = 0.5  # Hunger points per second
        self.metabolic_rate_sleep = 0.3 # Sleepiness points per second
        self.max_sleepiness = 100.0

    def process(self, dt: float) -> None:
        entities_to_remove = []
        
        # DEBUG: Log entities being processed by MetabolismSystem
        all_agents_for_metabolism = list(esper.get_components(AgentMarker, Position, Energy, Age))
        print(f"[DEBUG MetabolismSystem] Processing {len(all_agents_for_metabolism)} agents. dt={dt:.6f}")

        for ent, (agent, pos, energy, age) in all_agents_for_metabolism:
            # DEBUG: Log individual agent stats before update
            # print(f"[DEBUG MetabolismSystem] Before Ent {ent}: Hunger={energy.hunger:.2f}, Sleep={energy.sleepiness:.2f}, Age={age.current_age:.2f}/{age.lifespan:.2f}")

            age.current_age += dt
            energy.hunger -= self.metabolic_rate_hunger * dt
            energy.sleepiness += self.metabolic_rate_sleep * dt
            energy.sleepiness = min(energy.sleepiness, self.max_sleepiness) # Cap sleepiness

            # DEBUG: Log individual agent stats after update
            # print(f"[DEBUG MetabolismSystem] After Ent {ent}: Hunger={energy.hunger:.2f}, Sleep={energy.sleepiness:.2f}, Age={age.current_age:.2f}")

            dead = False
            death_reason = ""

            if energy.hunger <= 0:
                dead = True
                death_reason = f"starvation (hunger: {energy.hunger:.2f})"
            elif energy.sleepiness >= self.max_sleepiness:
                dead = True
                death_reason = f"exhaustion (sleepiness: {energy.sleepiness:.2f})"
            elif age.current_age >= age.lifespan:
                dead = True
                death_reason = f"old age (age: {age.current_age:.2f}, lifespan: {age.lifespan:.2f})"
            
            if dead:
                print(f"[DEBUG MetabolismSystem] Zorp {ent} marked dead! Reason: {death_reason} at ({pos.x:.0f}, {pos.y:.0f}) age {age.current_age:.4f}.")
                # Original print for continuity, can be removed if too verbose with new debug
                # print(f"Zorp {ent} died of {death_reason} at ({pos.x:.0f}, {pos.y:.0f}) at age {age.current_age:.2f}.")
                entities_to_remove.append(ent)

                # Corpse Drop: Change tile to ORGANIC_REMAINS
                # Convert world coordinates to tile coordinates
                tile_x = int(pos.x / self.tile_render_size)
                tile_y = int(pos.y / self.tile_render_size)

                if self.map_generator.tile_grid_np is not None and \
                   0 <= tile_y < self.map_generator.height and \
                   0 <= tile_x < self.map_generator.width:
                    
                    # Decrement counts for the tile being replaced
                    original_tile_at_corpse = self.map_generator.tile_grid_np[tile_y][tile_x]
                    original_resource_at_corpse = original_tile_at_corpse.metadata.resource_type
                    
                    if original_tile_at_corpse in self.map_generator.biome_counts:
                        self.map_generator.biome_counts[original_tile_at_corpse] -= 1
                        if self.map_generator.biome_counts[original_tile_at_corpse] == 0:
                            del self.map_generator.biome_counts[original_tile_at_corpse]
                    
                    if original_resource_at_corpse != ResourceType.NONE and \
                       original_resource_at_corpse in self.map_generator.resource_counts:
                        self.map_generator.resource_counts[original_resource_at_corpse] -= 1
                        if self.map_generator.resource_counts[original_resource_at_corpse] == 0:
                            del self.map_generator.resource_counts[original_resource_at_corpse]

                    # Set new tile and update counts
                    self.map_generator.tile_grid_np[tile_y][tile_x] = Tile.ORGANIC_REMAINS
                    self.map_generator.biome_counts[Tile.ORGANIC_REMAINS] = self.map_generator.biome_counts.get(Tile.ORGANIC_REMAINS, 0) + 1
                    self.map_generator.resource_counts[ResourceType.ORGANIC_MATTER] = self.map_generator.resource_counts.get(ResourceType.ORGANIC_MATTER, 0) + 1
                    print(f"Tile ({tile_x},{tile_y}) changed to ORGANIC_REMAINS.")
                else:
                    print(f"Warning: Corpse position ({tile_x},{tile_y}) is out of map bounds.")

        # DEBUG: Log entities about to be removed
        if entities_to_remove:
            print(f"[DEBUG MetabolismSystem] Entities to remove: {entities_to_remove}")
        # else:
            # print(f"[DEBUG MetabolismSystem] No entities to remove this tick.")

        for ent in entities_to_remove:
            esper.delete_entity(ent, immediate=True)
            print(f"[DEBUG MetabolismSystem] Entity {ent} deleted.")

class ReproductionSystem(System):
    """Handles Zorp reproduction."""
    def __init__(self, tile_render_size: int) -> None:
        super().__init__()
        self.tile_render_size = tile_render_size # To convert world pos to tile pos if needed for proximity
        self.reproduction_energy_threshold = 70.0 # Hunger must be > this
        self.min_reproduction_age = 15.0 # Min age in game time units
        self.max_reproduction_age = 80.0 # Max age
        self.reproduction_cooldown = 10.0 # Game time units before an agent can reproduce again (per agent)
        self.partner_search_radius = 2.0 * tile_render_size # World units for finding a partner
        self.last_reproduction_time: Dict[int, float] = {} # ent_id: game_time
        self.current_game_time = 0.0 # Needs to be updated by the engine or a time system

    def process(self, dt: float) -> None:
        self.current_game_time += dt # Simple way to track game time for cooldowns
        
        potential_parents = []
        for ent, (agent, pos, energy, age, genetics) in esper.get_components(AgentMarker, Position, Energy, Age, Genetics):
            can_reproduce_now = (
                energy.hunger > self.reproduction_energy_threshold and
                self.min_reproduction_age <= age.current_age <= self.max_reproduction_age and
                (ent not in self.last_reproduction_time or 
                 self.current_game_time - self.last_reproduction_time[ent] > self.reproduction_cooldown)
            )
            if can_reproduce_now:
                potential_parents.append({'ent': ent, 'pos': pos, 'genetics': genetics, 'age': age.current_age})

        # Naive O(N^2) partner search - can be optimized later with spatial hashing if needed
        # Also, ensure each pair reproduces only once per frame cycle if they both find each other.
        reproduced_this_frame = set()

        for i in range(len(potential_parents)):
            parent1_data = potential_parents[i]
            if parent1_data['ent'] in reproduced_this_frame:
                continue

            for j in range(i + 1, len(potential_parents)):
                parent2_data = potential_parents[j]
                if parent2_data['ent'] in reproduced_this_frame:
                    continue

                # Check distance
                p1 = np.array([parent1_data['pos'].x, parent1_data['pos'].y])
                p2 = np.array([parent2_data['pos'].x, parent2_data['pos'].y])
                distance = np.linalg.norm(p1 - p2)

                if distance < self.partner_search_radius:
                    # Spawn child
                    child_x = (parent1_data['pos'].x + parent2_data['pos'].x) / 2 + random.uniform(-5, 5)
                    child_y = (parent1_data['pos'].y + parent2_data['pos'].y) / 2 + random.uniform(-5, 5)
                    
                    # Simple genetics: new ID, could inherit/mutate other traits later
                    child_genetics = Genetics() # Gets a new UUID
                    # Example mutation/inheritance (placeholder)
                    # if random.random() < 0.5: child_genetics.some_trait = parent1_data['genetics'].some_trait
                    # else: child_genetics.some_trait = parent2_data['genetics'].some_trait
                    # if random.random() < 0.1: child_genetics.some_trait = mutate(child_genetics.some_trait)

                    child_entity = esper.create_entity(
                        AgentMarker(),
                        Position(x=child_x, y=child_y),
                        Energy(hunger=100.0, sleepiness=0.0), # Born full and rested
                        Age(current_age=0.0, lifespan=random.uniform(80, 120)), # Lifespan can vary slightly
                        child_genetics
                    )
                    print(f"Zorps {parent1_data['ent']} and {parent2_data['ent']} reproduced! New Zorp: {child_entity} with ID {child_genetics.genetic_id}")
                    
                    self.last_reproduction_time[parent1_data['ent']] = self.current_game_time
                    self.last_reproduction_time[parent2_data['ent']] = self.current_game_time
                    reproduced_this_frame.add(parent1_data['ent'])
                    reproduced_this_frame.add(parent2_data['ent'])
                    break # parent1 found a partner

class RenderSystem(System):
    def __init__(
        self,
        window: pyglet.window.Window,
        camera: 'Camera', 
        batch: pyglet.graphics.Batch, 
        tile_render_size: int,
        world_ref: Optional['ZorpWorld'] = None # Optional ZorpWorld reference
    ) -> None:
        super().__init__()
        self.window = window
        self.camera = camera
        self.batch = batch # Use the shared batch from Engine
        self.tile_sprites: List[pyglet.shapes.ShapeBase] = [] # Store all tile sprites for potential clearing
        self.zorp_visuals: Dict[str, pyglet.shapes.ShapeBase] = {}
        self.map_data: Optional[List[List[Tile]]] = None # Will be set by the Engine after map generation
        self.tile_render_size = tile_render_size
        self.tile_group = pyglet.graphics.Group(order=0)  # For tiles (background)
        self.agent_group = pyglet.graphics.Group(order=1) # For agents (foreground)
        self.debug_group = pyglet.graphics.Group(order=2) # For debug text
        self.world_ref: Optional['ZorpWorld'] = world_ref

        self.fps_label = pyglet.text.Label(
            'FPS: 0', 
            font_name='Arial', font_size=12, 
            x=10, y=self.window.height - 20, anchor_x='left', anchor_y='top',
            batch=None # Drawn separately, not part of the batch
        )
        self._prepare_map_renderables_call_count = 0 # DEBUG COUNTER

    def _prepare_map_renderables(self) -> None:
        self._prepare_map_renderables_call_count += 1 # DEBUG COUNTER
        print(f"[DEBUG RenderSystem] _prepare_map_renderables call count: {self._prepare_map_renderables_call_count}") # DEBUG COUNTER

        if self.map_data is None:
            print("[RenderSystem] Map data not available for rendering.")
            return
        
        # Clear existing tile sprites before redrawing
        for sprite in self.tile_sprites:
            sprite.delete()
        self.tile_sprites.clear()
        
        print(f"[RenderSystem] Preparing map renderables. Tile size: {self.tile_render_size}")
        rows, cols = self.map_data.shape
        for r in range(rows):
            for c in range(cols):
                tile_enum = self.map_data[r, c]
                try:
                    meta = tile_enum.metadata
                    # Define in WORLD coordinates
                    base_x = c * self.tile_render_size 
                    base_y = r * self.tile_render_size

                    square = pyglet.shapes.Rectangle(
                        x=base_x, 
                        y=base_y,
                        width=self.tile_render_size,
                        height=self.tile_render_size,
                        color=meta.base_color[:3],
                        batch=self.batch,
                        group=self.tile_group
                    )
                    self.tile_sprites.append(square)

                    if meta.spot_color and meta.spot_count > 0:
                        # Seed for deterministic spot placement for THIS tile.
                        # Using a tuple of coordinates and tile type ensures that if the tile type changes,
                        # the spots will also change, which is desirable.
                        random.seed(hash((c, r, tile_enum.value)))

                        spot_world_size = self.tile_render_size * meta.spot_size_ratio
                        spot_world_size = max(1, min(spot_world_size, self.tile_render_size / 2))
                        
                        for _ in range(meta.spot_count):
                            margin = spot_world_size / 2 # World space margin
                            spot_center_x_world = base_x + margin + random.uniform(0, self.tile_render_size - spot_world_size)
                            spot_center_y_world = base_y + margin + random.uniform(0, self.tile_render_size - spot_world_size)
                            
                            spot_shape = pyglet.shapes.Circle(
                                x=spot_center_x_world, # WORLD coordinates for center
                                y=spot_center_y_world,
                                radius=spot_world_size / 2, # WORLD radius
                                color=meta.spot_color[:3],
                                batch=self.batch,
                                group=self.tile_group
                            )
                            self.tile_sprites.append(spot_shape)
                except Exception as e:
                    print(f"Error preparing sprite for tile {tile_enum} at ({r},{c}): {e}")
        print(f"Map renderables prepared. {len(self.tile_sprites)} tile sprites created.")

    def _get_zorp_color(self, energy: float, max_energy: float = 100.0) -> Tuple[int, int, int]:
        energy_ratio = max(0, min(1, energy / max_energy))
        if energy_ratio > 0.7:
            return (int(100 * (1 - energy_ratio) * 2.5), 200, int(50 * (1 - energy_ratio) * 2.5))
        elif energy_ratio > 0.3:
            return (255, int(150 + 105 * (energy_ratio - 0.3) / 0.4), 0)
        else:
            return (255, int(100 * energy_ratio / 0.3), 0)

    def process_main_render_loop(self, dt: float) -> None:
        self.window.clear()

        # Set the window's view transform for the world batch
        self.window.view = self.camera.get_view_matrix()

        # Check if map visuals need updating
        if self.world_ref and self.world_ref.map_visuals_dirty:
            print("[RenderSystem] map_visuals_dirty is True. Re-preparing map renderables.")
            self._prepare_map_renderables()
            self.world_ref.map_visuals_dirty = False # Reset the flag

        current_zorp_ids_in_world = set()
        for zorp in self.world_ref.all_zorps:
            current_zorp_ids_in_world.add(zorp.id)
            if not zorp.alive:
                if zorp.id in self.zorp_visuals:
                    self.zorp_visuals[zorp.id].delete()
                    del self.zorp_visuals[zorp.id]
                    if zorp.id + '_debug' in self.zorp_visuals:
                        self.zorp_visuals[zorp.id + '_debug'].delete()
                        del self.zorp_visuals[zorp.id + '_debug']
                continue

            # Zorp positions and sizes are in WORLD coordinates
            tile_x, tile_y = zorp.position
            world_x_center = tile_x * self.tile_render_size + self.tile_render_size / 2.0
            world_y_center = tile_y * self.tile_render_size + self.tile_render_size / 2.0
            world_radius = self.tile_render_size / 3.0 
            color = self._get_zorp_color(zorp.energy, zorp.max_energy)

            if zorp.id in self.zorp_visuals:
                shape = self.zorp_visuals[zorp.id]
                shape.x = world_x_center
                shape.y = world_y_center
                shape.radius = world_radius 
                shape.color = color
                shape.opacity = 255
                # Debug shape also in world coordinates
                if zorp.id + '_debug' in self.zorp_visuals:
                    debug_shape = self.zorp_visuals[zorp.id + '_debug']
                    debug_shape.x = world_x_center - self.tile_render_size / 4 # Offset from center
                    debug_shape.y = world_y_center - self.tile_render_size / 4
                    # Assuming debug shape width/height are fixed in world units
                    debug_shape.width = self.tile_render_size / 2 
                    debug_shape.height = self.tile_render_size / 2
            else:
                new_shape = pyglet.shapes.Circle(
                    world_x_center, world_y_center, world_radius, # WORLD coordinates
                    color=color,
                    batch=self.batch,
                    group=self.agent_group
                )
                self.zorp_visuals[zorp.id] = new_shape
                # Optionally create debug rect in world coordinates if needed
                # debug_rect = pyglet.shapes.Rectangle(
                #     x = world_x_center - self.tile_render_size / 4, 
                #     y = world_y_center - self.tile_render_size / 4, 
                #     width = self.tile_render_size / 2, 
                #     height = self.tile_render_size / 2, 
                #     color=(255,0,255), batch=self.batch, group=self.agent_group)
                # self.zorp_visuals[zorp.id + '_debug'] = debug_rect

        zorp_ids_in_visuals = list(self.zorp_visuals.keys())
        for zorp_id_visual in zorp_ids_in_visuals:
            is_debug_visual = zorp_id_visual.endswith('_debug')
            actual_zorp_id = zorp_id_visual.replace('_debug', '')
            if actual_zorp_id not in current_zorp_ids_in_world:
                if self.zorp_visuals[zorp_id_visual]:
                    self.zorp_visuals[zorp_id_visual].delete()
                    del self.zorp_visuals[zorp_id_visual]

        self.batch.draw() # This draws everything affected by self.window.view

        # Reset view for UI elements (like FPS label) to draw in screen coordinates
        self.window.view = math.Mat4() # Identity matrix
        if self.fps_label: # Ensure fps_label is drawn after resetting the view
            self.fps_label.draw()

    def process(self, dt: float) -> None:
        self.process_main_render_loop(dt)

    def update_fps_display(self, fps: float) -> None:
        """Update the FPS label text.

        Args:
            fps: The current frames per second value.
        """
        self.fps_label.text = f"FPS: {fps:.1f}"

# --- Main Game Class (Example of how systems might be wired) --- #
# This is a conceptual placeholder. Actual game class might be elsewhere.
# ... existing code ... 