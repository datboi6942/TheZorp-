import esper
import pyglet
from pyglet.graphics import Group # Changed from OrderedGroup
from typing import Any, TYPE_CHECKING, Optional, List, Dict, Tuple
import numpy as np # For type hint np.ndarray
from zorplife.ui.camera import Camera # Import Camera
from zorplife.ui.sprites import SpriteSheet # Import SpriteSheet
from zorplife.world.tiles import Tile, ResourceType # For type hint & ORGANIC_MATTER
from zorplife.agents.components import Position, Energy, Age, Genetics, AgentMarker, Inventory # Agent components
import random
from pyglet import math # Import pyglet.math for Mat4 identity
from zorplife.world.world import REGROW_TICKS

# if TYPE_CHECKING:
#     # No longer needed as esper.World object is not used.
#     pass

if TYPE_CHECKING:
    from ..world.world import ZorpWorld # For type hinting world_ref
    from ..core.zorp import Zorp # For type hinting Zorp entities
    from ..world.mapgen import MapGenerator # For type hinting map_generator

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

            # Debug print for camera state
            # print(f"Camera Panned to: ({self.camera.offset_x:.2f}, {self.camera.offset_y:.2f}), Zoom: {self.camera.zoom:.2f}, DeltaTime: {dt:.4f}") # Commented out to reduce log spam
        # else: # DEBUG for when no keys are pressed - REMOVED
            # print("No pan movement detected (dx and dy are 0).") 

    def handle_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Handles mouse scroll events for camera zooming."""
        if scroll_y > 0:
            self.camera.zoom_in()
        elif scroll_y < 0:
            self.camera.zoom_out()

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if not self.camera: return

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
                        child_genetics,
                        Inventory() # Add empty inventory to new Zorps
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
        self.tile_visuals: Dict[Tuple[int, int], List[pyglet.shapes.ShapeBase]] = {} # Stores sprites per (c,r) coordinate
        self.rendered_map_state: Optional[np.ndarray] = None # Stores the state of map_data as it was last rendered
        self.zorp_visuals: Dict[str, pyglet.shapes.ShapeBase] = {} # Renamed from zorp_sprites
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
        self.zorp_count_label = pyglet.text.Label( # New Zorp counter label
            'Zorps: 0',
            font_name='Arial', font_size=12,
            x=10, y=self.window.height - 40, anchor_x='left', anchor_y='top', # Position below FPS
            batch=None # Drawn separately
        )
        self._prepare_map_renderables_call_count = 0 # DEBUG COUNTER

    def _create_tile_sprites_at_coord(self, r: int, c: int, tile_enum: Tile) -> List[pyglet.shapes.ShapeBase]:
        """Creates and returns all pyglet shapes for a single tile at given map coordinates (r, c)."""
        created_sprites: List[pyglet.shapes.ShapeBase] = []
        try:
            meta = tile_enum.metadata
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
            created_sprites.append(square)

            if meta.spot_color and meta.spot_count > 0:
                random.seed(hash((c, r, tile_enum.value))) # Deterministic spots
                spot_world_size = self.tile_render_size * meta.spot_size_ratio
                spot_world_size = max(1, min(spot_world_size, self.tile_render_size / 2))

                for _ in range(meta.spot_count):
                    margin = spot_world_size / 2
                    spot_center_x_world = base_x + margin + random.uniform(0, self.tile_render_size - spot_world_size)
                    spot_center_y_world = base_y + margin + random.uniform(0, self.tile_render_size - spot_world_size)

                    spot_shape = pyglet.shapes.Circle(
                        x=spot_center_x_world,
                        y=spot_center_y_world,
                        radius=spot_world_size / 2,
                        color=meta.spot_color[:3],
                        batch=self.batch,
                        group=self.tile_group
                    )
                    created_sprites.append(spot_shape)
        except Exception as e:
            print(f"Error creating sprite for tile {tile_enum} at map ({r},{c}) world ({base_x},{base_y}): {e}")
        return created_sprites

    def _prepare_map_renderables(self) -> None: # Initial full map draw
        self._prepare_map_renderables_call_count += 1
        print(f"[DEBUG RenderSystem] _prepare_map_renderables (FULL DRAW) call count: {self._prepare_map_renderables_call_count}")

        if not self.world_ref or self.world_ref.map_data is None:
            print("[RenderSystem _prepare_map_renderables] World reference or map data not available.")
            return

        # Clear any existing visuals (e.g., if called again after a reset)
        for coord_sprites in self.tile_visuals.values():
            for sprite in coord_sprites:
                sprite.delete()
        self.tile_visuals.clear()
        
        print(f"[RenderSystem _prepare_map_renderables] Preparing initial map renderables. Tile size: {self.tile_render_size}")
        current_map_data = self.world_ref.map_data
        rows, cols = current_map_data.shape
        
        total_sprites_created = 0
        for r_idx in range(rows): # Renamed r to r_idx
            for c_idx in range(cols): # Renamed c to c_idx
                tile_enum = current_map_data[r_idx, c_idx]
                new_sprites = self._create_tile_sprites_at_coord(r_idx, c_idx, tile_enum)
                self.tile_visuals[(c_idx, r_idx)] = new_sprites
                total_sprites_created += len(new_sprites)
        
        self.rendered_map_state = np.copy(current_map_data) # Store a copy of the rendered state
        print(f"Initial map renderables prepared. {total_sprites_created} tile sprites created for {rows*cols} tiles.")

    def _update_dirty_map_renderables(self) -> None: # Partial map update based on diff
        print(f"[DEBUG RenderSystem] _update_dirty_map_renderables called.")
        if not self.world_ref or self.world_ref.map_data is None or self.rendered_map_state is None:
            print("[RenderSystem _update_dirty_map_renderables] World ref, map data, or rendered state missing.")
            return

        current_map_data = self.world_ref.map_data
        rows, cols = current_map_data.shape
        changed_tiles_count = 0
        
        for r_idx in range(rows): # Renamed r to r_idx
            for c_idx in range(cols): # Renamed c to c_idx
                current_tile_enum = current_map_data[r_idx, c_idx]
                rendered_tile_enum = self.rendered_map_state[r_idx, c_idx]

                if current_tile_enum != rendered_tile_enum:
                    changed_tiles_count +=1
                    # Delete old sprites for this coordinate
                    if (c_idx, r_idx) in self.tile_visuals:
                        for old_sprite in self.tile_visuals[(c_idx, r_idx)]:
                            old_sprite.delete()
                        self.tile_visuals[(c_idx, r_idx)].clear()
                    
                    # Create and store new sprites
                    new_sprites = self._create_tile_sprites_at_coord(r_idx, c_idx, current_tile_enum)
                    self.tile_visuals[(c_idx, r_idx)] = new_sprites
                    
                    # Update the rendered state for this tile
                    self.rendered_map_state[r_idx, c_idx] = current_tile_enum
        
        if changed_tiles_count > 0:
            print(f"[RenderSystem _update_dirty_map_renderables] Updated {changed_tiles_count} changed tiles.")
        
        self.world_ref.map_visuals_dirty = False # Reset the flag

    def process_main_render_loop(self, dt: float) -> None:
        self.window.clear()
        self.window.view = self.camera.get_view_matrix()

        if self.rendered_map_state is None: # First-time map draw
            if self.world_ref and self.world_ref.map_data is not None:
                print("[RenderSystem process_main_render_loop] Initial map draw.")
                self._prepare_map_renderables()
            else:
                print("[RenderSystem process_main_render_loop] Waiting for world_ref and map_data for initial draw.")
        elif self.world_ref and self.world_ref.map_visuals_dirty:
            print("[RenderSystem process_main_render_loop] map_visuals_dirty is True. Updating dirty map renderables.")
            self._update_dirty_map_renderables()
        # Check if map visuals need updating - OLD LOGIC REMOVED
        # if self.world_ref and self.world_ref.map_visuals_dirty:
        #     print("[RenderSystem] map_visuals_dirty is True. Re-preparing map renderables.")
        #     self._prepare_map_renderables() # This was the full redraw
        #     self.world_ref.map_visuals_dirty = False # Reset the flag

        current_zorp_ids_in_world = set()
        for zorp in self.world_ref.all_zorps:
            zorp_id_str = str(zorp.id)
            current_zorp_ids_in_world.add(zorp_id_str)
            if not zorp.alive:
                if zorp_id_str in self.zorp_visuals:
                    self.zorp_visuals[zorp_id_str].delete()
                    del self.zorp_visuals[zorp_id_str]
                    if zorp_id_str + '_debug' in self.zorp_visuals:
                        self.zorp_visuals[zorp_id_str + '_debug'].delete()
                        del self.zorp_visuals[zorp_id_str + '_debug']
                continue

            tile_x, tile_y = zorp.position
            world_x_center = tile_x * self.tile_render_size + self.tile_render_size / 2.0
            world_y_center = tile_y * self.tile_render_size + self.tile_render_size / 2.0
            world_radius = self.tile_render_size / 3.0 
            color = self._get_zorp_color(zorp.energy, zorp.max_energy)

            # DEBUG: Print Zorp and visual positions
            # print(f"[RenderSystem] Zorp {zorp_id_str} at tile ({tile_x},{tile_y}) -> world ({world_x_center:.1f},{world_y_center:.1f})")

            if zorp_id_str in self.zorp_visuals:
                shape = self.zorp_visuals[zorp_id_str]
                shape.x = world_x_center
                shape.y = world_y_center
                shape.radius = world_radius 
                shape.color = color
                shape.opacity = 255
                if zorp_id_str + '_debug' in self.zorp_visuals:
                    debug_shape = self.zorp_visuals[zorp_id_str + '_debug']
                    debug_shape.x = world_x_center - self.tile_render_size / 4
                    debug_shape.y = world_y_center - self.tile_render_size / 4
                    debug_shape.width = self.tile_render_size / 2 
                    debug_shape.height = self.tile_render_size / 2
            else:
                new_shape = pyglet.shapes.Circle(
                    world_x_center, world_y_center, world_radius,
                    color=color,
                    batch=self.batch,
                    group=self.agent_group
                )
                self.zorp_visuals[zorp_id_str] = new_shape
                # Optionally create debug rect in world coordinates if needed
                # debug_rect = pyglet.shapes.Rectangle(
                #     x = world_x_center - self.tile_render_size / 4, 
                #     y = world_y_center - self.tile_render_size / 4, 
                #     width = self.tile_render_size / 2, 
                #     height = self.tile_render_size / 2, 
                #     color=(255,0,255), batch=self.batch, group=self.agent_group)
                # self.zorp_visuals[zorp_id_str + '_debug'] = debug_rect

        zorp_ids_in_visuals = list(self.zorp_visuals.keys())
        for zorp_id_visual in zorp_ids_in_visuals:
            if not isinstance(zorp_id_visual, str):
                continue
            is_debug_visual = zorp_id_visual.endswith('_debug')
            actual_zorp_id = zorp_id_visual.replace('_debug', '')
            if actual_zorp_id not in current_zorp_ids_in_world:
                if self.zorp_visuals[zorp_id_visual]:
                    self.zorp_visuals[zorp_id_visual].delete()
                    del self.zorp_visuals[zorp_id_visual]
        
        if self.world_ref: # Update zorp count
            self.update_zorp_count_display(len(current_zorp_ids_in_world))

        self.batch.draw() # This draws everything affected by self.window.view

        # Reset view for UI elements (like FPS label) to draw in screen coordinates
        self.window.view = math.Mat4() # Identity matrix
        if self.fps_label: # Ensure fps_label is drawn after resetting the view
            self.fps_label.draw()
        if self.zorp_count_label: # Ensure zorp_count_label is drawn
            self.zorp_count_label.draw()

    def process(self, dt: float) -> None:
        self.process_main_render_loop(dt)

    def update_fps_display(self, fps: float) -> None:
        """Update the FPS label text.

        Args:
            fps: The current frames per second value.
        """
        self.fps_label.text = f"FPS: {fps:.1f}"

    def update_zorp_count_display(self, count: int) -> None:
        """Update the Zorp count label text.

        Args:
            count: The current number of Zorps.
        """
        if self.zorp_count_label:
            self.zorp_count_label.text = f"Zorps: {count}"

    def _get_zorp_color(self, energy: 'Energy', max_energy: float) -> Tuple[int, int, int]:
        """Calculates Zorp color based on energy. Healthy = green, Starving = red."""
        if max_energy == 0: # Avoid division by zero
            return (128, 128, 128) # Grey for undefined max_energy state
        
        # Correctly access the hunger attribute from the Energy object
        energy_ratio = max(0, min(1, energy.hunger / max_energy))
        
        # Interpolate between red (low energy) and green (high energy)
        red = int(255 * (1 - energy_ratio))
        green = int(255 * energy_ratio)
        blue = 0
        return (red, green, blue)

class ForagingSystem(System):
    """Handles Zorp foraging behavior, including finding and consuming food."""

    def __init__(self, world_ref: Optional['ZorpWorld'] = None, map_generator_ref: Optional['MapGenerator'] = None):
        super().__init__()
        self.world_ref = world_ref
        self.map_generator_ref = map_generator_ref
        self.zorp_inventories: Dict[str, Inventory] = {} # zorp_id (UUID string) -> Inventory

    def add_new_zorp_inventory(self, zorp_id: str) -> None:
        """Initializes an inventory for a newly created Zorp.

        Args:
            zorp_id: The unique identifier (UUID string) of the Zorp.
        """
        if zorp_id not in self.zorp_inventories:
            self.zorp_inventories[zorp_id] = Inventory()
            # print(f"[ForagingSystem DEBUG] Initialized inventory for new Zorp ID: {zorp_id}") # DEBUG
        # else: # DEBUG
            # print(f"[ForagingSystem WARNING] Zorp ID {zorp_id} already has an inventory. Not re-initializing.") # DEBUG

    def process(self, dt: float) -> None:
        # Check if essential references are set
        if not self.world_ref or self.world_ref.map_data is None or not self.map_generator_ref:
            # print("[ForagingSystem] World reference, map_data, or map_generator not available. Skipping process.")
            return

        # Iterate over all Zorps that are currently in the "foraging" state
        for zorp in self.world_ref.all_zorps:
            if not zorp.alive or zorp.current_action != "foraging":
                continue

            # Check if inventory is already full (or carrying something)
            if zorp.inventory.carrying is not None and zorp.inventory.amount > 0:
                # Already carrying something, shouldn't be in "foraging" state or should re-evaluate
                # For now, we assume decide_action correctly sets foraging only if inventory is empty
                # or we can clear action here if Zorp is full but still foraging.
                # zorp.current_action = None # Or back to wandering
                continue

            tile_x, tile_y = zorp.position
            
            # Boundary check for tile coordinates
            if not (0 <= tile_y < self.map_generator_ref.height and 0 <= tile_x < self.map_generator_ref.width):
                print(f"[ForagingSystem] Zorp {zorp.id} at invalid position ({tile_x},{tile_y}). Skipping.")
                continue
            
            tile_under_zorp = self.world_ref.map_data[tile_y, tile_x]
            resource_on_tile = tile_under_zorp.metadata.resource_type
            energy_from_tile = tile_under_zorp.metadata.energy_value # May not be directly used for pickup, but good to know

            # Check if the resource is one the Zorp wants (or any food)
            # For simplicity, let's assume Zorp attempts to pick up any recognized food resource.
            # The Zorp's genome (`preferred_food_types`) is used when *eating* from inventory.
            # Here, we check if the tile itself is a harvestable resource.
            
            # What tile does it become after harvesting? (e.g. FRUIT_BUSH -> GRASS)
            # This needs a defined mapping or logic.
            # For now, let's assume some common transitions:
            # FRUIT_BUSH -> GRASS
            # MUSHROOM_PATCH -> DIRT or GRASS
            # ORGANIC_REMAINS -> DIRT or GRASS
            # Other resources like IRON_DEPOSIT might not be "picked up" in the same way by Zorps for food.

            harvestable_food_resources = {
                ResourceType.APPLES: Tile.GRASSLAND, # Changed FRUIT to APPLES. When apples are picked from APPLE_TREE_LEAVES, the leaves tile might become just GRASSLAND, or perhaps APPLE_TREE_LEAVES with no apples (needs more logic for regrowth).
                ResourceType.MUSHROOMS: Tile.FOREST_FLOOR,
                ResourceType.ORGANIC_MATTER: Tile.FOREST_FLOOR # From ORGANIC_REMAINS tile
            }

            if resource_on_tile in harvestable_food_resources:
                if zorp.inventory.carrying is None and zorp.inventory.amount == 0:
                    # Pick up the resource
                    zorp.inventory.carrying = resource_on_tile
                    zorp.inventory.amount = 1 # Assume 1 unit picked up
                    
                    original_tile_enum = tile_under_zorp
                    new_tile_enum: Tile
                    regrow_ticks = REGROW_TICKS

                    if resource_on_tile == ResourceType.APPLES:
                        if original_tile_enum == Tile.APPLE_TREE_LEAVES: # Ensure we are picking from the correct tile
                            self.world_ref.consumed_tile_original_type[(tile_x, tile_y)] = Tile.APPLE_TREE_LEAVES
                            new_tile_enum = Tile.TREE_CANOPY
                        else: # Zorp is on a tile that isn't APPLE_TREE_LEAVES but claims to have apples, skip
                            zorp.inventory.carrying = None # Undo pickup
                            zorp.inventory.amount = 0
                            continue
                    elif resource_on_tile == ResourceType.MUSHROOMS:
                        if original_tile_enum == Tile.MUSHROOM_PATCH: # Ensure we are picking from the correct tile
                            self.world_ref.consumed_tile_original_type[(tile_x, tile_y)] = Tile.MUSHROOM_PATCH
                            new_tile_enum = Tile.FOREST_FLOOR
                        else: # Zorp is on a tile that isn't MUSHROOM_PATCH but claims to have mushrooms, skip
                            zorp.inventory.carrying = None # Undo pickup
                            zorp.inventory.amount = 0
                            continue
                    elif resource_on_tile == ResourceType.ORGANIC_MATTER:
                        # Assuming ORGANIC_MATTER comes from ORGANIC_REMAINS and regrows as FOREST_FLOOR
                        # This might not need special consumed_tile_original_type handling if it doesn't revert to ORGANIC_REMAINS
                        self.world_ref.consumed_tile_original_type[(tile_x, tile_y)] = Tile.ORGANIC_REMAINS # Or original_tile_enum
                        new_tile_enum = Tile.FOREST_FLOOR # Or whatever ORGANIC_REMAINS should become when depleted
                    else:
                        # Should not happen if resource_on_tile is in harvestable_food_resources
                        # but as a fallback, or if other resources are added:
                        zorp.inventory.carrying = None # Undo pickup
                        zorp.inventory.amount = 0
                        continue

                    # Update map data in ZorpWorld (which should handle map_generator's grid too, or do it consistently)
                    self.world_ref.map_data[tile_y, tile_x] = new_tile_enum # Change tile in ZorpWorld's map_data
                    # self.map_generator_ref.tile_grid_np[tile_y, tile_x] = new_tile_enum # This might be redundant if ZorpWorld.map_data is the source of truth

                    # Deplete resource and set regrowth timer in ZorpWorld
                    self.world_ref.resource_energy_map[tile_y, tile_x] = 0.0
                    self.world_ref.resource_regrow_timer[tile_y, tile_x] = regrow_ticks
                    
                    self.world_ref.map_visuals_dirty = True

                    # Update biome and resource counts (critical)
                    # Decrement old tile/resource
                    if original_tile_enum in self.map_generator_ref.biome_counts:
                        self.map_generator_ref.biome_counts[original_tile_enum] -= 1
                        if self.map_generator_ref.biome_counts[original_tile_enum] == 0:
                            del self.map_generator_ref.biome_counts[original_tile_enum]
                    
                    if resource_on_tile != ResourceType.NONE and resource_on_tile in self.map_generator_ref.resource_counts:
                        self.map_generator_ref.resource_counts[resource_on_tile] -= 1
                        if self.map_generator_ref.resource_counts[resource_on_tile] == 0:
                            del self.map_generator_ref.resource_counts[resource_on_tile]

                    # Increment new tile/resource (new tile might have ResourceType.NONE)
                    self.map_generator_ref.biome_counts[new_tile_enum] = self.map_generator_ref.biome_counts.get(new_tile_enum, 0) + 1
                    new_resource_type = new_tile_enum.metadata.resource_type
                    if new_resource_type != ResourceType.NONE:
                         self.map_generator_ref.resource_counts[new_resource_type] = self.map_generator_ref.resource_counts.get(new_resource_type, 0) + 1

                    zorp.current_action = None # Successfully foraged, reset action (will be decided next tick)
                    print(f"[ForagingSystem] Zorp {zorp.id} picked up {resource_on_tile.name} from ({tile_x},{tile_y}). Tile changed to {new_tile_enum.name}. Inventory: {zorp.inventory.carrying.name}({zorp.inventory.amount})")
                # else: Zorp already carrying something or inventory full, though decide_action should prevent this state.

# --- Main Game Class (Example of how systems might be wired) --- #
# This is a conceptual placeholder. Actual game class might be elsewhere.
# ... existing code ... 