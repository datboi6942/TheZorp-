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
        camera: Camera, 
        map_data: Optional[np.ndarray], # Expect np.ndarray[Tile]
        tile_render_size: int,
        world_ref: 'ZorpWorld' # New parameter
    ) -> None:
        super().__init__()
        self.window = window
        self.camera = camera
        self.map_data = map_data # This is np.ndarray[Tile]
        self.tile_render_size = tile_render_size
        self.world_ref = world_ref # Store reference to ZorpWorld

        # Rendering batches and groups
        self.batch = pyglet.graphics.Batch() # Main batch for all rendering
        self.tile_group = Group(order=0) # Group for map tiles
        self.agent_group = Group(order=1) # Group for agents (Zorps)
        # self.ui_group = Group(order=2) # Group for UI elements (if any)

        self.tile_sprites: List[pyglet.sprite.Sprite] = [] # List to hold tile sprites
        
        # self.zorp_sprites: Dict[int, pyglet.sprite.Sprite] = {} # Old: ent_id -> Sprite
        self.zorp_visuals: Dict[str, pyglet.shapes.ShapeBase] = {} # New: zorp.id (str) -> pyglet.shapes.Circle or similar
        
        # Placeholder for Zorp sprite sheet if we use sprites later
        # self.zorp_sprite_sheet = SpriteSheet(image_path='path/to/zorp_sheet.png', ...) 

        if self.map_data is not None:
            self._prepare_map_renderables()
        else:
            print("RenderSystem initialized with no map_data.")
        
        # Debug label for FPS
        self.fps_label = pyglet.text.Label(
            'FPS: 0', 
            font_name='Arial', 
            font_size=12, 
            x=10, y=self.window.height - 20, 
            anchor_x='left', anchor_y='top',
            batch=self.batch # Add to main batch
            # group=self.ui_group # If UI group is used
        )
        self.current_fps = 0.0 # To update the label

    def update_fps_display(self, fps: float) -> None:
        self.current_fps = fps
        self.fps_label.text = f"FPS: {self.current_fps:.2f}"
        # Adjust position if window resizes, or do it in on_resize
        self.fps_label.y = self.window.height - 20

    def _prepare_map_renderables(self) -> None:
        if self.map_data is None:
            print("Cannot prepare map renderables: map_data is None.")
            return

        # Clear existing tile sprites before regenerating
        for sprite in self.tile_sprites:
            sprite.delete()
        self.tile_sprites.clear()

        rows, cols = self.map_data.shape
        for r in range(rows):
            for c in range(cols):
                tile_enum = self.map_data[r, c]
                try:
                    meta = tile_enum.metadata
                    base_x = c * self.tile_render_size
                    base_y = r * self.tile_render_size

                    # Draw base rectangle
                    square = pyglet.shapes.Rectangle(
                        x=base_x,
                        y=base_y,
                        width=self.tile_render_size,
                        height=self.tile_render_size,
                        color=meta.base_color[:3], # pyglet shapes color is 3-tuple or 4-tuple for opacity
                        batch=self.batch,
                        group=self.tile_group
                    )
                    self.tile_sprites.append(square)

                    # Draw spots if defined
                    if meta.spot_color and meta.spot_count > 0:
                        spot_draw_size = self.tile_render_size * meta.spot_size_ratio
                        # Ensure spots are not too small to see or too large
                        spot_draw_size = max(1, min(spot_draw_size, self.tile_render_size / 2))
                        
                        for _ in range(meta.spot_count):
                            # Random offset within the tile, ensuring spots are fully visible
                            # by leaving a margin equal to spot_draw_size / 2
                            margin = spot_draw_size / 2
                            spot_x = base_x + margin + random.uniform(0, self.tile_render_size - spot_draw_size)
                            spot_y = base_y + margin + random.uniform(0, self.tile_render_size - spot_draw_size)
                            
                            spot_shape = pyglet.shapes.Circle(
                                x=spot_x + spot_draw_size / 2, # Circle x,y is center
                                y=spot_y + spot_draw_size / 2,
                                radius=spot_draw_size / 2,
                                color=meta.spot_color[:3],
                                batch=self.batch,
                                group=self.tile_group # Same group, drawn after base
                            )
                            self.tile_sprites.append(spot_shape)

                except Exception as e:
                    print(f"Error preparing sprite for tile {tile_enum} at ({r},{c}): {e}")
        print(f"Map renderables prepared. {len(self.tile_sprites)} tile sprites created.")

    def _get_zorp_color(self, energy: float, max_energy: float = 100.0) -> Tuple[int, int, int]:
        """Calculates Zorp color based on energy (Green to Red)."""
        energy_ratio = max(0, min(1, energy / max_energy))
        red = int(255 * (1 - energy_ratio))
        green = int(255 * energy_ratio)
        blue = 0
        return (red, green, blue)

    def process_main_render_loop(self, dt: float) -> None:
        self.window.clear()
        print(f"[RenderSystem] process_main_render_loop called")
        print(f"[RenderSystem] Number of tile sprites: {len(self.tile_sprites)}")
        print(f"[RenderSystem] Number of Zorps in world: {len(self.world_ref.all_zorps)}")
        print(f"[RenderSystem] Number of Zorp visuals: {len(self.zorp_visuals)}")
        
        # Apply camera transformations
        # pyglet.gl.glPushMatrix() # Removed due to AttributeError
        self.window.view = self.camera.get_view_matrix()
        
        # Update FPS label (Engine should call update_fps_display with actual FPS)
        # For now, just to ensure it's visible if batching is correct
        # self.fps_label.text = f"FPS: {self.current_fps:.2f}" 

        # --- Zorp Rendering --- 
        current_zorp_ids = {zorp.id for zorp in self.world_ref.all_zorps if zorp.alive}

        # Remove visuals for Zorps that are no longer present or dead
        for zorp_id in list(self.zorp_visuals.keys()):
            if zorp_id not in current_zorp_ids:
                self.zorp_visuals[zorp_id].delete()
                del self.zorp_visuals[zorp_id]

        # Update existing Zorps and add new ones
        for zorp in self.world_ref.all_zorps:
            print(f"DEBUG RENDER: Processing Zorp {zorp.id} Pos:{zorp.position} E:{zorp.energy:.1f} Alive:{zorp.alive}")
            if not zorp.alive:
                print(f"DEBUG RENDER: Zorp {zorp.id} is not alive, skipping visual update/creation.")
                continue

            tile_x, tile_y = zorp.position
            # Center of the tile for rendering
            world_x = tile_x * self.tile_render_size + self.tile_render_size / 2.0
            world_y = tile_y * self.tile_render_size + self.tile_render_size / 2.0
            radius = self.tile_render_size / 3.0
            color = self._get_zorp_color(zorp.energy)

            if zorp.id in self.zorp_visuals:
                shape = self.zorp_visuals[zorp.id]
                shape.x = world_x
                shape.y = world_y
                shape.radius = radius # In case tile_render_size changes with zoom, though radius is fixed here
                shape.color = color
                # shape.opacity = 255 # Ensure visible
            else:
                print(f"DEBUG RENDER: Creating NEW visual for Zorp {zorp.id} at W_Pos:({world_x:.1f},{world_y:.1f}) Color:{color}")
                new_shape = pyglet.shapes.Circle(
                    world_x, world_y, radius,
                    color=color,
                    batch=self.batch, # Use the main batch
                    group=self.agent_group # Render on top of tiles
                )
                self.zorp_visuals[zorp.id] = new_shape
                # Add debug rectangle overlay
                debug_rect = pyglet.shapes.Rectangle(
                    x=world_x - self.tile_render_size/4,
                    y=world_y - self.tile_render_size/4,
                    width=self.tile_render_size/2,
                    height=self.tile_render_size/2,
                    color=(255, 0, 255),
                    batch=self.batch,
                    group=self.agent_group
                )
                self.zorp_visuals[str(zorp.id) + '_debug'] = debug_rect
        
        print(f"[RenderSystem] About to call self.batch.draw()")
        # --- Draw everything --- 
        self.batch.draw() # Draw all batched sprites and shapes
        
        # Revert camera transformations for UI elements if any (or reset view)
        # pyglet.gl.glPopMatrix() # Removed due to AttributeError
        # To draw UI elements in screen space after this, reset the view:
        # self.window.view = pyglet.math.Mat4() # Identity matrix
        
        # Any UI elements drawn AFTER revert_transform will be in screen space
        # Example: self.fps_label.draw() if not in batch or if batch is drawn before revert
        # If FPS label is in the main batch, it's already drawn. Ensure its position is updated if needed.

    # This is the method called by esper.add_processor
    def process(self, dt: float) -> None:
        # print(f"RenderSystem.process called, dt: {dt}") # DEBUG
        if self.window.has_exit:
            return
        self.process_main_render_loop(dt) 