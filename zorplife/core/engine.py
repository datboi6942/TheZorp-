import pyglet
import esper
import time
import random # For spawning Zorps
from typing import List, Tuple, Type, Optional
import numpy as np

from zorplife.core.systems import System, RenderSystem, InputSystem #, MetabolismSystem, ReproductionSystem # Agent Systems
# from zorplife.agents.components import Position, Energy, Age, Genetics, AgentMarker # Agent Components
from zorplife.ui.camera import Camera # Import Camera
from zorplife.world.mapgen import MapGenerator # Import MapGenerator
from zorplife.world.tiles import Tile # For type hint
from zorplife.config import GameConfig
from zorplife.world.world import ZorpWorld # New import
from zorplife.core.zorp import Zorp  # New import

TARGET_FPS = 60.0
TARGET_SPF = 1.0 / TARGET_FPS  # Seconds per frame

class Engine:
    """Manages the game window, ECS world, systems, and the main game loop."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self.window: Optional[pyglet.window.Window] = None
        self.camera: Optional[Camera] = None
        # self.world = esper.World() # esper v3 uses module-level functions # No longer using esper for Zorp entities directly
        self.game_speed_multiplier = 1.0
        self.stop_requested = False
        self.first_process_call_done = False # For debugging system registration
        # self.total_zorps_spawned = 0 # Track spawned zorps, will be handled by ZorpWorld

        self.map_generator: MapGenerator = MapGenerator(
            width=self.config.map_width,
            height=self.config.map_height,
            seed=self.config.map_seed # Pass the seed from config
            # tile_render_size was removed as it's not an __init__ param for MapGenerator
        )
        
        # Initialize ZorpWorld
        self.world: Optional[ZorpWorld] = None


        # Initialize systems to None, they will be created in initialize_systems
        self.render_system: Optional[RenderSystem] = None
        self.input_system: Optional[InputSystem] = None
        # self.metabolism_system: Optional[MetabolismSystem] = None # To be phased out for Zorp class logic
        # self.reproduction_system: Optional[ReproductionSystem] = None # To be phased out for Zorp class logic

        self.frame_count = 0
        self.fps_update_interval = 1.0  # Update FPS display every N seconds
        self.time_since_last_fps_update = 0.0
        self.current_fps = 0.0

        # Initialize Pyglet window and ECS systems
        self._initialize_pyglet()
        self.initialize_systems()
        self.initialize_world()

        # Schedule the main update function
        pyglet.clock.schedule_interval(self.update, 1 / self.config.target_fps)

    def _initialize_pyglet(self) -> None:
        if not self.config.headless_mode:
            try:
                pyglet.options['debug_gl'] = self.config.pyglet_debug_gl
                self.window = pyglet.window.Window(
                    width=self.config.window_width,
                    height=self.config.window_height,
                    caption="ZorpLife Engine",
                    resizable=True
                )
                self.camera = Camera(self.window, self.config.camera_pan_speed, self.config.camera_zoom_speed)
                # print("Pyglet window and camera initialized.") # DEBUG

                @self.window.event
                def on_draw() -> None:
                    # if self.window: # Ensure window exists # Redundant clear, RenderSystem handles it.
                        # self.window.clear() 
                    if self.render_system: # Ensure render_system exists
                        # RenderSystem's process method is called by esper.process
                        # but if we needed direct drawing control it would be here.
                        # For now, on_draw is mainly for window.clear if not using RenderSystem for it.
                        # However, RenderSystem.process_main_render_loop already calls window.clear().
                        pass # RenderSystem handles drawing within its process method via the batch

                @self.window.event
                def on_resize(width: int, height: int) -> None:
                    if self.camera:
                        # self.camera.on_resize(width, height) # Removed: Camera class does not have this method
                        pass # Camera.get_view_matrix() uses window.width/height directly
                    print(f"Window resized to: {width}x{height}")
                
                @self.window.event
                def on_close() -> None:
                    print("Window closed, exiting engine...")
                    self.stop_requested = True
                    # pyglet.app.exit() # This will be handled by the update loop checking stop_requested
            except Exception as e:
                print(f"Failed to initialize Pyglet window: {e}")
                self.window = None
                self.camera = None
        else:
            print("Headless mode: Pyglet window not initialized.")

    def initialize_systems(self) -> None:
        print("Initializing systems...")
        # Input System (only if not headless)
        if self.window and self.camera: # Ensure window and camera are initialized
            self.input_system = InputSystem(self.window, self.camera)
            esper.add_processor(self.input_system, priority=0) # InputSystem might still use esper if it's a Processor
            print(f"Added system: InputSystem with priority 0")

        # Metabolism System (Commented out - Zorp class handles its metabolism)
        # self.metabolism_system = MetabolismSystem(map_generator_ref=self.map_generator, tile_render_size=self.config.tile_render_size)
        # esper.add_processor(self.metabolism_system, priority=5)
        # print(f"Added system: MetabolismSystem with priority 5")

        # Reproduction System (Commented out - Zorp class will handle reproduction)
        # self.reproduction_system = ReproductionSystem(tile_render_size=self.config.tile_render_size)
        # esper.add_processor(self.reproduction_system, priority=10)
        # print(f"Added system: ReproductionSystem with priority 10")

        # Render System (only if not headless)
        if self.window and self.camera: # Ensure window and camera are initialized
            self.render_system = RenderSystem(
                window=self.window, 
                camera=self.camera, 
                map_data=self.map_generator.tile_grid_np, # Initial map data
                tile_render_size=self.config.tile_render_size,
                world_ref=self.world # Pass the ZorpWorld instance
            )
            # esper.add_processor(self.render_system, priority=100) # No longer needed
            print(f"Added system: RenderSystem with priority 100")
        
        # print("Systems initialized and added to Esper.") # DEBUG

    def update(self, dt: float) -> None:
        # Calculate actual delta time, considering game speed
        # actual_dt = dt * self.game_speed_multiplier # Apply game speed multiplier if you have one

        if self.stop_requested:
            # print("Engine stop requested, exiting pyglet app.") # DEBUG
            pyglet.app.exit()
            return

        # Update ZorpWorld
        self.world.tick()

        # Process inputs to update camera state
        if self.input_system:
            self.input_system.process(dt)

        # Call RenderSystem directly (not via esper)
        if self.render_system:
            self.render_system.process_main_render_loop(dt)

        # First debug print for systems
        if not self.first_process_call_done:
            print("--- Registered Esper Processors (before first process call) ---")
            # We can't easily inspect module-level esper.processors like in older versions.
            # We'll rely on the print statements from initialize_systems to confirm order.
            # Priorities: Input (0), Metabolism (5), Reproduction (10), Render (100)
            # Example of manual print if needed for debugging system presence:
            if self.input_system: print(f"  - InputSystem Present")
            # if self.metabolism_system: print(f"  - MetabolismSystem Present") # Disabled
            # if self.reproduction_system: print(f"  - ReproductionSystem Present") # Disabled
            if self.render_system: print(f"  - RenderSystem Present")
            print("-------------------------------------------------------------")
            self.first_process_call_done = True
        
        # Log entities just before calling esper.process - This will likely show 0 if we stop creating ECS agents
        # try:
        #     agents_before_tick = list(esper.get_components(AgentMarker, Position, Energy, Age))
        #     print(f"[DEBUG Engine.update] Agents before esper.process ({len(agents_before_tick)}): IDs {[e for e,c in agents_before_tick[:3]]}...") # Show first few IDs
        #     if agents_before_tick:
        #         ent_id, (marker, pos, energy_comp, age_comp) = agents_before_tick[0]
        #         print(f"    First agent {ent_id} details: Pos=({pos.x:.1f},{pos.y:.1f}), Hunger={energy_comp.hunger:.1f}, Age={age_comp.current_age:.2f}")
        #     elif self.total_zorps_spawned > 0: # Only print if we expected Zorps
        #          print(f"    No agents found by Engine.update before esper.process, despite spawning {self.total_zorps_spawned}.")
        #
        # except Exception as e:
        #     print(f"[ERROR Engine.update] during pre-process agent check: {e}")


        # esper.process(dt) # Use module-level function, pass dt - Commented out as Zorp logic is now in ZorpWorld.tick()
                          # InputSystem is event-driven, RenderSystem will be changed.

        # Log entities just after calling esper.process - Also likely 0
        # try:
        #     agents_after_tick = list(esper.get_components(AgentMarker, Position, Energy, Age))
        #     print(f"[DEBUG Engine.update] Agents after esper.process ({len(agents_after_tick)}).")
        #     if agents_before_tick and not agents_after_tick: # If they disappeared THIS tick
        #          print(f"    !!! ALL AGENTS ({len(agents_before_tick)}) DISAPPEARED DURING THIS esper.process() CALL !!!")
        #     elif not agents_after_tick and self.total_zorps_spawned > 0 and not agents_before_tick : # Still gone, were gone before
        #          print(f"    Still no agents found by Engine.update after esper.process.")
        #
        # except Exception as e:
        #     print(f"[ERROR Engine.update] during post-process agent check: {e}")

        # FPS calculation
        self.frame_count += 1
        self.time_since_last_fps_update += dt
        if self.time_since_last_fps_update >= self.fps_update_interval:
            self.current_fps = self.frame_count / self.time_since_last_fps_update
            self.frame_count = 0
            self.time_since_last_fps_update = 0
            # print(f"FPS: {self.current_fps:.2f}") # DEBUG FPS
            if self.render_system: # Update FPS display in RenderSystem
                self.render_system.update_fps_display(self.current_fps)

        # Control game loop speed if not vsyncing or if target_fps is very high
        # This simple sleep is not ideal for precise timing but can prevent 100% CPU usage
        # if 1.0 / self.config.target_fps > dt:
        #     sleep_time = (1.0 / self.config.target_fps) - dt
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)
        # A more robust way is to let pyglet.clock handle it, or use pyglet.app.run() without manual sleep.

    def initialize_world(self) -> None:
        print(f"Generating map ({self.map_generator.width}x{self.map_generator.height}) with seed: {self.config.map_seed}")
        self.map_generator.generate_map()
        print("Map generation complete.")

        # Pass the generated map_data (Tile enum array) to the RenderSystem
        if self.render_system:
            self.render_system.map_data = self.map_generator.tile_grid_np
            self.render_system._prepare_map_renderables() # Call to update renderables

        # Re-instantiate ZorpWorld with map_data
        self.world = ZorpWorld(self.config, self.map_generator.tile_grid_np)
        if self.render_system:
            self.render_system.world_ref = self.world

        # Spawn initial Zorp population into ZorpWorld
        print(f"Spawning {self.config.initial_zorp_population} Zorps into ZorpWorld...")
        for i in range(self.config.initial_zorp_population):
            # Find a walkable tile for spawning
            spawn_attempts = 0
            max_spawn_attempts = 1000 # Prevent infinite loop if map has no walkable tiles
            tile_x, tile_y = -1, -1
            while spawn_attempts < max_spawn_attempts:
                tile_x = random.randint(0, self.map_generator.width - 1)
                tile_y = random.randint(0, self.map_generator.height - 1)
                if self.map_generator.tile_grid_np[tile_y, tile_x].metadata.passable:
                    break
                spawn_attempts += 1
            if spawn_attempts == max_spawn_attempts:
                print(f"Warning: Could not find a walkable tile to spawn Zorp {i+1} after {max_spawn_attempts} attempts. Skipping this Zorp.")
                continue
            # Create Zorp instance with tile-based coordinates
            new_zorp = Zorp(position=(tile_x, tile_y)) # Energy, genome, etc., use defaults from Zorp class
            self.world.add_zorp(new_zorp)
            # Debug: Print spawn info
            tile_enum = self.map_generator.tile_grid_np[tile_y][tile_x]
            resource_type = tile_enum.metadata.resource_type
            print(f"DEBUG: Spawned Zorp {new_zorp.id} at {new_zorp.position} on tile {tile_enum.name} (resource: {resource_type}) with E: {new_zorp.energy:.1f} Alive: {new_zorp.alive}")
            # Set high starting energy/hunger for debug
            new_zorp.energy = 1000.0
        print(f"Successfully spawned {len(self.world.all_zorps)} Zorps into ZorpWorld.")

        # Count Zorps in ECS after spawning (This will be 0 for AgentMarker if we removed creation)
        # zorp_entities_after_spawn = list(esper.get_components(AgentMarker, Position, Energy, Age, Genetics))
        # print(f"[DEBUG Engine.initialize_world] Zorp entities in ECS after spawn: {len(zorp_entities_after_spawn)}")
        # if zorp_entities_after_spawn:
            # print(f"[DEBUG] First Zorp entity ID after spawn: {zorp_entities_after_spawn[0][0]}") # Entity ID
            # first_ent_id = zorp_entities_after_spawn[0][0]
            # Quick check for key components
            # if esper.has_component(first_ent_id, Position) and esper.has_component(first_ent_id, AgentMarker) and esper.has_component(first_ent_id, Energy) and esper.has_component(first_ent_id, Age):
            #     print(f"[DEBUG] First Zorp entity {first_ent_id} has required components.")
            # else:
            #     print(f"[DEBUG] First Zorp entity {first_ent_id} MISSING some components.")
        # elif self.config.initial_zorp_population > 0:
        # print(f"[DEBUG Engine.initialize_world] No Zorp entities with AgentMarker found in ECS after ZorpWorld spawning (expected).")


    def run(self) -> None:
        print("Starting engine loop...")
        if not self.config.headless_mode and self.window:
            pyglet.app.run() # This starts the main pyglet event loop
        elif self.config.headless_mode:
            # Headless mode main loop (simple version)
            # For a more robust headless mode, you might want a more sophisticated loop
            # that still respects target_fps without relying on pyglet.app.run()
            print("Running in headless mode...")
            last_time = time.perf_counter()
            while not self.stop_requested:
                current_time = time.perf_counter()
                dt = current_time - last_time
                last_time = current_time
                self.update(dt) # Call update directly
                
                # Approximate sleep to maintain target_fps
                # This is very basic; a more precise timer might be needed for strict headless FPS
                time_to_sleep = (1.0 / self.config.target_fps) - dt
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
            print("Headless mode finished.")
        else:
            print("Error: Window not initialized, cannot start Pyglet app.run(). Engine will not run.")

        print("Engine loop stopped.")

    def shutdown(self) -> None:
        print("Shutting down ZorpLife Engine...")
        self.stop_requested = True # Signal update loop to stop
        
        # Explicitly clear Esper database if necessary, though module-level might not require this
        # For esper v3, if you want to ensure a clean state for potential re-runs or tests:
        # esper.clear_database() # Use with caution, check if this is the desired behavior for esper v3
        # print("Esper database cleared (if applicable for esper v3).")

        if self.window:
            # print("Attempting to close window...") # DEBUG
            # self.window.close() # Closing window here might be too soon if events are pending
            pass # Window closure is handled by on_close event setting stop_requested and update loop exiting pyglet.app

        print("ZorpLife Engine finished.")

