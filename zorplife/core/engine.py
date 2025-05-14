import pyglet
import esper # esper module is now the world context
import time
from typing import List, Tuple, Type # Added Type for SystemClass

from zorplife.core.systems import System, RenderSystem, InputSystem # Import InputSystem
from zorplife.ui.camera import Camera # Import Camera

TARGET_FPS = 60.0
TARGET_SPF = 1.0 / TARGET_FPS  # Seconds per frame

class Engine:
    """Manages the game window, ECS world, systems, and the main game loop."""

    def __init__(self, width: int = 640, height: int = 480, caption: str = "ZorpLife") -> None:
        """Initializes the game engine.

        Args:
            width: The width of the game window.
            height: The height of the game window.
            caption: The title of the game window.
        """
        self.window = pyglet.window.Window(width, height, caption, resizable=True)
        # self.world = esper.World() # Removed: esper v3 uses module-level functions for default world
        self._running = False
        self._last_tick_time = time.perf_counter()

        # Initialize Camera first as other systems might need it
        self.camera = Camera(window=self.window, x=0.0, y=0.0) # Centered view for now

        # System priorities (lower number = higher priority, processed first)
        # This is a conceptual guide; esper processes in order of addition by default,
        # but add_processor has a priority argument.
        self._system_priority_map: List[Tuple[Type[System], int, tuple]] = [
            # (SystemClass, priority, init_args_tuple)
            (InputSystem, 0, (self.window, self.camera)),
            # (LogicSystem, 10, (self.camera,)), # Example, if logic needs camera
            (RenderSystem, 100, (self.window, self.camera)), # Pass camera to RenderSystem
            # Add other systems here, e.g.:
            # (PhysicsSystem, 10, ()),
            # (AISystem, 20, ()),
        ]

        self._initialize_systems()
        self._setup_event_handlers()
        self.window.push_handlers(on_resize=self.on_resize) # Handle window resize

    def _initialize_systems(self) -> None:
        """Initializes and adds systems to the ECS world based on priority."""
        print("Initializing systems...")
        # Ensure default esper world context is clean if re-initializing (though usually not needed for default)
        # esper.clear_database() # Or esper.delete_world(esper.DEFAULT_WORLD) then esper.switch_world(esper.DEFAULT_WORLD)
        # For now, assume fresh start or rely on esper's default behavior.

        for system_class, priority, init_args in sorted(self._system_priority_map, key=lambda x: x[1]):
            system_instance = system_class(*init_args)
            esper.add_processor(system_instance, priority=priority)
            # No longer assigning esper module to system_instance.world
            print(f"Added system: {system_class.__name__} with priority {priority}")

    def _setup_event_handlers(self) -> None:
        """Sets up Pyglet window event handlers (mostly for engine control)."""
        # Most input events (keys for panning, mouse scroll) are handled by InputSystem now.
        # Engine-level handlers are for things like ESC to quit, window close.
        @self.window.event
        def on_key_press(symbol: int, modifiers: int) -> None:
            if symbol == pyglet.window.key.ESCAPE:
                print("Escape pressed, exiting engine...")
                self.stop()
            # Other keys are handled by InputSystem via its pushed KeyStateHandler
            # or specific handlers like on_mouse_scroll.

        @self.window.event
        def on_close() -> None:
            print("Window closed, exiting engine...")
            self.stop()
        
        # Mouse scroll is handled in InputSystem's __init__ by setting window.on_mouse_scroll

    def on_resize(self, width: int, height: int) -> None:
        """Handles window resize events."""
        print(f"Window resized to: {width}x{height}")
        # Update camera viewport or projection if necessary
        # For basic 2D with glTranslate/glScale, this mainly affects gl.glViewport
        pyglet.gl.glViewport(0, 0, width, height)
        # The camera's apply_transform uses window.width/height, so it adapts automatically
        # to centering content. No specific camera update needed here for that part.
        # If using a projection matrix, it would need to be updated here.

    def run(self) -> None:
        """Starts and runs the main game loop."""
        if self._running:
            print("Engine is already running.")
            return

        print("Starting engine loop...")
        self._running = True
        self._last_tick_time = time.perf_counter() # Reset tick time before loop

        pyglet.app.event_loop.has_exit = False # Ensure pyglet loop doesn't exit prematurely

        while self._running and not self.window.has_exit:
            current_time = time.perf_counter()
            dt = current_time - self._last_tick_time
            self._last_tick_time = current_time

            # 1. Process events (Pyglet does this implicitly with pyglet.app.event_loop.iter_event() or dispatch_events)
            pyglet.app.platform_event_loop.dispatch_posted_events()
            self.window.dispatch_events() # Process window events like key presses, mouse, etc.

            if not self._running or self.window.has_exit: # Check after events
                break

            # 2. ECS Tick (Logic Delta)
            # Pass delta_time (dt) to all systems' process methods
            esper.process(dt) # Use module-level function, pass dt
            
            # 3. Render Delta (Handled by RenderSystem's process method called above)
            # self.window.clear() is now in RenderSystem
            # Drawing is also in RenderSystem
            
            self.window.flip() # Swaps the back buffer to the front

            # 4. Sleep to target FPS
            # This is a simple sleep; a more robust solution might involve a frame limiter
            # or a more complex timing mechanism.
            # pyglet.clock.tick() can also be used to regulate FPS if not using a custom loop.
            # For now, we use our own sleep.
            frame_duration = time.perf_counter() - current_time
            sleep_time = TARGET_SPF - frame_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            # else: we're running slow!

        print("Engine loop stopped.")
        if not self.window.has_exit:
            self.window.close()
        # pyglet.app.exit() # This can sometimes cause issues if called too eagerly or from wrong thread
        # The pyglet.app.run() in main.py will handle the actual application exit.
        # For a custom loop, ensuring self._running = False and window closed is key.
        pyglet.app.event_loop.exit() # Use this to signal the event loop (if managed externally) to stop

    def stop(self) -> None:
        """Stops the game loop."""
        print("Engine stop requested.")
        self._running = False
        # self.window.close() # Closing window here might be too soon if events are pending

