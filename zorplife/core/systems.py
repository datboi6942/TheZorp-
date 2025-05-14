import esper
import pyglet
from typing import Any, TYPE_CHECKING
from zorplife.ui.camera import Camera # Import Camera
from zorplife.ui.sprites import SpriteSheet # Import SpriteSheet

# if TYPE_CHECKING:
#     # No longer needed as esper.World object is not used.
#     pass

class System(esper.Processor):
    """Base class for all systems in the ECS.
    
    The esper module functions (e.g., esper.get_components) should be used directly
    by importing esper in the system's file.
    """
    # world: esper.World # Removed, esper v3 uses module level functions.

    def __init__(self) -> None:
        super().__init__()

    def process(self, *args: Any, **kwargs: Any) -> None:
        """Process game logic for the system.

        This method is called by `esper.process()` typically once per frame.
        Subclasses must override this method to implement their logic.

        Args:
            *args: Variable length argument list, often includes delta_time (dt).
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the process method."
        )

class InputSystem(System):
    """Handles player input for camera control and other actions."""
    def __init__(self, window: pyglet.window.Window, camera: Camera) -> None:
        super().__init__()
        self.window = window
        self.camera = camera
        self.keys = pyglet.window.key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        
        # Register mouse scroll handler directly on the window
        # because InputSystem.process is driven by dt, not events directly for scroll
        self.window.on_mouse_scroll = self.handle_mouse_scroll

    def process(self, dt: float) -> None:
        """Process continuous input like key holds for camera panning."""
        dx, dy = 0.0, 0.0
        if self.keys[pyglet.window.key.LEFT] or self.keys[pyglet.window.key.A]:
            dx -= 1.0
        if self.keys[pyglet.window.key.RIGHT] or self.keys[pyglet.window.key.D]:
            dx += 1.0
        if self.keys[pyglet.window.key.UP] or self.keys[pyglet.window.key.W]:
            dy += 1.0
        if self.keys[pyglet.window.key.DOWN] or self.keys[pyglet.window.key.S]:
            dy -= 1.0

        if dx != 0.0 or dy != 0.0:
            self.camera.pan(dx, dy, dt)

    def handle_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        """Handles mouse scroll events for camera zooming."""
        if scroll_y > 0:
            self.camera.zoom_in()
        elif scroll_y < 0:
            self.camera.zoom_out()

class RenderSystem(System):
    def __init__(self, window: pyglet.window.Window, camera: Camera) -> None:
        super().__init__()
        self.window = window
        self.camera = camera
        try:
            self.fps_display = pyglet.window.FPSDisplay(window=self.window)
        except AttributeError: 
            self.fps_display = pyglet.text.Label("FPS: N/A", x=10, y=self.window.height - 20)
            print("Warning: pyglet.window.FPSDisplay not found, using basic Label for FPS.")

        # Load the sprite sheet
        # Assuming the user will create a placeholder atlas.png in the specified path
        self.sprite_sheet = SpriteSheet("zorplife/data/sprites/atlas.png", 32, 32)
        self.default_tile_sprite = pyglet.sprite.Sprite(img=self.sprite_sheet.get_default_tile())

    def process(self, dt: float) -> None:
        self.window.clear()

        original_view = self.window.view
        self.window.view = self.camera.get_view_matrix()

        # === Draw world objects here ===
        # Draw the default tile at world origin (0,0)
        # Sprite position is its bottom-left corner. To center it at (0,0):
        self.default_tile_sprite.x = -self.default_tile_sprite.width / 2
        self.default_tile_sprite.y = -self.default_tile_sprite.height / 2
        self.default_tile_sprite.draw()
        # ==============================

        self.window.view = original_view

        if isinstance(self.fps_display, pyglet.text.Label):
            self.fps_display.y = self.window.height - 20
        self.fps_display.draw() 