import esper
import pyglet
from typing import Any, TYPE_CHECKING, Optional, List
import numpy as np # For type hint np.ndarray
from zorplife.ui.camera import Camera # Import Camera
from zorplife.ui.sprites import SpriteSheet # Import SpriteSheet
from zorplife.world.tiles import Tile # For type hint

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

class RenderSystem(System):
    def __init__(
        self, 
        window: pyglet.window.Window, 
        camera: Camera, 
        map_data: Optional[np.ndarray], # Expect np.ndarray[Tile]
        tile_render_size: int
    ) -> None:
        super().__init__()
        self.window = window
        self.camera = camera
        self.map_data = map_data
        self.tile_render_size = tile_render_size
        
        # For drawing tiles - a batch is more efficient
        self.tile_batch = pyglet.graphics.Batch()
        self._tile_sprites: List[pyglet.shapes.Rectangle] = [] # Or pyglet.sprite.Sprite if using atlas

        try:
            self.fps_display = pyglet.window.FPSDisplay(window=self.window)
        except AttributeError: 
            self.fps_display = pyglet.text.Label("FPS: N/A", x=10, y=self.window.height - 20)
            print("Warning: pyglet.window.FPSDisplay not found, using basic Label for FPS.")

        # SpriteSheet for default tile is no longer needed here if drawing map tiles directly
        # self.sprite_sheet = SpriteSheet("zorplife/data/sprites/atlas.png", 32, 32)
        # self.default_tile_sprite = pyglet.sprite.Sprite(img=self.sprite_sheet.get_default_tile())

        self._prepare_map_renderables() # Initial creation of renderable shapes for tiles

    def _prepare_map_renderables(self) -> None:
        """Prepares pyglet shapes for each map tile. For now, uses colored rectangles."""
        if self.map_data is None:
            return
        
        self._tile_sprites.clear() # Clear previous sprites if any (e.g. on map reload)
        # self.tile_batch = pyglet.graphics.Batch() # Recreate batch or update existing items

        height, width = self.map_data.shape
        for r in range(height):
            for c in range(width):
                tile_enum: Tile = self.map_data[r, c]
                tile_meta = tile_enum.metadata
                world_x = c * self.tile_render_size
                world_y = r * self.tile_render_size

                rect = pyglet.shapes.Rectangle(
                    x=world_x, 
                    y=world_y, 
                    width=self.tile_render_size, 
                    height=self.tile_render_size, 
                    color=tile_meta.color[:3],
                    batch=self.tile_batch
                )
                self._tile_sprites.append(rect)

                # Draw spots/dots for special tiles
                if tile_enum == Tile.APPLE_TREE:
                    # Green base, red spots
                    rect.color = (34, 139, 34)  # Overwrite to green
                    for i in range(3):
                        spot = pyglet.shapes.Circle(
                            x=world_x + self.tile_render_size * (0.3 + 0.2 * i),
                            y=world_y + self.tile_render_size * (0.6 - 0.2 * i),
                            radius=self.tile_render_size * 0.12,
                            color=(220, 20, 60),  # Red
                            batch=self.tile_batch
                        )
                        self._tile_sprites.append(spot)
                elif tile_enum == Tile.GOLD_DEPOSIT:
                    # Gray base, yellow spots
                    rect.color = (130, 130, 130)
                    for i in range(2):
                        spot = pyglet.shapes.Circle(
                            x=world_x + self.tile_render_size * (0.4 + 0.2 * i),
                            y=world_y + self.tile_render_size * (0.4 - 0.2 * i),
                            radius=self.tile_render_size * 0.13,
                            color=(255, 215, 0),  # Yellow
                            batch=self.tile_batch
                        )
                        self._tile_sprites.append(spot)
                elif tile_enum == Tile.IRON_DEPOSIT:
                    # Gray base, beige spots
                    rect.color = (130, 130, 130)
                    for i in range(2):
                        spot = pyglet.shapes.Circle(
                            x=world_x + self.tile_render_size * (0.3 + 0.3 * i),
                            y=world_y + self.tile_render_size * (0.5 - 0.2 * i),
                            radius=self.tile_render_size * 0.13,
                            color=(222, 184, 135),  # Beige
                            batch=self.tile_batch
                        )
                        self._tile_sprites.append(spot)
                elif tile_enum == Tile.TREE:
                    # Brown base, darker brown spots
                    rect.color = (139, 69, 19)
                    for i in range(2):
                        spot = pyglet.shapes.Circle(
                            x=world_x + self.tile_render_size * (0.4 + 0.2 * i),
                            y=world_y + self.tile_render_size * (0.3 + 0.2 * i),
                            radius=self.tile_render_size * 0.10,
                            color=(90, 40, 10),  # Darker brown
                            batch=self.tile_batch
                        )
                        self._tile_sprites.append(spot)
                elif tile_enum == Tile.TREE_CANOPY:
                    rect.color = (34, 139, 34)  # Green, no spots

        print(f"Prepared {len(self._tile_sprites)} tile renderables for map.")

    def process(self, dt: float) -> None:
        self.window.clear()

        original_view = self.window.view
        self.window.view = self.camera.get_view_matrix()

        # === Draw world objects here ===
        if self.map_data is not None:
            self.tile_batch.draw() # Draw all tiles in the batch
        # ==============================

        self.window.view = original_view

        if isinstance(self.fps_display, pyglet.text.Label):
            self.fps_display.y = self.window.height - 20
        self.fps_display.draw() 