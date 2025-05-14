import pyglet
from typing import Tuple
from pyglet import math # Import pyglet.math

DEFAULT_ZOOM_LEVELS = [0.5, 1.0, 2.0]
DEFAULT_PAN_SPEED = 200.0  # Pixels per second at zoom 1.0

class Camera:
    """Manages the game view's position (pan) and zoom level."""

    def __init__(
        self,
        window: pyglet.window.Window,
        x: float = 0.0,
        y: float = 0.0,
        zoom_levels: Tuple[float, ...] = tuple(DEFAULT_ZOOM_LEVELS),
        default_zoom_index: int = 1, # Index for DEFAULT_ZOOM_LEVELS (1.0)
    ) -> None:
        """Initializes the camera.

        Args:
            window: The pyglet window this camera is associated with.
            x: Initial x-coordinate of the camera's view center.
            y: Initial y-coordinate of the camera's view center.
            zoom_levels: A tuple of available zoom levels (e.g., 0.5, 1.0, 2.0).
            default_zoom_index: The initial index into zoom_levels.
        """
        self.window = window
        self.x = x
        self.y = y
        self.zoom_levels = sorted(list(zoom_levels))
        if not self.zoom_levels:
            self.zoom_levels = [1.0]
        
        if not (0 <= default_zoom_index < len(self.zoom_levels)):
            default_zoom_index = len(self.zoom_levels) // 2
        self._current_zoom_index = default_zoom_index
        self.zoom = self.zoom_levels[self._current_zoom_index]
        self.pan_speed = DEFAULT_PAN_SPEED

    def pan(self, dx: float, dy: float, dt: float) -> None:
        """Pans the camera by a given delta, adjusted for zoom and delta time.

        Args:
            dx: Change in x-direction (e.g., -1 for left, 1 for right).
            dy: Change in y-direction (e.g., -1 for down, 1 for up).
            dt: Delta time since the last frame, for frame-rate independent speed.
        """
        # Adjust pan speed by current zoom level: faster pan when zoomed out?
        # Or constant screen-space pan speed? For now, constant world units per second.
        effective_speed = self.pan_speed # / self.zoom # if speed should be world-units based
        self.x += dx * effective_speed * dt
        self.y += dy * effective_speed * dt
        # print(f"Panned to: ({self.x:.2f}, {self.y:.2f})")

    def zoom_in(self) -> None:
        """Zooms in to the next level if available."""
        if self._current_zoom_index < len(self.zoom_levels) - 1:
            self._current_zoom_index += 1
            self.zoom = self.zoom_levels[self._current_zoom_index]
            print(f"Zoomed in to: {self.zoom:.2f}")

    def zoom_out(self) -> None:
        """Zooms out to the next level if available."""
        if self._current_zoom_index > 0:
            self._current_zoom_index -= 1
            self.zoom = self.zoom_levels[self._current_zoom_index]
            print(f"Zoomed out to: {self.zoom:.2f}")

    def get_view_matrix(self) -> pyglet.math.Mat4:
        """Constructs and returns the camera's view matrix (pyglet.math.Mat4)."""
        matrix = math.Mat4() # Start with identity
        # The order of operations for Mat4 is typically: scale, then rotate, then translate.
        # However, to achieve the effect of: 
        # 1. Translate world so camera_pos is at origin (translate by -cam_x, -cam_y)
        # 2. Scale by zoom factor around this new origin
        # 3. Translate so the origin (which is now camera_pos) is at screen center
        # This can be done by: T_screen_center * S_zoom * T_neg_cam_pos
        
        # Create transformations
        translate_to_center = math.Mat4.from_translation(math.Vec3(self.window.width / 2, self.window.height / 2, 0))
        scale_transform = math.Mat4.from_scale(math.Vec3(self.zoom, self.zoom, 1))
        translate_camera_to_origin = math.Mat4.from_translation(math.Vec3(-self.x, -self.y, 0))
        
        # Combine them: effectively T_center * S_zoom * T_camera_to_origin
        # Pyglet's Mat4 multiplication is M_new = M_old @ M_transform
        # So, we apply transforms in reverse order of operation on points:
        matrix = translate_camera_to_origin @ matrix # Step 1 (applied first to points)
        matrix = scale_transform @ matrix           # Step 2
        matrix = translate_to_center @ matrix       # Step 3 (applied last to points)
        return matrix

    def reset_transform(self) -> None:
        """Resets the OpenGL matrix to identity, usually after drawing world objects
        and before drawing UI elements that should not be affected by the camera.
        Alternatively, use pyglet.gl.glPushMatrix/glPopMatrix around apply_transform.
        """
        # This is a bit simplistic. A better way is to use glPushMatrix/glPopMatrix
        # around the world drawing code. For now, this will undo the specific transforms.
        # However, it's better to instruct the user to use push/pop matrix.
        # For now, this function might not be strictly necessary if RenderSystem
        # manages its GL state carefully (e.g. always call apply_transform in its context).
        pass # Better managed with push/pop matrix in render system

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Converts world coordinates to screen coordinates."""
        # Apply zoom and camera translation
        view_x = (world_x - self.x) * self.zoom
        view_y = (world_y - self.y) * self.zoom
        # Translate to screen origin (center of window)
        screen_x = view_x + self.window.width / 2
        screen_y = view_y + self.window.height / 2
        return screen_x, screen_y

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Converts screen coordinates to world coordinates."""
        # Translate from screen origin (center of window)
        view_x = screen_x - self.window.width / 2
        view_y = screen_y - self.window.height / 2
        # Apply inverse zoom and camera translation
        world_x = (view_x / self.zoom) + self.x
        world_y = (view_y / self.zoom) + self.y
        return world_x, world_y 