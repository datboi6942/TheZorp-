import pyglet
from typing import Dict, Tuple

class SpriteSheet:
    """Loads a sprite sheet and provides access to individual tiles as Pyglet Images/Textures."""

    def __init__(self, image_path: str, tile_width: int, tile_height: int) -> None:
        """Initializes the sprite sheet by loading the image and creating tile regions.

        Args:
            image_path: Path to the sprite sheet image file.
            tile_width: Width of a single tile in pixels.
            tile_height: Height of a single tile in pixels.
        """
        try:
            self.image = pyglet.image.load(image_path)
        except FileNotFoundError as e:
            print(f"Error loading sprite sheet image '{image_path}': {e}")
            print("Please ensure a placeholder image (e.g., 32x32px) exists at this path.")
            # Create a dummy 32x32 pink texture as a fallback
            pixels = [255, 105, 180, 255] * (32 * 32) # RGBA pink
            image_data = pyglet.image.ImageData(32, 32, 'RGBA', bytes(pixels))
            self.image = image_data
            # Ensure tile_width and tile_height match the fallback image
            tile_width = 32 
            tile_height = 32
            print(f"Using a fallback pink {tile_width}x{tile_height} tile.")

        self.tile_width = tile_width
        self.tile_height = tile_height
        self.texture = self.image.get_texture()

        # Create image regions for each tile (can be done on demand too)
        # For now, we might just provide a method to get a tile by grid coordinates.
        self._tiles: Dict[Tuple[int, int], pyglet.image.TextureRegion] = {}

    def get_tile(self, tile_x: int, tile_y: int) -> pyglet.image.TextureRegion:
        """Gets a specific tile from the sprite sheet by its grid coordinates (column, row).

        Args:
            tile_x: The column index of the tile (0-indexed).
            tile_y: The row index of the tile (0-indexed from top or bottom, depending on atlas).
                    Assuming 0,0 is top-left for region calculation from image data.

        Returns:
            A Pyglet TextureRegion for the specified tile.
        """
        if (tile_x, tile_y) in self._tiles:
            return self._tiles[(tile_x, tile_y)]

        region_x = tile_x * self.tile_width
        region_y = tile_y * self.tile_height # y from top of the image
        
        # Ensure region is within image bounds
        if not (
            0 <= region_x < self.image.width
            and 0 <= region_y < self.image.height
            and region_x + self.tile_width <= self.image.width
            and region_y + self.tile_height <= self.image.height
        ):
            print(f"Warning: Tile ({tile_x}, {tile_y}) is out of bounds for the sprite sheet '{self.image}'.")
            # Return a fallback tile (e.g., the first tile of the loaded texture, or a specific error tile)
            # For simplicity, returning the 0,0 region of the (potentially fallback) texture.
            return self.texture.get_region(0, 0, min(self.tile_width, self.texture.width), min(self.tile_height, self.texture.height))

        # y for texture.get_region is from bottom-left of the texture
        texture_region_y = self.texture.height - region_y - self.tile_height

        tile_region = self.texture.get_region(
            x=region_x, 
            y=texture_region_y, 
            width=self.tile_width, 
            height=self.tile_height
        )
        self._tiles[(tile_x, tile_y)] = tile_region
        return tile_region

    def get_default_tile(self) -> pyglet.image.TextureRegion:
        """Gets the tile at (0,0) as the default tile."""
        return self.get_tile(0, 0) 