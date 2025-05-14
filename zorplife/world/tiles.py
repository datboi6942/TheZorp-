from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List

class ResourceType(Enum):
    """Enum for different types of resources available in the world."""
    PLANT_MATTER = auto()
    WOOD = auto()
    STONE = auto()
    WATER = auto()
    CLAY = auto()
    IRON_ORE = auto()
    GOLD_ORE = auto()
    MUSHROOMS = auto()
    APPLES = auto()
    ORGANIC_MATTER = auto() # From corpses, etc.
    NONE = auto() # For tiles with no specific resource

@dataclass(frozen=True)
class TileMetadata:
    """Metadata associated with each tile type.

    Args:
        name: User-friendly name of the tile type.
        char: Character representation for simple text-based rendering or debugging.
        color: RGBA tuple for simple graphical rendering (0-255).
        passable: Whether agents can walk on this tile.
        resource_type: The primary resource this tile can provide (if any).
        harvest_time_modifier: Multiplier for time taken to harvest resource (1.0 is normal).
        # harvest_action: Action needed to harvest (e.g., GATHER, MINE, CHOP) - for later stage
    """
    name: str
    char: str
    color: Tuple[int, int, int, int] # RGBA
    passable: bool = True
    resource_type: ResourceType = ResourceType.NONE
    harvest_time_modifier: float = 1.0

class Tile(Enum):
    """Enum for different tile types in the game world.
    
    Each enum member holds TileMetadata.
    """
    # Biome/Terrain Tiles
    GRASSLAND = TileMetadata(name="Grassland", char=".", color=(100, 200, 50, 255), resource_type=ResourceType.PLANT_MATTER)
    FOREST_FLOOR = TileMetadata(name="Forest Floor", char=":", color=(50, 150, 20, 255), resource_type=ResourceType.PLANT_MATTER)
    WATER_SHALLOW = TileMetadata(name="Shallow Water", char="~", color=(50, 100, 200, 150), passable=False, resource_type=ResourceType.WATER)
    WATER_DEEP = TileMetadata(name="Deep Water", char="≈", color=(30, 70, 180, 255), passable=False, resource_type=ResourceType.WATER)
    CLIFF_ROCK = TileMetadata(name="Cliff Rock", char="#", color=(100, 100, 100, 255), passable=False, resource_type=ResourceType.STONE)
    RIVERBANK_CLAY = TileMetadata(name="Riverbank Clay", char="r", color=(160, 120, 80, 255), resource_type=ResourceType.CLAY)
    MOUNTAIN_PEAK = TileMetadata(name="Mountain Peak", char="▲", color=(140, 140, 150, 255), passable=False, resource_type=ResourceType.STONE)
    MOUNTAIN_SLOPE = TileMetadata(name="Mountain Slope", char="/", color=(110, 110, 120, 255), passable=True, resource_type=ResourceType.STONE) # Potential for IRON_ORE/GOLD_ORE
    FOREST_HEAVY = TileMetadata(name="Heavy Forest", char="F", color=(30, 120, 10, 255), passable=True, resource_type=ResourceType.WOOD) # Denser, source of MUSHROOMS
    
    # Resource-specific Tiles (can overlay or be distinct)
    TREE = TileMetadata(name="Tree", char="T", color=(139, 69, 19, 255), passable=False, resource_type=ResourceType.WOOD)
    TREE_CANOPY = TileMetadata(name="Tree Canopy", char="^", color=(34, 139, 34, 255), passable=True, resource_type=ResourceType.NONE)
    APPLE_TREE = TileMetadata(name="Apple Tree", char="A", color=(255, 0, 0, 255), passable=False, resource_type=ResourceType.APPLES) # Red 'A' for apple, also provides WOOD
    ROCK_OUTCROP = TileMetadata(name="Rock Outcrop", char="S", color=(130, 130, 130, 255), passable=True, resource_type=ResourceType.STONE) # Small rocks on passable terrain
    IRON_DEPOSIT = TileMetadata(name="Iron Deposit", char="I", color=(192, 192, 192, 255), passable=True, resource_type=ResourceType.IRON_ORE)
    GOLD_DEPOSIT = TileMetadata(name="Gold Deposit", char="G", color=(255, 215, 0, 255), passable=True, resource_type=ResourceType.GOLD_ORE)
    MUSHROOM_PATCH = TileMetadata(name="Mushroom Patch", char="m", color=(150, 75, 0, 255), passable=True, resource_type=ResourceType.MUSHROOMS)

    # Fallback/Empty
    VOID = TileMetadata(name="Void", char=" ", color=(0, 0, 0, 255), passable=False)

    @property
    def metadata(self) -> TileMetadata:
        return self.value

    @classmethod
    def get_all_tiles(cls) -> List['Tile']:
        return list(cls)

# Example usage:
# grassland_tile = Tile.GRASSLAND
# print(f"Tile: {grassland_tile.name}, Color: {grassland_tile.metadata.color}, Passable: {grassland_tile.metadata.passable}")
# print(f"Resource: {grassland_tile.metadata.resource_type.name}") 