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
        base_color: RGBA tuple for the primary color of the tile (0-255).
        passable: Whether agents can walk on this tile.
        resource_type: The primary resource this tile can provide (if any).
        harvest_time_modifier: Multiplier for time taken to harvest resource (1.0 is normal).
        spot_color: Optional RGBA tuple for spots/details on the tile.
        spot_count: Number of spots to draw if spot_color is defined.
        spot_size_ratio: Ratio of spot size relative to tile_render_size (e.g., 0.1 for 10%).
        is_structure_component: If this tile is part of a larger structure (e.g. tree part).
        energy_value: Amount of energy a Zorp gains from eating this tile (if applicable).
    """
    name: str
    char: str
    base_color: Tuple[int, int, int, int] # RGBA
    passable: bool = True
    resource_type: ResourceType = ResourceType.NONE
    harvest_time_modifier: float = 1.0
    spot_color: Optional[Tuple[int, int, int, int]] = None
    spot_count: int = 0
    spot_size_ratio: float = 0.1 # Default spot size ratio
    # is_structure_component: bool = False # Consider if needed for apple trees
    energy_value: float = 0.0 # New field for energy gained from eating

class Tile(Enum):
    """Enum for different tile types in the game world.
    
    Each enum member holds TileMetadata.
    """
    # Biome/Terrain Tiles
    GRASSLAND = TileMetadata(name="Grassland", char=".", base_color=(100, 200, 50, 255), resource_type=ResourceType.PLANT_MATTER, energy_value=20.0)
    FOREST_FLOOR = TileMetadata(name="Forest Floor", char=":", base_color=(50, 150, 20, 255), resource_type=ResourceType.PLANT_MATTER, energy_value=20.0)
    WATER_SHALLOW = TileMetadata(name="Shallow Water", char="~", base_color=(50, 100, 200, 150), passable=False, resource_type=ResourceType.WATER)
    WATER_DEEP = TileMetadata(name="Deep Water", char="≈", base_color=(30, 70, 180, 255), passable=False, resource_type=ResourceType.WATER)
    CLIFF_ROCK = TileMetadata(name="Cliff Rock", char="#", base_color=(100, 100, 100, 255), passable=False, resource_type=ResourceType.STONE)
    RIVERBANK_CLAY = TileMetadata(name="Riverbank Clay", char="r", base_color=(160, 120, 80, 255), resource_type=ResourceType.CLAY)
    MOUNTAIN_PEAK = TileMetadata(name="Mountain Peak", char="▲", base_color=(140, 140, 150, 255), passable=False, resource_type=ResourceType.STONE)
    MOUNTAIN_SLOPE = TileMetadata(name="Mountain Slope", char="/", base_color=(110, 110, 120, 255), passable=True, resource_type=ResourceType.STONE) # Potential for IRON_ORE/GOLD_ORE
    FOREST_HEAVY = TileMetadata(name="Heavy Forest", char="F", base_color=(30, 120, 10, 255), passable=True, resource_type=ResourceType.WOOD) # Denser, source of MUSHROOMS
    
    # Resource-specific Tiles (can overlay or be distinct)
    TREE_TRUNK = TileMetadata(name="Tree Trunk", char="T", base_color=(139, 69, 19, 255), passable=False, resource_type=ResourceType.WOOD)
    TREE_CANOPY = TileMetadata(name="Tree Canopy", char="^", base_color=(34, 139, 34, 200), passable=True, resource_type=ResourceType.NONE) # Slightly transparent

    # APPLE_TREE = TileMetadata(name="Apple Tree", char="A", base_color=(255, 0, 0, 255), passable=False, resource_type=ResourceType.APPLES) # Old definition, to be replaced
    APPLE_TREE_TRUNK = TileMetadata(name="Apple Tree Trunk", char="t", base_color=(165, 42, 42, 255), passable=False, resource_type=ResourceType.WOOD) # Brown trunk
    APPLE_TREE_LEAVES = TileMetadata(name="Apple Tree Leaves", char="a", base_color=(50, 205, 50, 255), passable=True, resource_type=ResourceType.APPLES, spot_color=(255,0,0,255), spot_count=3, spot_size_ratio=0.15, energy_value=10.0) # Green leaves, red apple spots

    ROCK_OUTCROP = TileMetadata(name="Rock Outcrop", char="S", base_color=(130, 130, 130, 255), passable=True, resource_type=ResourceType.STONE) # Small rocks on passable terrain
    
    IRON_DEPOSIT = TileMetadata(
        name="Iron Deposit", char="I", 
        base_color=(128, 128, 128, 255), # Grey base
        passable=True, resource_type=ResourceType.IRON_ORE,
        spot_color=(210, 180, 140, 255), # Beige spots
        spot_count=4, spot_size_ratio=0.12
    )
    GOLD_DEPOSIT = TileMetadata(
        name="Gold Deposit", char="G", 
        base_color=(105, 105, 105, 255), # Darker Grey base
        passable=True, resource_type=ResourceType.GOLD_ORE,
        spot_color=(255, 215, 0, 255), # Yellow spots
        spot_count=3, spot_size_ratio=0.1
    )
    MUSHROOM_PATCH = TileMetadata(
        name="Mushroom Patch", char="m", 
        base_color=(220, 0, 0, 255), # Red base (for mushroom cap)
        passable=True, resource_type=ResourceType.MUSHROOMS,
        spot_color=(255, 255, 255, 255), # White spots
        spot_count=3, spot_size_ratio=0.15, # Spots represent mushroom caps
        energy_value=30.0 # Increased energy_value
    )

    # Special / Agent-related
    ORGANIC_REMAINS = TileMetadata(name="Organic Remains", char="X", base_color=(101, 67, 33, 255), passable=True, resource_type=ResourceType.ORGANIC_MATTER, energy_value=8.0)

    # Fallback/Empty
    VOID = TileMetadata(name="Void", char=" ", base_color=(0, 0, 0, 255), passable=False)

    @property
    def metadata(self) -> TileMetadata:
        return self.value

    @classmethod
    def get_all_tiles(cls) -> List['Tile']:
        return list(cls)

# Example usage:
# grassland_tile = Tile.GRASSLAND
# print(f"Tile: {grassland_tile.name}, Color: {grassland_tile.metadata.base_color}, Passable: {grassland_tile.metadata.passable}")
# print(f"Resource: {grassland_tile.metadata.resource_type.name}") 