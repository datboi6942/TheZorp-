import noise # Perlin/Simplex noise
import random
from typing import List, Tuple, Dict, Optional
import numpy as np

from zorplife.world.tiles import Tile, ResourceType

# Map Generation Parameters
DEFAULT_MAP_WIDTH = 100  # In tiles
DEFAULT_MAP_HEIGHT = 100 # In tiles
DEFAULT_SEED = 0

# Noise parameters - these will require tuning
ELEVATION_SCALE = 50.0  # Lower = more zoomed in, features are larger
ELEVATION_OCTAVES = 4
ELEVATION_PERSISTENCE = 0.5
ELEVATION_LACUNARITY = 2.0

MOISTURE_SCALE = 30.0
MOISTURE_OCTAVES = 4
MOISTURE_PERSISTENCE = 0.5
MOISTURE_LACUNARITY = 2.0

# Thresholds for biome determination (values from -1.0 to 1.0 for pnoise2/snoise2)
# These will need significant tuning based on noise output
WATER_LEVEL_DEEP = -0.1  # Adjusted based on observed noise range (min_elev ~ -0.22)
WATER_LEVEL_SHALLOW = -0.05 # Adjusted
CLIFF_LEVEL = 0.20          # Adjusted (max_elev ~ 0.32)
MOUNTAIN_SLOPE_LEVEL = 0.25 # Adjusted
MOUNTAIN_PEAK_LEVEL = 0.30    # Adjusted, will be very top parts of map
FOREST_MOISTURE = 0.1         # Keep or adjust slightly based on moisture range (max_moist ~ 0.42)
FOREST_HEAVY_MOISTURE = 0.25  # Adjusted (max_moist ~ 0.42)

# Noise parameters for specific resources - refined based on testing
TREE_SPAWN_THRESHOLD = 0.1
ROCK_OUTCROP_SPAWN_THRESHOLD = 0.4
APPLE_TREE_SPAWN_THRESHOLD = 0.3
MUSHROOM_SPAWN_THRESHOLD = 0.1
IRON_DEPOSIT_SPAWN_THRESHOLD = 0.2
GOLD_DEPOSIT_SPAWN_THRESHOLD = 0.15 # Increased to make gold less abundant

class MapGenerator:
    """Generates a 2D tile-based game map using procedural noise."""

    def __init__(
        self,
        width: int = DEFAULT_MAP_WIDTH,
        height: int = DEFAULT_MAP_HEIGHT,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the map generator.

        Args:
            width: Width of the map in tiles.
            height: Height of the map in tiles.
            seed: Seed for random number generation to ensure deterministic maps.
                  If None, a random seed will be used (or system time based).
        """
        self.width = width
        self.height = height
        if seed is None:
            # For true determinism in testing, always provide a seed.
            # For gameplay, random.randint(0, 1_000_000) or similar is fine.
            self.seed = DEFAULT_SEED 
        else:
            self.seed = seed
        
        # Initialize internal map representation
        # Using NumPy array for easier manipulation with noise, then convert to Tile enum.
        # self.tile_grid: List[List[Tile]] = [] 
        self.tile_grid_np: Optional[np.ndarray] = None # Will hold Tile enum values

        # For biome stats, resource counts etc.
        self.biome_counts: Dict[Tile, int] = {} 
        self.resource_counts: Dict[ResourceType, int] = {}

    def _generate_noise_map(self, scale: float, octaves: int, persistence: float, lacunarity: float, offset_x: float, offset_y: float) -> np.ndarray:
        """Generates a 2D noise map using pnoise2."""
        noise_map = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                # Adding seed-based offsets to ensure different noise patterns per seed/map type
                nx = (x / scale) + offset_x
                ny = (y / scale) + offset_y
                noise_map[y][x] = noise.pnoise2(
                    nx,
                    ny,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    repeatx=1024, # Large repeat values to avoid obvious tiling on smaller maps
                    repeaty=1024,
                    base=self.seed # base is equivalent to seed for pnoise2
                )
        return noise_map

    def generate_map(self) -> None:
        """Generates the full tile map based on noise functions and rules."""
        print(f"Generating map ({self.width}x{self.height}) with seed: {self.seed}")

        # Seed-dependent offsets to ensure different noise types (elevation, moisture)
        # use different sections of the noise space even with the same base seed.
        # A common practice is to use a different 'base' for each noise type,
        # or large offsets. Here, using self.seed as base for all, and adding offsets.
        elevation_offset_x = self.seed * 1.1 
        elevation_offset_y = self.seed * 1.3

        moisture_offset_x = self.seed * 1.7 + 1000 # Add a large constant to further separate
        moisture_offset_y = self.seed * 1.9 + 1000

        elevation_map = self._generate_noise_map(
            ELEVATION_SCALE, ELEVATION_OCTAVES, ELEVATION_PERSISTENCE, ELEVATION_LACUNARITY,
            elevation_offset_x, elevation_offset_y
        )
        moisture_map = self._generate_noise_map(
            MOISTURE_SCALE, MOISTURE_OCTAVES, MOISTURE_PERSISTENCE, MOISTURE_LACUNARITY,
            moisture_offset_x, moisture_offset_y
        )

        # print(f"Elevation map range: min={np.min(elevation_map):.4f}, max={np.max(elevation_map):.4f}") # Commented out after initial check
        # print(f"Moisture map range:  min={np.min(moisture_map):.4f}, max={np.max(moisture_map):.4f}") # Commented out

        # Initialize tile_grid_np with a default type (e.g., VOID)
        # This allows direct assignment of Tile enum members.
        self.tile_grid_np = np.full((self.height, self.width), Tile.VOID, dtype=object)

        self.biome_counts.clear()
        self.resource_counts.clear()

        for y in range(self.height):
            for x in range(self.width):
                elev = elevation_map[y][x]
                moist = moisture_map[y][x]
                tile_type: Tile

                if elev < WATER_LEVEL_DEEP:
                    tile_type = Tile.WATER_DEEP
                elif elev < WATER_LEVEL_SHALLOW:
                    tile_type = Tile.WATER_SHALLOW
                elif elev >= MOUNTAIN_PEAK_LEVEL:
                    tile_type = Tile.MOUNTAIN_PEAK
                elif elev >= MOUNTAIN_SLOPE_LEVEL:
                    tile_type = Tile.MOUNTAIN_SLOPE
                elif elev > CLIFF_LEVEL:
                    tile_type = Tile.CLIFF_ROCK
                else: # Land
                    # Determine land biome based on moisture and remaining elevation
                    if moist > FOREST_HEAVY_MOISTURE and elev < CLIFF_LEVEL - 0.1: # Ensure forest is not right at cliff edge
                        tile_type = Tile.FOREST_HEAVY
                    elif moist > FOREST_MOISTURE and elev < CLIFF_LEVEL - 0.1:
                        tile_type = Tile.FOREST_FLOOR
                    else:
                        tile_type = Tile.GRASSLAND
                
                # Riverbank_Clay: if near shallow water and not water itself, and on land
                if tile_type != Tile.WATER_SHALLOW and tile_type != Tile.WATER_DEEP:
                    is_near_water = False
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx == 0 and dy == 0: continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.height and 0 <= nx < self.width:
                                neighbor_elev = elevation_map[ny][nx]
                                if WATER_LEVEL_DEEP <= neighbor_elev < WATER_LEVEL_SHALLOW:
                                    is_near_water = True
                                    break
                        if is_near_water: break
                    if is_near_water and elev < CLIFF_LEVEL - 0.2: # Ensure not too high up
                        tile_type = Tile.RIVERBANK_CLAY

                self.tile_grid_np[y][x] = tile_type
                
                # Update counts
                self.biome_counts[tile_type] = self.biome_counts.get(tile_type, 0) + 1
                tile_resource = tile_type.metadata.resource_type
                if tile_resource != ResourceType.NONE:
                    self.resource_counts[tile_resource] = self.resource_counts.get(tile_resource, 0) + 1
        
        # Second pass for distinct resource entities like Trees and Rock Outcrops
        # This needs to be careful not to overwrite important terrain features unless intended.
        # For simplicity, we'll place these based on the generated terrain.
        
        # Example: Placing TREEs on FOREST_FLOOR tiles based on another noise layer or probability
        # For now, the TREE tile type in tiles.py implies a tree.
        # If we want a grid of base terrain and then entities on top, that's a different structure.
        # The plan says "Tile enum with metadata (passable, resource_type...)"
        # And "Resource spawner seeds: Wood on tree tiles"
        # This implies some tiles ARE trees. So, mapgen should produce TREE tiles.

        tree_noise_scale = 20.0
        tree_offset_x = self.seed * 2.3 + 2000
        tree_offset_y = self.seed * 2.5 + 2000
        tree_noise_map = self._generate_noise_map(
            tree_noise_scale, 3, 0.6, 2.0, tree_offset_x, tree_offset_y
        )

        rock_outcrop_noise_scale = 15.0
        rock_outcrop_offset_x = self.seed * 2.7 + 3000
        rock_outcrop_offset_y = self.seed * 2.9 + 3000
        rock_outcrop_noise_map = self._generate_noise_map(
            rock_outcrop_noise_scale, 3, 0.5, 2.0, rock_outcrop_offset_x, rock_outcrop_offset_y
        )

        apple_tree_noise_scale = 22.0 # Slightly different from normal trees
        apple_tree_offset_x = self.seed * 3.1 + 4000
        apple_tree_offset_y = self.seed * 3.3 + 4000
        apple_tree_noise_map = self._generate_noise_map(
            apple_tree_noise_scale, 3, 0.6, 2.0, apple_tree_offset_x, apple_tree_offset_y
        )

        mushroom_noise_scale = 18.0
        mushroom_offset_x = self.seed * 3.5 + 5000
        mushroom_offset_y = self.seed * 3.7 + 5000
        mushroom_noise_map = self._generate_noise_map(
            mushroom_noise_scale, 3, 0.5, 2.0, mushroom_offset_x, mushroom_offset_y
        )

        iron_deposit_noise_scale = 25.0
        iron_deposit_offset_x = self.seed * 3.9 + 6000
        iron_deposit_offset_y = self.seed * 4.1 + 6000
        iron_deposit_noise_map = self._generate_noise_map(
            iron_deposit_noise_scale, 4, 0.5, 2.0, iron_deposit_offset_x, iron_deposit_offset_y # More octaves for irregularity
        )

        gold_deposit_noise_scale = 30.0 # Gold veins might be larger scale features
        gold_deposit_offset_x = self.seed * 4.3 + 7000
        gold_deposit_offset_y = self.seed * 4.5 + 7000
        gold_deposit_noise_map = self._generate_noise_map(
            gold_deposit_noise_scale, 2, 0.7, 2.0, gold_deposit_offset_x, gold_deposit_offset_y # Fewer octaves, higher persistence for rarer, larger patches
        )
        # print(f"Gold noise map range: min={np.min(gold_deposit_noise_map):.4f}, max={np.max(gold_deposit_noise_map):.4f}") # DEBUG REMOVED

        for y in range(self.height):
            for x in range(self.width):
                current_tile = self.tile_grid_np[y][x]
                original_resource = current_tile.metadata.resource_type

                # Function to safely decrement counts
                def decrement_counts(tile_to_decrement: Tile, resource_to_decrement: ResourceType):
                    if tile_to_decrement in self.biome_counts:
                        self.biome_counts[tile_to_decrement] -= 1
                        if self.biome_counts[tile_to_decrement] == 0: del self.biome_counts[tile_to_decrement]
                    if resource_to_decrement != ResourceType.NONE and resource_to_decrement in self.resource_counts:
                        self.resource_counts[resource_to_decrement] -= 1
                        if self.resource_counts[resource_to_decrement] == 0: del self.resource_counts[resource_to_decrement]
                
                # Try to place Trees on FOREST_FLOOR or FOREST_HEAVY
                if (current_tile == Tile.FOREST_FLOOR or current_tile == Tile.FOREST_HEAVY) and tree_noise_map[y][x] > TREE_SPAWN_THRESHOLD:
                    decrement_counts(current_tile, original_resource)
                    self.tile_grid_np[y][x] = Tile.TREE_TRUNK
                    self.biome_counts[Tile.TREE_TRUNK] = self.biome_counts.get(Tile.TREE_TRUNK, 0) + 1
                    self.resource_counts[ResourceType.WOOD] = self.resource_counts.get(ResourceType.WOOD, 0) + 1
                    # Place canopy above if in bounds and not already a tree/canopy
                    if y + 1 < self.height:
                        above_tile = self.tile_grid_np[y+1][x]
                        if above_tile not in (Tile.TREE_TRUNK, Tile.TREE_CANOPY):
                            # Decrement counts for the tile being replaced by canopy
                            above_tile_original_resource = above_tile.metadata.resource_type
                            decrement_counts(above_tile, above_tile_original_resource)

                            self.tile_grid_np[y+1][x] = Tile.TREE_CANOPY
                            self.biome_counts[Tile.TREE_CANOPY] = self.biome_counts.get(Tile.TREE_CANOPY, 0) + 1
                            # Canopy usually doesn't add new primary resources, handled by trunk

                # Try to place Apple Trees on FOREST_FLOOR or FOREST_HEAVY (rarer)
                elif (current_tile == Tile.FOREST_FLOOR or current_tile == Tile.FOREST_HEAVY) and apple_tree_noise_map[y][x] > APPLE_TREE_SPAWN_THRESHOLD:
                    decrement_counts(current_tile, original_resource)

                    # Place Apple Tree Trunk
                    self.tile_grid_np[y][x] = Tile.APPLE_TREE_TRUNK
                    self.biome_counts[Tile.APPLE_TREE_TRUNK] = self.biome_counts.get(Tile.APPLE_TREE_TRUNK, 0) + 1
                    # Apple tree trunks provide wood
                    self.resource_counts[ResourceType.WOOD] = self.resource_counts.get(ResourceType.WOOD, 0) + 1 

                    # Place Apple Tree Leaves above if in bounds
                    if y + 1 < self.height:
                        # Check if the tile above is suitable (e.g., not already a trunk or another structure part)
                        # For simplicity, we'll assume it can overwrite most passable tiles or be placed if it's air-like.
                        # A more complex check might be `above_tile.metadata.passable` or specific types.
                        above_tile_original_resource = self.tile_grid_np[y+1][x].metadata.resource_type
                        decrement_counts(self.tile_grid_np[y+1][x], above_tile_original_resource)
                        
                        self.tile_grid_np[y+1][x] = Tile.APPLE_TREE_LEAVES
                        self.biome_counts[Tile.APPLE_TREE_LEAVES] = self.biome_counts.get(Tile.APPLE_TREE_LEAVES, 0) + 1
                        self.resource_counts[ResourceType.APPLES] = self.resource_counts.get(ResourceType.APPLES, 0) + 1

                # Try to place Mushroom Patches on FOREST_FLOOR or FOREST_HEAVY
                # Ensure the current tile is still FOREST_FLOOR or FOREST_HEAVY 
                # (i.e., it wasn't just turned into a tree trunk by a preceding rule in this iteration)
                current_tile_for_mushroom_check = self.tile_grid_np[y][x] # Re-fetch, as it might have been changed by tree/apple tree placement
                if (current_tile_for_mushroom_check == Tile.FOREST_FLOOR or \
                    current_tile_for_mushroom_check == Tile.FOREST_HEAVY) and \
                   mushroom_noise_map[y][x] > MUSHROOM_SPAWN_THRESHOLD:
                    
                    original_resource_for_mushroom = current_tile_for_mushroom_check.metadata.resource_type
                    decrement_counts(current_tile_for_mushroom_check, original_resource_for_mushroom)
                    
                    self.tile_grid_np[y][x] = Tile.MUSHROOM_PATCH
                    self.biome_counts[Tile.MUSHROOM_PATCH] = self.biome_counts.get(Tile.MUSHROOM_PATCH, 0) + 1
                    self.resource_counts[ResourceType.MUSHROOMS] = self.resource_counts.get(ResourceType.MUSHROOMS, 0) + 1

                # Try to place Rock Outcrops on GRASSLAND or MOUNTAIN_SLOPE (less rocky parts)
                # This should still be an elif, as a tile can't be both a mushroom patch and a rock outcrop simultaneously
                # if it was converted to mushroom patch in the lines above.
                # However, the original logic was:
                # elif (current_tile == Tile.GRASSLAND or current_tile == Tile.MOUNTAIN_SLOPE) ...
                # We need to ensure this elif condition correctly uses the potentially updated tile type if a mushroom was placed.
                # OR, more simply, these distinct features (rock, iron, gold) should be checked against the *original* tile type *before* trees/mushrooms,
                # or their placement should be mutually exclusive by design.
                # For now, let's assume the original `current_tile` (from start of loop iteration) is intended for these.
                # This means the structure becomes:
                # if tree: ...
                # elif apple_tree: ...
                #
                # (independent)
                # current_tile_for_mushroom_check = self.tile_grid_np[y][x]
                # if mushroom_on_original_forest_base: ...
                #
                # (back to original logic, checking the tile state *before* any tree/mushroom placement for this iteration)
                elif (current_tile == Tile.GRASSLAND or current_tile == Tile.MOUNTAIN_SLOPE) and rock_outcrop_noise_map[y][x] > ROCK_OUTCROP_SPAWN_THRESHOLD:
                    decrement_counts(current_tile, original_resource)

                    self.tile_grid_np[y][x] = Tile.ROCK_OUTCROP
                    self.biome_counts[Tile.ROCK_OUTCROP] = self.biome_counts.get(Tile.ROCK_OUTCROP, 0) + 1
                    self.resource_counts[ResourceType.STONE] = self.resource_counts.get(ResourceType.STONE, 0) + 1

                # Try to place Iron Deposits on MOUNTAIN_SLOPE or CLIFF_ROCK
                elif (current_tile == Tile.MOUNTAIN_SLOPE or current_tile == Tile.CLIFF_ROCK) and iron_deposit_noise_map[y][x] > IRON_DEPOSIT_SPAWN_THRESHOLD:
                    decrement_counts(current_tile, original_resource)
                    
                    self.tile_grid_np[y][x] = Tile.IRON_DEPOSIT
                    self.biome_counts[Tile.IRON_DEPOSIT] = self.biome_counts.get(Tile.IRON_DEPOSIT, 0) + 1
                    self.resource_counts[ResourceType.IRON_ORE] = self.resource_counts.get(ResourceType.IRON_ORE, 0) + 1

                # Try to place Gold Deposits on MOUNTAIN_SLOPE or MOUNTAIN_PEAK (rarely, or deeper in cliffs)
                elif (current_tile == Tile.MOUNTAIN_SLOPE or current_tile == Tile.MOUNTAIN_PEAK):
                    # DEBUG: Print gold noise value for ALL mountain slope/peak tiles being considered
                    # print(f"GOLD CHECK: Tile ({x},{y}) is {current_tile.name}, Gold Noise: {gold_deposit_noise_map[y][x]:.4f}, Threshold: {GOLD_DEPOSIT_SPAWN_THRESHOLD}") # DEBUG REMOVED
                    if gold_deposit_noise_map[y][x] > GOLD_DEPOSIT_SPAWN_THRESHOLD:
                        # Gold is rare, ensure it does not overwrite other specific deposits if logic gets more complex
                        decrement_counts(current_tile, original_resource)
                        
                        self.tile_grid_np[y][x] = Tile.GOLD_DEPOSIT
                        self.biome_counts[Tile.GOLD_DEPOSIT] = self.biome_counts.get(Tile.GOLD_DEPOSIT, 0) + 1
                        self.resource_counts[ResourceType.GOLD_ORE] = self.resource_counts.get(ResourceType.GOLD_ORE, 0) + 1

        print("Map generation complete.")
        print("Biome counts:", {k.name: v for k,v in self.biome_counts.items()})
        print("Resource counts:", {k.name: v for k,v in self.resource_counts.items()})


    def get_tile_at(self, x: int, y: int) -> Tile:
        """Returns the tile type at the given map coordinates."""
        if self.tile_grid_np is None:
            raise RuntimeError("Map has not been generated yet. Call generate_map() first.")
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tile_grid_np[y][x] # Numpy array is (row, col) which is (y, x)
        return Tile.VOID # Out of bounds

    def get_map_data_for_rendering(self) -> Optional[np.ndarray]: # Return type is np.ndarray[Tile]
        """Returns the generated map data (numpy array of Tile enums)."""
        return self.tile_grid_np

# Example usage:
if __name__ == "__main__":
    map_gen = MapGenerator(width=50, height=30, seed=42)
    map_gen.generate_map()
    
    if map_gen.tile_grid_np is not None:
        for y in range(map_gen.height):
            row_str = ""
            for x in range(map_gen.width):
                row_str += map_gen.get_tile_at(x,y).metadata.char
            print(row_str)
        
        print("\nBiome Counts:")
        for tile_type, count in map_gen.biome_counts.items():
            print(f"  {tile_type.name}: {count}")

        print("\nResource Counts (from tile metadata):")
        for resource_type, count in map_gen.resource_counts.items():
            print(f"  {resource_type.name}: {count}") 