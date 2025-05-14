import pytest
from typing import Dict
import numpy as np

from zorplife.world.mapgen import MapGenerator, DEFAULT_MAP_WIDTH, DEFAULT_MAP_HEIGHT
from zorplife.world.tiles import Tile, ResourceType

# Define expected approximate ratios for a standard 100x100 map with seed 42
# These are just example targets and would need to be derived from running the generator
# and deciding on desired balance. For now, we mostly test that counts are non-zero and deterministic.
EXPECTED_MIN_TILES = (DEFAULT_MAP_WIDTH * DEFAULT_MAP_HEIGHT) * 0.01 # Expect at least 1% of any major biome

@pytest.mark.mapgen
def test_map_generation_determinism_and_counts() -> None:
    """Tests that map generation is deterministic and produces reasonable biome/resource counts."""
    seed = 42
    map_width = 50  # Use a smaller map for faster testing
    map_height = 50
    total_tiles = map_width * map_height

    # Generate map 1
    map_gen1 = MapGenerator(width=map_width, height=map_height, seed=seed)
    map_gen1.generate_map()
    map_data1 = map_gen1.get_map_data_for_rendering()
    biome_counts1 = dict(map_gen1.biome_counts) # Copy for comparison
    resource_counts1 = dict(map_gen1.resource_counts)

    assert map_data1 is not None, "Map data should be generated."
    assert map_data1.shape == (map_height, map_width), "Map dimensions are incorrect."
    assert sum(biome_counts1.values()) == total_tiles, "Sum of biome tiles does not match total tiles."

    # Check that some major biomes exist (very basic check)
    # A more robust test would check proportions against a defined configuration.
    assert biome_counts1.get(Tile.GRASSLAND, 0) > 0, "Grassland biome not generated."
    assert biome_counts1.get(Tile.WATER_SHALLOW, 0) > 0 or biome_counts1.get(Tile.WATER_DEEP, 0) > 0, "Water biomes not generated."
    # Check at least one primary resource type is present
    assert len(resource_counts1) > 0, "No resources generated."
    assert resource_counts1.get(ResourceType.PLANT_MATTER, 0) > 0, "Plant matter not generated."

    # Generate map 2 with the same seed
    map_gen2 = MapGenerator(width=map_width, height=map_height, seed=seed)
    map_gen2.generate_map()
    map_data2 = map_gen2.get_map_data_for_rendering()
    biome_counts2 = map_gen2.biome_counts
    resource_counts2 = map_gen2.resource_counts

    # Test for determinism
    assert biome_counts1 == biome_counts2, "Biome counts are not deterministic for the same seed."
    assert resource_counts1 == resource_counts2, "Resource counts are not deterministic for the same seed."
    # Compare tile data directly (can be slow for large maps, but good for smaller test maps)
    # np.array_equal is needed for numpy arrays of objects.
    assert map_data1 is not None and map_data2 is not None # Ensure type checker happy
    assert np.array_equal(map_data1, map_data2), "Generated map tile data is not deterministic."

    print(f"Test Seed {seed} Biome Counts: {{k.name: v for k,v in biome_counts1.items()}}")
    print(f"Test Seed {seed} Resource Counts: {{k.name: v for k,v in resource_counts1.items()}}")

@pytest.mark.mapgen
def test_resource_spawning_logic() -> None:
    """Validates that resources are spawned according to Tile metadata rules."""
    seed = 123
    map_gen = MapGenerator(width=30, height=30, seed=seed)
    map_gen.generate_map()

    generated_resource_tiles: Dict[ResourceType, int] = {}
    for r_idx in range(map_gen.height):
        for c_idx in range(map_gen.width):
            tile = map_gen.get_tile_at(c_idx, r_idx)
            resource = tile.metadata.resource_type
            if resource != ResourceType.NONE:
                generated_resource_tiles[resource] = generated_resource_tiles.get(resource, 0) + 1
    
    # Compare counts from iterating map vs. map_gen internal counts
    # This also verifies the resource_counts attribute is correctly populated.
    assert generated_resource_tiles == map_gen.resource_counts, \
        "Resource counts from direct map scan do not match MapGenerator's internal counts."

    # Example: Ensure if TREE tiles exist, WOOD resource count is positive
    if map_gen.biome_counts.get(Tile.TREE, 0) > 0:
        assert map_gen.resource_counts.get(ResourceType.WOOD, 0) > 0, "Trees exist but no wood resource counted."
    # Example: Ensure if CLIFF_ROCK tiles exist, STONE resource count is positive
    if map_gen.biome_counts.get(Tile.CLIFF_ROCK, 0) > 0:
        assert map_gen.resource_counts.get(ResourceType.STONE, 0) > 0, "Cliffs exist but no stone resource counted."

    # The "+-5% of config" would require a config file defining target percentages per biome/resource.
    # For now, this test focuses on consistency and presence. 