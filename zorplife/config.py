from dataclasses import dataclass

@dataclass
class GameConfig:
    """Configuration settings for the ZorpLife game engine."""
    map_width: int = 100
    map_height: int = 100
    tile_render_size: int = 32
    initial_zorp_population: int = 10
    map_seed: int = 42

    window_width: int = 1024
    window_height: int = 768
    headless_mode: bool = False
    pyglet_debug_gl: bool = False # Set to True for Pyglet OpenGL debugging

    camera_pan_speed: float = 300.0 # Pixels per second
    camera_zoom_speed: float = 0.1  # Zoom factor per scroll step

    target_fps: float = 60.0

    # Add other configuration parameters as needed
    # For example:
    # ollama_model: str = "llama2"
    # ollama_host: str = "http://localhost:11434"
    # cultural_db_path: str = "zorplife/data/cultural_memory.db"
    # max_agent_tokens: int = 512 