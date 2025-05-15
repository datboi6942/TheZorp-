"""Main entry point for ZorpLife.

This script initializes and runs the ZorpLife simulation.
"""

from zorplife.core.engine import Engine
from zorplife.config import GameConfig

def main() -> None:
    """Initializes and runs the ZorpLife simulation."""
    print("Initializing ZorpLife Engine...")
    config = GameConfig()
    engine = Engine(config)
    print("Starting ZorpLife Engine...")
    engine.run()
    print("ZorpLife Engine finished.")

if __name__ == "__main__":
    main() 