"""Main entry point for ZorpLife.

This script initializes and runs the ZorpLife simulation.
"""

from zorplife.core.engine import Engine

def main() -> None:
    """Initializes and runs the ZorpLife simulation."""
    print("Initializing ZorpLife Engine...")
    engine = Engine()
    print("Starting ZorpLife Engine...")
    engine.run()
    print("ZorpLife Engine finished.")

if __name__ == "__main__":
    main() 