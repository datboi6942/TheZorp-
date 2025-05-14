"""Main entry point for ZorpLife.

This script initializes and runs the ZorpLife simulation.
"""

import pyglet
import time


def main() -> None:
    """Initializes and runs the ZorpLife simulation."""
    window = pyglet.window.Window(640, 480, "ZorpLife")

    @window.event
    def on_draw() -> None:
        window.clear()
        label = pyglet.text.Label(
            "hello",
            font_name="Arial",
            font_size=36,
            x=window.width // 2,
            y=window.height // 2,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()

    @window.event
    def on_key_press(symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            print("Escape pressed, exiting...")
            window.close()
            pyglet.app.exit()

    def exit_after_delay(dt: float) -> None:
        print(f"Exiting after {dt} seconds due to timer.")
        window.close()
        pyglet.app.exit()

    # Schedule exit after 1 second for the initial test
    pyglet.clock.schedule_once(exit_after_delay, 1.0)

    print("Starting ZorpLife... Opening window.")
    try:
        pyglet.app.run()
    finally:
        print("ZorpLife finished.")


if __name__ == "__main__":
    main() 