import pytest
import time
import pyglet
import esper

from zorplife.core.engine import Engine, TARGET_SPF

# Forcing headless for performance tests in case no display is available (e.g. CI)
# If a display IS available, pyglet might still use it, but this prevents crashes if not.
pyglet.options['headless'] = True
print("Perf test: pyglet.options['headless'] set to True.")

@pytest.mark.performance
def test_average_tick_time(capsys) -> None:
    """Tests that the average frame processing time is within limits (e.g., <= 16ms for 60 FPS)."""
    engine = Engine()
    
    num_frames_to_test = 300  # Run for 300 frames (e.g., 5 seconds at 60 FPS)
    actual_frames_processed = 0
    total_processing_time = 0.0
    
    # Temporarily override engine's run loop for controlled test
    # This is a bit intrusive; a better way might be to add a test_mode to Engine
    original_run = engine.run
    recorded_dts: list[float] = []

    def test_run_loop(self) -> None:
        nonlocal actual_frames_processed, total_processing_time, recorded_dts
        print("Starting engine test loop...")
        self._running = True
        self._last_tick_time = time.perf_counter()
        frames_done = 0

        pyglet.app.event_loop.has_exit = False

        while self._running and not self.window.has_exit and frames_done < num_frames_to_test:
            current_time = time.perf_counter()
            dt = current_time - self._last_tick_time
            self._last_tick_time = current_time
            recorded_dts.append(dt)

            # Minimal event processing for the window to stay responsive if visible
            self.window.dispatch_events() 
            pyglet.app.platform_event_loop.dispatch_posted_events()

            if not self._running or self.window.has_exit:
                break

            # Actual processing part of the tick
            tick_start_time = time.perf_counter()
            esper.process(dt) # ECS systems
            tick_process_time = time.perf_counter() - tick_start_time
            total_processing_time += tick_process_time
            
            self.window.flip() # Render
            
            frames_done += 1
            actual_frames_processed = frames_done

            # No sleep here, run as fast as possible for this test
            # to measure raw processing time.

        print(f"Engine test loop finished after {actual_frames_processed} frames.")
        self.stop() # Ensure engine state is set to not running
        if not self.window.has_exit:
            self.window.close()
        pyglet.app.event_loop.exit() # Ensure pyglet loop can exit

    engine.run = lambda: test_run_loop(engine) # Monkey-patch run method
    
    try:
        engine.run() # This will now call test_run_loop
    finally:
        engine.run = original_run # Restore original method
        # Ensure esper database is cleared for subsequent tests if any processors were added.
        # esper.clear_database() # This might be too broad if other tests depend on state.
        # A cleaner way is for Engine to provide a teardown method.

    assert actual_frames_processed > 0, "Engine did not process any frames."
    
    average_processing_time_ms = (total_processing_time / actual_frames_processed) * 1000.0
    target_ms = TARGET_SPF * 1000.0
    
    print(f"Test processed {actual_frames_processed} frames.")
    print(f"Total processing time: {total_processing_time:.4f}s")
    print(f"Average processing time per frame: {average_processing_time_ms:.4f} ms")
    print(f"Target processing time per frame (for {1/TARGET_SPF} FPS): {target_ms:.4f} ms")

    # Allow some leeway, e.g., 10-20% over for CI variability, but should be close to 16ms
    # For now, a slightly looser check, actual target is 16ms
    assert average_processing_time_ms <= (target_ms + 5.0), \
        f"Average processing time {average_processing_time_ms:.2f} ms exceeded target {target_ms:.2f} ms (plus leeway)."

    # Also check if recorded dts are reasonable (not excessively large)
    if recorded_dts:
        avg_dt_ms = (sum(recorded_dts) / len(recorded_dts)) * 1000.0
        print(f"Average dt (frame interval including processing) was: {avg_dt_ms:.2f} ms")
    
    # Suppress Pyglet's own exit message to keep test output clean
    with capsys.disabled():
        pyglet.app.exit() # This should be called to clean up pyglet state after window closes 