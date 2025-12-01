import numpy as np
import argparse
import time
import sys
import os
import pygame
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.utils.visualizer import Visualizer
from src.control.lqr import LQRController

def run_lqr(duration=10.0, save_gif=False, output_gif="docs/images/lqr_stabilization.gif"):
    # LQR requires starting near the equilibrium (UP)
    env = DoublePendulumCartEnv(reset_mode="up", wind_std=0.0)
    
    # Initialize LQR Controller
    try:
        controller = LQRController(env)
        print("LQR Controller initialized successfully.")
        print(f"LQR Gain K:\n{controller.K}")
    except ImportError:
        print("Error: scipy is required for LQR. Please install it: pip install scipy")
        return

    viz = Visualizer(env)
    
    state, _ = env.reset()
    step = 0
    start_time = time.time()
    frames = []
    
    print("Starting LQR Stabilization...")
    print("The pendulum starts near the top (UP). LQR should hold it there.")
    print("Press LEFT/RIGHT to perturb the cart and test robustness.")
    
    try:
        while duration <= 0 or (time.time() - start_time < duration):
            # Get Action from LQR
            action = controller.get_action(state)
            
            # Clip action to max force (actuator saturation)
            # LQR assumes infinite authority, but real motor has limits.
            # If perturbation is too large, LQR will fail due to clipping.
            action = np.clip(action, -env.force_mag, env.force_mag)
            
            # Handle User Input (Perturbations)
            keys = pygame.key.get_pressed()
            impulse = 0.0
            if keys[pygame.K_LEFT]:
                impulse = -10.0
            elif keys[pygame.K_RIGHT]:
                impulse = 10.0
            
            if impulse != 0:
                env.apply_impulse(impulse)
            
            # Step Env
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Render
            viz.render(state, force=action[0], external_force=impulse, step=step, reward=reward, episode="LQR")
            
            if save_gif:
                str_data = pygame.image.tostring(viz.screen, 'RGBA')
                img = Image.frombytes('RGBA', (viz.width, viz.height), str_data)
                frames.append(img)
            
            state = next_state
            step += 1
            
            if terminated or truncated:
                print(f"Episode finished at step {step}. Resetting.")
                if save_gif:
                    break
                state, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        viz.close()
        if save_gif and frames:
            print(f"Saving GIF to {output_gif}...")
            os.makedirs(os.path.dirname(output_gif), exist_ok=True)
            frames[0].save(
                output_gif,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=16, # ~60 fps
                loop=0
            )
            print("GIF saved.")
        print("LQR Simulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LQR Controller")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--save_gif", action="store_true", help="Save the run as a GIF")
    parser.add_argument("--output", type=str, default="docs/images/lqr_stabilization.gif", help="Output path for GIF")
    
    args = parser.parse_args()
    
    run_lqr(args.duration, args.save_gif, args.output)
