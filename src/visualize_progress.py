import os
import glob
import re
import argparse
import sys
import time
import pygame
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def get_checkpoints(log_dir):
    """Finds and sorts all .pth checkpoints in the log directory."""
    files = glob.glob(os.path.join(log_dir, "*.pth"))
    
    # Extract episode numbers
    checkpoints = []
    for f in files:
        match = re.search(r"_(\d+)\.pth$", f)
        if match:
            episode = int(match.group(1))
            checkpoints.append((episode, f))
            
    # Sort by episode
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def visualize_progress(log_dir="logs", num_samples=5, duration_per_clip=5.0, save_gif=False, output_gif="docs/images/training_montage.gif"):
    """
    Plays a sequence of checkpoints to show learning progress.
    """
    checkpoints = get_checkpoints(log_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {log_dir}")
        return

    # Select evenly spaced samples
    if len(checkpoints) > num_samples:
        indices = list(map(int, np.linspace(0, len(checkpoints) - 1, num_samples)))
        selected_checkpoints = [checkpoints[i] for i in indices]
    else:
        selected_checkpoints = checkpoints

    print(f"Found {len(checkpoints)} checkpoints. Showing {len(selected_checkpoints)} samples.")
    
    env = DoublePendulumCartEnv()
    viz = Visualizer(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    frames = []
    
    try:
        for i, (episode_num, model_path) in enumerate(selected_checkpoints):
            print(f"\n--- Playing Episode {episode_num} ---")
            print(f"Loading {model_path}")
            
            agent = PPOAgent(state_dim, action_dim)
            agent.load(model_path)
            agent.policy.eval()
            
            state, _ = env.reset()
            step = 0
            start_time = time.time()
            
            # Run clip
            while time.time() - start_time < duration_per_clip:
                action, _ = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Render with overlay text
                viz.render(state, force=action[0], step=step, reward=reward)
                
                # Overlay Episode Number
                # We can't easily modify render() without passing more args, so let's just rely on the console log
                # or if we want to burn it into the GIF, we'd need to draw on the surface here.
                # Let's draw it manually on the surface if saving GIF
                if save_gif:
                    font = pygame.font.SysFont("Arial", 24)
                    text = font.render(f"Episode: {episode_num}", True, (0, 0, 0))
                    viz.screen.blit(text, (viz.width // 2 - 50, 50))
                    pygame.display.flip() # Update display with text
                    
                    # Capture frame
                    # Pygame surface to string buffer (RGBA)
                    str_data = pygame.image.tostring(viz.screen, 'RGBA')
                    img = Image.frombytes('RGBA', (viz.width, viz.height), str_data)
                    frames.append(img)
                
                state = next_state
                step += 1
                
                if terminated or truncated:
                    state, _ = env.reset()
            
            # Pause briefly between clips (add black frames or just repeat last frame?)
            if save_gif and i < len(selected_checkpoints) - 1:
                # Add a few frames of pause
                for _ in range(10):
                    frames.append(frames[-1])
            
            if not save_gif:
                time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        viz.close()
        
        if save_gif and frames:
            print(f"Saving GIF to {output_gif}...")
            os.makedirs(os.path.dirname(output_gif), exist_ok=True)
            # Save as GIF
            frames[0].save(
                output_gif,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=33, # ~30 fps
                loop=0
            )
            print("GIF saved.")

if __name__ == "__main__":
    import numpy as np # Imported here to avoid top-level dependency if not running
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing .pth checkpoints")
    parser.add_argument("--samples", type=int, default=5, help="Number of checkpoints to play")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration per clip in seconds")
    parser.add_argument("--save_gif", action="store_true", help="Save the montage as a GIF")
    parser.add_argument("--output", type=str, default="docs/images/training_montage.gif", help="Output path for GIF")
    
    args = parser.parse_args()
    
    visualize_progress(args.log_dir, args.samples, args.duration, args.save_gif, args.output)
