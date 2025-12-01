import torch
import numpy as np
import sys
import os
import glob
import pygame
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.double_pendulum import DoublePendulumCartEnv
from src.agent.ppo import PPOAgent
from src.utils.visualizer import Visualizer

def visualize_overlay(log_dir="logs", num_checkpoints=5, duration=10.0, save_gif=False, output_gif="docs/images/overlay_montage.gif"):
    # 1. Find Checkpoints
    checkpoints = glob.glob(os.path.join(log_dir, "ppo_*_*.pth"))
    # Filter out 'final' if we want specific steps, or include it.
    # Let's sort by episode number.
    # Filename format: ppo_YYYYMMDD_HHMMSS_EPISODE.pth
    
    def get_episode(f):
        try:
            return int(f.split('_')[-1].split('.')[0])
        except:
            return 999999 # Final

    checkpoints.sort(key=get_episode)
    
    if len(checkpoints) == 0:
        print("No checkpoints found.")
        return

    # Select N evenly spaced checkpoints
    if len(checkpoints) > num_checkpoints:
        indices = np.linspace(0, len(checkpoints)-1, num_checkpoints, dtype=int)
        selected_checkpoints = [checkpoints[i] for i in indices]
    else:
        selected_checkpoints = checkpoints

    print(f"Selected checkpoints: {[os.path.basename(c) for c in selected_checkpoints]}")

    # 2. Initialize Environments and Agents
    envs = []
    agents = []
    states = []
    
    # Common Env for parameters
    # Note: Use reset_mode="down" if that's what we trained on!
    # We should probably detect or default to "down" since that's the latest task.
    base_env = DoublePendulumCartEnv(reset_mode="down")
    viz = Visualizer(base_env)
    
    state_dim = base_env.observation_space.shape[0]
    action_dim = base_env.action_space.shape[0]

    for cp in selected_checkpoints:
        env = DoublePendulumCartEnv(reset_mode="down")
        state, _ = env.reset(seed=42) # Same seed for all to compare behavior!
        envs.append(env)
        states.append(state)
        
        agent = PPOAgent(state_dim, action_dim)
        agent.load(cp)
        agent.policy.eval()
        agents.append(agent)

    # Colors: Fade from Red (early) to Green (late)
    # Or just use different alphas.
    # Let's use Blue for all but vary alpha.
    # Or: Early = Red, Mid = Yellow, Late = Green.
    
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('viridis') # Blue to Green/Yellow
    colors = [cmap(i) for i in np.linspace(0, 1, len(selected_checkpoints))]
    # Convert to 0-255 RGB
    pygame_colors = [(int(r*255), int(g*255), int(b*255)) for r,g,b,a in colors]

    # 3. Simulation Loop
    frames = []
    clock = pygame.time.Clock()
    
    steps = int(duration * 60)
    
    print("Starting overlay simulation...")
    
    for step in range(steps):
        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update all agents
        for i in range(len(agents)):
            action, _ = agents[i].select_action(states[i])
            next_state, _, terminated, truncated, _ = envs[i].step(action)
            states[i] = next_state
            
            if terminated or truncated:
                states[i], _ = envs[i].reset(seed=42) # Reset to same start

        # Render
        viz.draw_background()
        
        # Draw all pendulums
        for i, state in enumerate(states):
            # Alpha: Make earlier ones more transparent? Or later ones?
            # Let's make all somewhat transparent so they blend.
            alpha = 100 if i < len(states) - 1 else 255 # Last one (best) is solid
            
            color = pygame_colors[i]
            viz.draw_pendulum(state, color_p1=color, color_p2=color, alpha=alpha)

        # Draw Legend with Background
        legend_x = 10
        legend_y = 10
        padding = 5
        line_height = 25
        
        # Background Box
        bg_rect = pygame.Rect(legend_x, legend_y, 120, len(selected_checkpoints) * line_height + 30)
        s = pygame.Surface((bg_rect.width, bg_rect.height))
        s.set_alpha(200)
        s.fill(viz.WHITE)
        viz.screen.blit(s, (legend_x, legend_y))
        
        # Title
        title = viz.font.render("Progress", True, viz.BLACK)
        viz.screen.blit(title, (legend_x + padding, legend_y + padding))
        
        for i, cp in enumerate(selected_checkpoints):
            ep_num = get_episode(cp)
            color = pygame_colors[i]
            
            # Color Box
            box_y = legend_y + 30 + i * line_height
            pygame.draw.rect(viz.screen, color, (legend_x + padding, box_y, 15, 15))
            
            # Text
            text = viz.font.render(f"Ep {ep_num}", True, viz.BLACK)
            viz.screen.blit(text, (legend_x + padding + 20, box_y))

        pygame.display.flip()
        
        if save_gif:
            str_data = pygame.image.tostring(viz.screen, 'RGBA')
            img = Image.frombytes('RGBA', (viz.width, viz.height), str_data)
            frames.append(img)
            
        clock.tick(60)

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

if __name__ == "__main__":
    visualize_overlay(save_gif=True)
