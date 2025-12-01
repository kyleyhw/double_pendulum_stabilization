import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env.double_pendulum import DoublePendulumCartEnv
from utils.visualizer import Visualizer

def test_visualizer():
    env = DoublePendulumCartEnv()
    viz = Visualizer(env)
    
    state, _ = env.reset()
    
    print("Running visualizer test for 5 seconds...")
    start_time = time.time()
    
    step = 0
    while time.time() - start_time < 5.0:
        # Random action
        action = env.action_space.sample()
        
        # Step
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Render
        viz.render(state, force=action[0], step=step, reward=reward)
        
        state = next_state
        step += 1
        
        if terminated or truncated:
            state, _ = env.reset()
            
    viz.close()
    print("Visualizer test complete.")

if __name__ == "__main__":
    test_visualizer()
