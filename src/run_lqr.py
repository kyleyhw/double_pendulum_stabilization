"""
Interactive demo of the model-based controller: designed swing-up from
hanging (trajectory optimisation + TVLQR tracking) followed by LQR balance.
See ``src/control/swingup.py``.

    python src/run_lqr.py                    # swing up from hanging, then balance
    python src/run_lqr.py --reset_mode up    # balance only
    python src/run_lqr.py --save_gif
    python src/run_lqr.py --switching        # keys 1-4: DD / UU / DU / UD

Press LEFT/RIGHT to push the cart while it balances. With ``--switching`` the
controller holds one of the four equilibria (``src/control/switching.py``);
press 1 (down-down), 2 (up-up), 3 (down-up) or 4 (up-down) to move to another.
With ``--save_gif`` in switching mode it tours DD -> UU -> DU -> UD -> DD by itself.
"""
import argparse
import os
import sys
import time

import pygame
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.control.swingup import PlantParams, SwingUpController  # noqa: E402
from src.control.switching import NAMES, SwitchingController, load_library  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402
from src.utils.visualizer import Visualizer  # noqa: E402
from tools.eval_swingup import DEFAULT_TRAJ, load_or_optimize  # noqa: E402

SWITCH_KEYS = {pygame.K_1: "DD", pygame.K_2: "UU", pygame.K_3: "DU", pygame.K_4: "UD"}
AUTO_TOUR = ["UU", "DU", "UD", "DD"]


def run_lqr(duration=15.0, save_gif=False, output_gif="docs/images/lqr_stabilization.gif",
            reset_mode="down", difficulty=1.0, seed=0, switching=False):
    # The controller outputs a force in newtons, so the env must use ForceControl
    # (the default VelocityControl would read it as a velocity command).
    env = DoublePendulumCartEnv(reset_mode=reset_mode, control_strategy=ForceControl())
    env.set_curriculum(difficulty)
    if switching:
        # The library was optimised for full difficulty (see tools/eval_switching.py).
        controller = SwitchingController(load_library())
        reset_mode = "down"
        env.reset_mode = "down"
        tour = list(AUTO_TOUR)
        dwell = 0
    else:
        traj = load_or_optimize(PlantParams.from_env(env), DEFAULT_TRAJ, reoptimize=False)
        controller = SwingUpController(traj)
    max_force = env.control_strategy.max_force

    viz = Visualizer(env)

    state, _ = env.reset(seed=seed)
    if switching:
        controller.reset()
    else:
        controller.reset(env.state)
    step = 0
    start_time = time.time()
    frames = []

    if switching:
        print("Holding down-down. Keys: 1=DD 2=UU 3=DU 4=UD (" + ", ".join(NAMES) + ").")
    else:
        print(f"Starting from '{reset_mode}'. Swing-up takes ~{traj.duration:.1f} s after settling.")
    print("Press LEFT/RIGHT to perturb the cart and test robustness.")

    try:
        while duration <= 0 or (time.time() - start_time < duration):
            # Handle User Input (Perturbations)
            keys = pygame.key.get_pressed()
            impulse = 0.0
            if keys[pygame.K_LEFT]:
                impulse = -10.0
            elif keys[pygame.K_RIGHT]:
                impulse = 10.0

            if impulse != 0:
                env.apply_impulse(impulse)

            if switching:
                for key, name in SWITCH_KEYS.items():
                    if keys[key]:
                        controller.request(name)
                if save_gif and controller.settled:
                    dwell += 1
                    if dwell * env.dt >= 2.0 and tour:
                        controller.request(tour.pop(0))
                        dwell = 0
                    elif not tour and dwell * env.dt >= 2.0:
                        break
                label = (f"hold {controller.current}" if controller.phase == "hold"
                         else f"{controller.current} -> {controller.target}")
            else:
                label = controller.phase

            action = controller.action(env.state, max_force)
            next_state, reward, terminated, truncated, _ = env.step(action)

            viz.render(env.state, force=float(action[0]) * max_force, external_force=impulse,
                       step=step, reward=reward, episode=label)

            if save_gif:
                str_data = pygame.image.tostring(viz.screen, 'RGBA')
                img = Image.frombytes('RGBA', (viz.width, viz.height), str_data)
                frames.append(img)

            step += 1

            if terminated or truncated:
                print(f"Episode finished at step {step}. Resetting.")
                if save_gif:
                    break
                env.reset()
                if switching:
                    controller.reset()
                else:
                    controller.reset(env.state)

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
        print("Simulation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the swing-up + LQR controller")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds")
    parser.add_argument("--save_gif", action="store_true", help="Save the run as a GIF")
    parser.add_argument("--output", type=str, default="docs/images/lqr_stabilization.gif", help="Output path for GIF")
    parser.add_argument("--reset_mode", type=str, default="down", choices=["down", "up"])
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--switching", action="store_true",
                        help="Equilibrium switching mode (keys 1-4)")

    args = parser.parse_args()

    run_lqr(args.duration, args.save_gif, args.output, args.reset_mode, args.difficulty, args.seed,
            args.switching)
