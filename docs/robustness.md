# Robustness & Perturbations

To validate the stability of the trained agent, we can introduce external disturbances.

## 1. Impulsive Forces (Push)
You can manually apply lateral forces to the cart during simulation to test if the agent can recover.

*   **Left Arrow**: Apply 10N force to the Left.
*   **Right Arrow**: Apply 10N force to the Right.

The external force is visualized as a **Magenta Arrow** above the cart.

## 2. Continuous Wind
You can simulate a noisy environment where the cart is constantly buffeted by random forces.

### Usage
```bash
# Run with wind (standard deviation of noise = 2.0)
python src/simulate.py --model logs/ppo_final.pth --wind 2.0
```

The wind force is added to the action at every timestep:
$$ F_{total} = F_{action} + \mathcal{N}(0, \sigma_{wind}) $$

## 3. Stress Testing
To quantify robustness, increase the `--wind` parameter until the agent fails to stabilize.
