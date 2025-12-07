import numpy as np
import gymnasium as gym
from abc import ABC, abstractmethod

class ControlStrategy(ABC):
    r"""
    Abstract Base Class for Control Strategies.
    
    This class defines the interface for mapping a high-level action $a_t$ (from the policies)
    to a low-level physical control signal $u_t$ (Force in Newtons).
    
    Mathematical Formulation:
    -------------------------
    Let $a_t \in [-1, 1]$ be the normalized action from the agent.
    Let $s_t$ be the state vector.
    The function $f(a_t, s_t)$ computes the control input $u_t$:
    $$ u_t = f(a_t, s_t; \theta_{env}) $$
    """
    @abstractmethod
    def get_force(self, action: np.ndarray, state: np.ndarray, env_params: dict) -> float:
        r"""
        Computes the physical force to apply to the cart.

        Args:
            action (np.ndarray): The normalized action vector $a_t \in \mathbb{R}^d$.
                                 Typically $a_t \in [-1, 1]$.
            state (np.ndarray): The full system state vector $s_t$.
                                E.g., for Double Pendulum: $s_t = [x, \theta_1, \theta_2, \dot{x}, \dot{\theta}_1, \dot{\theta}_2]$.
            env_params (dict):  Dictionary of environment parameters (e.g., $dt, F_{max}$).

        Returns:
            float: The force $u_t$ in Newtons.
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """
        Returns the Gymnasium action space definition.
        """
        pass

class ForceControl(ControlStrategy):
    r"""
    Direct Force Control Strategy.
    
    This strategy maps the action directly to force, proportional to the maximum capability.
    
    $$ F = a_t \cdot F_{max} $$
    
    where $a_t \in [-1, 1]$ and $F_{max}$ is the maximum actuator force.
    """
    def __init__(self, max_force: float = 5000.0):
        self.max_force = max_force
        
    def get_force(self, action: np.ndarray, state: np.ndarray, env_params: dict) -> float:
        r"""
        Computes $F = clip(a_t, -1, 1) \cdot F_{max}$.
        """
        # Direct Force Mapping
        force = float(np.clip(action[0], -1.0, 1.0) * self.max_force)
        return force
        
    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

class VelocityControl(ControlStrategy):
    r"""
    Velocity Control Strategy (High-Gain P-Controller).
    
    This strategy interprets the action as a *target velocity* $v_{cmd}$ for the cart.
    The force is computed using a Proportional Controller to track this velocity.
    
    Mathematical Formulation:
    -------------------------
    1. Target Velocity:
       $$ v_{cmd} = a_t \cdot v_{max} $$
    
    2. Velocity Error:
       $$ e_v = v_{cmd} - \dot{x} $$
    
    3. Control Force (P-Control):
       $$ F = K_p \cdot e_v $$
       
    4. Actuator Saturation:
       $$ F_{applied} = clip(F, -F_{max}, F_{max}) $$
       
    Note:
        A very high gain ($K_p \approx 10000$) is used to approximate "ideal velocity source" behavior,
        often resulting in valid but rapid bang-bang oscillation.
    """
    def __init__(self, max_velocity: float = 10.0, gain: float = 10000.0):
        self.max_velocity = max_velocity
        self.gain = gain # High K_p for stiff control
        
    def get_force(self, action: np.ndarray, state: np.ndarray, env_params: dict) -> float:
        """
        Computes force using $F = K_p(v_{cmd} - v_{current})$.
        """
        # Velocity Control (High-Gain P-Controller)
        # target_v from action [-1, 1] mapped to [-max_vel, max_vel]
        target_v = np.clip(action[0], -1.0, 1.0) * self.max_velocity
        
        # Determine velocity index dynamically or fallback
        # Standard: Double=[..., x_dot, ...], Single=[..., x_dot, ...]
        idx = env_params.get('velocity_index', -1)
        if idx == -1:
             # Fallback logic based on state vector length
             if len(state) == 6: idx = 3 # Double Pendulum
             elif len(state) == 4: idx = 2 # Single Pendulum
             else: idx = -1
        
        if idx != -1:
            current_v = state[idx]
        else:
            current_v = 0.0 
            
        err = target_v - current_v
        force = self.gain * err
        
        max_f = env_params.get('max_force', 5000.0)
        force = np.clip(force, -max_f, max_f)
        
        return float(force)

    def get_action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
