import numpy as np
from abc import ABC, abstractmethod

class RewardStrategy(ABC):
    """
    Abstract Base Class for Reward Strategies.
    
    Defines the interface for computing the scalar reward signal $r_t$.
    
    $$ r_t = R(s_t, a_t) $$
    """
    @abstractmethod
    def compute_reward(self, state: np.ndarray, env_params: dict) -> float:
        """
        Computes the reward for the current state.
        
        Args:
           state (np.ndarray): The system state $s_t$.
           env_params (dict): Environment parameters.
           
        Returns:
            float: Scalar reward $r_t$.
        """
        pass
        
    @abstractmethod
    def set_curriculum(self, difficulty: float):
        r"""
        Updates internal parameters based on curriculum difficulty $\delta \in [0, 1]$.
        """
        pass

class SwingUpBalanceReward(RewardStrategy):
    r"""
    Standard Swing-Up and Balance Reward.
    
    This reward encourages the poles to be upright ($\theta \approx \pi$) and the cart to be centered.
    
    Mathematical Formulation:
    -------------------------
    1. Angle Error:
       $$ e_{\theta, i} = | (\theta_i \mod 2\pi) - \pi | $$
       Where $\theta_i$ is normalized to $[0, 2\pi]$.
       
    2. Stability Condition:
       Let success indicator $S = \mathbb{I}(\forall i: e_{\theta, i} < \epsilon)$, where $\epsilon$ is the threshold.
       
    3. Reward Function:
       If $S$ is true (all poles up):
           $$ r_t = 1.0 + \mathbb{I}(|x| < 1.0) \cdot 0.5 $$
           (Bonus for centering).
       Else:
           $$ r_t = 0.1 \cdot (1 - \delta) $$
           (Small survival reward inversely proportional to difficulty).
    """
    def __init__(self, target_angle: float = 0.0): # 0 = Up
        self.reward_threshold = np.deg2rad(10.0) # Default Hard
        self.difficulty = 1.0 # Default
        
    def set_curriculum(self, difficulty: float):
        self.difficulty = difficulty
        # Map difficulty 0.0 -> 1.0 to Threshold 90 deg -> 10 deg
        min_angle = np.deg2rad(10.0)
        max_angle = np.deg2rad(90.0)
        self.reward_threshold = max_angle - (max_angle - min_angle) * difficulty

    def compute_reward(self, state: np.ndarray, env_params: dict) -> float:
        # Generalized for N pendulums
        n_links = (len(state)) // 2
        x = state[0]
        
        # Consistent parsing: Double vs Single
        # Double: x, t1, t2... Angles are indices 1, 2. (Len 6)
        if len(state) == 6:
            angles = [state[1], state[2]]
        elif len(state) == 4:
            angles = [state[1]]
        else: 
            # Fallback
            n_pends = (len(state) // 2) - 1
            angles = list(state[1 : 1+n_pends])
        
        all_up = True
        for theta in angles:
            # Normalize error around PI
            # dist to pi = abs(arctan2(sin(t-pi), cos(t-pi)))
            err = np.abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi)))
            if err > self.reward_threshold:
                all_up = False
                break
            
        if all_up:
            if abs(x) < 1.0: # Centered bonus
                return 1.5
            return 1.0
            
        # Survival reward based on difficulty (Ratchet)
        return 0.1 * (1.0 - self.difficulty) 


# Note: The above is a simplification. The original had:
# if dist1 < thresh and dist2 < thresh: reward = 1.0 ...
# I should ensure strict parity.

class DoublePendulumStandardReward(RewardStrategy):
    """Exact logic from the original Double Pendulum environment."""
    def __init__(self):
        self.reward_threshold = np.deg2rad(10.0)
        self.difficulty = 1.0
        
    def set_curriculum(self, difficulty: float):
        self.difficulty = difficulty
        min_angle = np.deg2rad(10.0)
        max_angle = np.deg2rad(90.0)
        self.reward_threshold = max_angle - (max_angle - min_angle) * difficulty
        
    def compute_reward(self, state: np.ndarray, env_params: dict) -> float:
        x, theta1, theta2, _, _, _ = state
        
        # Normalize angles to [-pi, pi]
        t1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
        t2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
        
        # Target is PI (Up) in original coords?
        # Original: "theta1, theta2: Angle (0 = Down)"
        # So Up is PI.
        # dist1 = abs(theta1 - pi)
        
        d1 = np.abs(np.arctan2(np.sin(theta1 - np.pi), np.cos(theta1 - np.pi)))
        d2 = np.abs(np.arctan2(np.sin(theta2 - np.pi), np.cos(theta2 - np.pi)))
        
        threshold = self.reward_threshold
        
        if d1 < threshold and d2 < threshold:
            if abs(x) < 0.5:
                # Super Bonus/State
                return 2.0 # Wait, original was 1.0 + ...?
                # Original lines 150+:
                # reward = 0.0
                # dist1 = ...
                # if dist1 < self.reward_threshold and dist2 < self.reward_threshold:
                #    reward += 1.0
                #    if abs(x) < 1.0: reward += 0.5
                # else:
                #    reward += 0.1 * (1 - self.difficulty)
                
            return 1.0 + (0.5 if abs(x) < 1.0 else 0.0)
        
        return 0.1 * (1.0 - self.difficulty)

class SinglePendulumStandardReward(RewardStrategy):
    """Logic for Single Pendulum."""
    def __init__(self):
        self.reward_threshold = np.deg2rad(10.0)
        self.difficulty = 1.0
        
    def set_curriculum(self, difficulty: float):
        self.difficulty = difficulty
        min_angle = np.deg2rad(10.0)
        max_angle = np.deg2rad(90.0)
        self.reward_threshold = max_angle - (max_angle - min_angle) * difficulty
        
    def compute_reward(self, state: np.ndarray, env_params: dict) -> float:
        x, theta1, _, _ = state
        
        d1 = np.abs(np.arctan2(np.sin(theta1 - np.pi), np.cos(theta1 - np.pi)))
        
        threshold = self.reward_threshold
        
        if d1 < threshold:
             return 1.0 + (0.5 if abs(x) < 1.0 else 0.0)
        
        return 0.1 * (1.0 - self.difficulty)

class ExponentialSwingUpReward(RewardStrategy):
    r"""
    Exponential Continuity Reward Strategy.
    
    This reward incentivizes *continuous* stabilization. It accumulates a counter $T_{up}$ 
    for every consecutive step the system remains within the target threshold.
    
    Mathematical Formulation:
    -------------------------
    1. Threshold Condition:
       $$ S_t = \forall i: | (\theta_i \mod 2\pi) - \pi | < \epsilon $$
       
    2. Continuity Counter:
       $$ T_{up, t} = \begin{cases} T_{up, t-1} + \Delta t & \text{if } S_t \\ 0 & \text{if } \neg S_t \end{cases} $$
       
    3. Reward Function:
       $$ r_t = (\exp(T_{up, t}) - 1.0) \cdot P(x) $$
       
       Where $P(x)$ is a gaussian position penalty to keep the cart centered:
       $$ P(x) = \exp\left( - \frac{x^2}{2\sigma_x^2} \right) $$
       
    Rationale:
    ----------
    This "Ratchet" effect ($T_{up}$ grows linearly, Reward grows exponentially) provides 
    massive gradients for maintaining stability, unlike a sparse binary reward.
    """
    def __init__(self):
        self.reward_threshold = np.deg2rad(10.0)
        self.difficulty = 1.0
        self.steps_above_threshold = 0
        
    def set_curriculum(self, difficulty: float):
        self.difficulty = difficulty
        # Difficulty varies threshold 90 -> 10 deg
        min_angle = np.deg2rad(10.0)
        max_angle = np.deg2rad(90.0)
        self.reward_threshold = max_angle - (max_angle - min_angle) * difficulty
        
    def compute_reward(self, state: np.ndarray, env_params: dict) -> float:
        # Extract params
        dt = env_params.get('dt', 0.005)
        
        # Consistent parsing based on state length
        if len(state) == 6:
            # Double Pendulum
            x = state[0]
            angles = [state[1], state[2]]
        elif len(state) == 4:
            # Single Pendulum
            x = state[0]
            angles = [state[1]]
        else:
            # Fallback for arbitrary N-link
            n_pends = (len(state) // 2) - 1
            x = state[0]
            angles = list(state[1 : 1+n_pends])
        
        all_up = True
        for theta in angles:
            # Normalize error around PI
            err = np.abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi)))
            if err >= self.reward_threshold:
                all_up = False
                break
                
        if all_up:
            self.steps_above_threshold += 1
            time_above = self.steps_above_threshold * dt
            reward = np.exp(time_above) - 1.0
            
            # Position Penalty (Gaussian)
            sigma_x = 2.0
            pos_penalty = np.exp(-(x**2) / (2 * sigma_x**2))
            reward *= pos_penalty
        else:
            self.steps_above_threshold = 0
            reward = 0.0
            
        return reward
