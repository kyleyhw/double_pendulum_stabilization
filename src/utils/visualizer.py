import pygame
import numpy as np
import sys
import time
from datetime import datetime

class Visualizer:
    def __init__(self, env, headless=False):
        self.env = env
        self.headless = headless
        
        # Screen Dimensions
        self.width = 800
        self.height = 600
        self.scale = 70 # Pixels per meter (Fits 10m track in 800px)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Initialize Pygame
        pygame.init()
        
        if self.headless:
            # Offscreen rendering
            self.screen = pygame.Surface((self.width, self.height))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Double Pendulum Stabilization")
            
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        
        # Reward History
        self.reward_history = []
        self.max_history = 100 # Show last 100 steps (2 seconds at dt=0.02)
        
    def render(self, state, force=0.0, external_force=0.0, episode=0, step=0, reward=0.0, reward_fn_label="Reward Fn: SwingUp + Balance", seed=None, plot_histories=None, plot_colors=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        # Update Reward History
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
            
        self.draw_background()
        if state is not None:
            self.draw_pendulum(state)
        
        # Force Indicators
        ox = self.width / 2
        oy = self.height / 2
        cart_x = ox + state[0] * self.scale
        cart_y = oy
        
        if abs(force) > 0.1:
            force_len = force * 2
            start_pos = (int(cart_x), int(cart_y))
            end_pos = (int(cart_x + force_len), int(cart_y))
            pygame.draw.line(self.screen, self.GREEN, start_pos, end_pos, 4)

        if abs(external_force) > 0.1:
            ext_len = external_force * 2
            start_pos = (int(cart_x), int(cart_y - 30))
            end_pos = (int(cart_x + ext_len), int(cart_y - 30))
            pygame.draw.line(self.screen, (255, 0, 255), start_pos, end_pos, 4)
            label = self.font.render("Ext", True, (255, 0, 255))
            self.screen.blit(label, (int(cart_x), int(cart_y - 50)))

        # Draw Reward Plot
        if plot_histories is not None:
            self.draw_reward_plot(histories=plot_histories, colors=plot_colors)
        else:
            self.draw_reward_plot()

        # Info Text
        info_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"Force: {force:.2f} N",
            f"x: {state[0]:.2f} m",
            f"Theta1: {state[1]:.2f} rad",
            f"Theta2: {state[2]:.2f} rad",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"Seed: {seed}" if seed is not None else "",
            reward_fn_label
        ]
        
        for i, line in enumerate(info_text):
            text_surf = self.font.render(line, True, self.BLACK)
            self.screen.blit(text_surf, (10, 10 + i * 20))
            
        # Update Display
        if not self.headless:
            pygame.display.flip()
            self.clock.tick(60)
        x_label = self.font.render("x (m)", True, self.GRAY)
        y_label = self.font.render("y (m)", True, self.GRAY)
        self.screen.blit(x_label, (self.width - 40, int(oy) + 5))
        self.screen.blit(y_label, (int(ox) + 5, 5))
        
        # Draw Track
        pygame.draw.line(self.screen, self.BLACK, (0, int(oy)), (self.width, int(oy)), 2)

    def draw_background(self):
        self.screen.fill(self.WHITE)
        
        # Center of screen is (width/2, height/2)
        ox = self.width / 2
        oy = self.height / 2
        
        # Draw Coordinate System (Cartesian)
        for i in range(-5, 6):
            gx = int(ox + i * self.scale)
            pygame.draw.line(self.screen, (240, 240, 240), (gx, 0), (gx, self.height), 1)
            gy = int(oy + i * self.scale)
            pygame.draw.line(self.screen, (240, 240, 240), (0, gy), (self.width, gy), 1)

        # Main Axes
        pygame.draw.line(self.screen, self.GRAY, (int(ox), 0), (int(ox), self.height), 1)
        pygame.draw.line(self.screen, self.GRAY, (0, int(oy)), (self.width, int(oy)), 1)
        
        # Labels
        x_label = self.font.render("x (m)", True, self.GRAY)
        y_label = self.font.render("y (m)", True, self.GRAY)
        self.screen.blit(x_label, (self.width - 40, int(oy) + 5))
        self.screen.blit(y_label, (int(ox) + 5, 5))
        
        # Draw Track
        pygame.draw.line(self.screen, self.BLACK, (0, int(oy)), (self.width, int(oy)), 2)

    def draw_pendulum(self, state, color_p1=None, color_p2=None, alpha=255):
        if color_p1 is None: color_p1 = self.BLUE
        if color_p2 is None: color_p2 = self.RED
        
        if len(state) == 6:
            x, theta1, theta2, _, _, _ = state
        elif len(state) == 8:
            # [x, sin1, cos1, sin2, cos2, x_dot, t1_dot, t2_dot]
            x, s1, c1, s2, c2, _, _, _ = state
            theta1 = np.arctan2(s1, c1)
            theta2 = np.arctan2(s2, c2)
        else:
            raise ValueError(f"Invalid state dimension: {len(state)}")
        
        ox = self.width / 2
        oy = self.height / 2
        
        cart_x = ox + x * self.scale
        cart_y = oy
        
        l1 = self.env.l1 * self.scale
        l2 = self.env.l2 * self.scale
        
        p1_x = cart_x + l1 * np.sin(theta1)
        p1_y = cart_y + l1 * np.cos(theta1)
        
        p2_x = p1_x + l2 * np.sin(theta2)
        p2_y = p1_y + l2 * np.cos(theta2)
        
        # Create a surface for transparency if alpha < 255
        if alpha < 255:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # Cart
            cart_w = 0.6 * self.scale
            cart_h = 0.3 * self.scale
            cart_rect = pygame.Rect(int(cart_x - cart_w/2), int(cart_y - cart_h/2), int(cart_w), int(cart_h))
            pygame.draw.rect(s, (*self.BLACK, alpha), cart_rect)
            
            # Links
            pygame.draw.line(s, (*color_p1, alpha), (int(cart_x), int(cart_y)), (int(p1_x), int(p1_y)), 6)
            pygame.draw.circle(s, (*color_p1, alpha), (int(p1_x), int(p1_y)), 10)
            
            pygame.draw.line(s, (*color_p2, alpha), (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), 6)
            pygame.draw.circle(s, (*color_p2, alpha), (int(p2_x), int(p2_y)), 10)
            
            self.screen.blit(s, (0,0))
        else:
            # Cart
            cart_w = 0.6 * self.scale
            cart_h = 0.3 * self.scale
            cart_rect = pygame.Rect(int(cart_x - cart_w/2), int(cart_y - cart_h/2), int(cart_w), int(cart_h))
            pygame.draw.rect(self.screen, self.BLACK, cart_rect)
            
            # Links
            pygame.draw.line(self.screen, color_p1, (int(cart_x), int(cart_y)), (int(p1_x), int(p1_y)), 6)
            pygame.draw.circle(self.screen, color_p1, (int(p1_x), int(p1_y)), 10)
            
            pygame.draw.line(self.screen, color_p2, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), 6)
            pygame.draw.circle(self.screen, color_p2, (int(p2_x), int(p2_y)), 10)

    def draw_reward_plot(self, histories=None, colors=None):
        """
        Draws a rolling plot of reward history.
        Args:
            histories: List of lists of reward values. If None, uses [self.reward_history].
            colors: List of colors for each history. If None, uses [self.BLUE].
        """
        if histories is None:
            if not self.reward_history:
                return
            histories = [self.reward_history]
            colors = [self.BLUE]
            
        if not histories or not histories[0]:
            return
            
        # Plot Dimensions
        plot_w = 200
        plot_h = 100
        margin = 10
        # Top Right
        x_start = self.width - plot_w - margin
        y_start = margin + 20 # Add space for title
        
        # Background
        bg_rect = pygame.Rect(x_start, y_start, plot_w, plot_h)
        pygame.draw.rect(self.screen, (240, 240, 240), bg_rect)
        pygame.draw.rect(self.screen, self.GRAY, bg_rect, 1) # Border
        
        # Determine Scale (Global max across all histories)
        all_values = [r for h in histories for r in h]
        if not all_values: return
        
        max_r = max(all_values)
        min_r = min(all_values)
        
        # Add padding
        r_range = max_r - min_r
        if r_range < 0.1:
            max_r += 0.1
            min_r -= 0.0 # Keep min at 0 if possible, or adjust
            
        # Ensure baseline 0 is visible if values are positive
        if min_r > 0: min_r = 0
        
        # Draw each history
        for idx, history in enumerate(histories):
            color = colors[idx] if colors and idx < len(colors) else self.BLUE
            
            points = []
            for i, r in enumerate(history):
                # x coordinate
                px = x_start + (i / (self.max_history - 1)) * plot_w if self.max_history > 1 else x_start
                
                # y coordinate
                if max_r > min_r:
                    norm_r = (r - min_r) / (max_r - min_r)
                else:
                    norm_r = 0
                
                py = y_start + plot_h - (norm_r * plot_h)
                points.append((px, py))
                
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)
            
        # Title
        title = self.font.render(f"Reward (Max: {max_r:.1f})", True, self.BLACK)
        self.screen.blit(title, (x_start, y_start - 20))
        
        # Axis Labels
        label_max = self.font.render(f"{max_r:.1f}", True, self.GRAY)
        label_min = self.font.render("0.0", True, self.GRAY)
        self.screen.blit(label_max, (x_start - 30, y_start))
        self.screen.blit(label_min, (x_start - 30, y_start + plot_h - 10))
        
        label_time = self.font.render("Time (-2s)", True, self.GRAY)
        self.screen.blit(label_time, (x_start + plot_w - 60, y_start + plot_h + 2))

    def get_frame(self):
        """Returns the current screen as a numpy array (H, W, 3) in RGB."""
        # Get the pixel data
        # pygame.surfarray.array3d returns (W, H, 3)
        # We need to transpose to (H, W, 3)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def close(self):
        pygame.quit()
