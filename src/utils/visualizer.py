import pygame
import numpy as np
import sys

class Visualizer:
    def __init__(self, env):
        self.env = env
        
        # Screen Dimensions
        self.width = 800
        self.height = 600
        self.scale = 100 # Pixels per meter
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Double Pendulum Stabilization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        
    def render(self, state, force=0.0, episode=0, step=0, reward=0.0):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.screen.fill(self.WHITE)
        
        # Unpack State
        x, theta1, theta2, x_dot, theta1_dot, theta2_dot = state
        
        # Dimensions
        cart_w = 0.6 * self.scale
        cart_h = 0.3 * self.scale
        l1 = self.env.l1 * self.scale
        l2 = self.env.l2 * self.scale
        
        # Center of screen is (width/2, height/2)
        # World origin (0,0) is at (width/2, height/2)
        ox = self.width / 2
        oy = self.height / 2
        
        # Cart Position
        cart_x = ox + x * self.scale
        cart_y = oy
        
        # Draw Coordinate System
        # Origin is at (ox, oy)
        pygame.draw.line(self.screen, self.GRAY, (int(ox), 0), (int(ox), self.height), 1) # Y-axis
        pygame.draw.line(self.screen, self.GRAY, (0, int(oy)), (self.width, int(oy)), 1) # X-axis
        
        # Labels
        x_label = self.font.render("x", True, self.GRAY)
        y_label = self.font.render("y", True, self.GRAY)
        self.screen.blit(x_label, (self.width - 20, int(oy) + 5))
        self.screen.blit(y_label, (int(ox) + 5, 5))
        
        # Draw Track
        pygame.draw.line(self.screen, self.BLACK, (0, int(oy)), (self.width, int(oy)), 2)
        
        # Draw Cart
        cart_rect = pygame.Rect(int(cart_x - cart_w/2), int(cart_y - cart_h/2), int(cart_w), int(cart_h))
        pygame.draw.rect(self.screen, self.BLACK, cart_rect)
        
        # Pendulum 1
        # theta=0 is UP in our derivation? No, derivation said 0 is DOWN.
        # But in Env reset, we set theta ~ pi for UP.
        # So if theta=pi, it should be UP.
        # Coordinates: x1 = x + l1 sin(theta1), y1 = -l1 cos(theta1) (y is UP)
        # Screen y is DOWN. So screen_y = oy - y_world.
        
        p1_x = cart_x + l1 * np.sin(theta1)
        p1_y = cart_y + l1 * np.cos(theta1) # + because screen Y is down, and cos(pi)=-1 -> up
        
        pygame.draw.line(self.screen, self.BLUE, (int(cart_x), int(cart_y)), (int(p1_x), int(p1_y)), 6)
        pygame.draw.circle(self.screen, self.BLUE, (int(p1_x), int(p1_y)), 10)
        
        # Pendulum 2
        p2_x = p1_x + l2 * np.sin(theta2)
        p2_y = p1_y + l2 * np.cos(theta2)
        
        pygame.draw.line(self.screen, self.RED, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), 6)
        pygame.draw.circle(self.screen, self.RED, (int(p2_x), int(p2_y)), 10)
        
        # Force Indicator
        if abs(force) > 0.1:
            force_len = force * 2 # Scale for visibility
            start_pos = (int(cart_x), int(cart_y))
            end_pos = (int(cart_x + force_len), int(cart_y))
            pygame.draw.line(self.screen, self.GREEN, start_pos, end_pos, 4)
            # Arrowhead
            
        # Info Text
        info_text = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"Force: {force:.2f} N",
            f"x: {x:.2f} m",
            f"Theta1: {theta1:.2f} rad",
            f"Theta2: {theta2:.2f} rad"
        ]
        
        for i, line in enumerate(info_text):
            text_surf = self.font.render(line, True, self.BLACK)
            self.screen.blit(text_surf, (10, 10 + i * 20))
            
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
