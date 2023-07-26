import sys
import pygame
import time

# Ajouter le chemin d'accès au répertoire parent de "src" pour que Python puisse le trouver
sys.path.append('/Users/annadiaw/Desktop/deep_reinforcement_learning_on_several_envs')

# Le chemin d'importation pour LineWorldDynamicProgramming
from src.Algorithm.DynamicProgramming.LineWorld import LineWorldDynamicProgramming

# Le chemin d'importation pour LineWorld
from src.env.LineWorld import LineWorld

# Initialize Pygame
pygame.init()

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set the width and height of the screen (adjust as needed)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Line World - Dynamic Programming")

# Define font
font = pygame.font.Font(None, 24)  # Adjust font size here

# Create the LineWorld environment
line_world = LineWorldDynamicProgramming()

def draw_line_world(agent_pos, policy, value_function):
    cell_width = SCREEN_WIDTH / len(line_world.states)
    cell_height = SCREEN_HEIGHT / 2
    for i in range(len(line_world.states)):
        state_x = i * cell_width
        state_y = SCREEN_HEIGHT / 4
        pygame.draw.rect(screen, WHITE, (state_x, state_y, cell_width, cell_height), 2)

        # Display the Policy for each state
        policy_text = f"Policy(s={i}): {policy[i]}"
        policy_text = font.render(policy_text, True, WHITE)

        # Center the text within the cell
        policy_text_rect = policy_text.get_rect(center=(state_x + cell_width / 2, state_y + cell_height / 2))
        screen.blit(policy_text, policy_text_rect)

        # Display the Value Function value for each state
        value = value_function[i]
        value_text = f"V(s={i}): {value:.2f}"
        value_text = font.render(value_text, True, WHITE)

        # Center the text within the cell
        value_text_rect = value_text.get_rect(center=(state_x + cell_width / 2, state_y + cell_height / 1.5))
        screen.blit(value_text, value_text_rect)

    # Draw the agent
    agent_x = agent_pos * cell_width + cell_width / 2
    agent_y = SCREEN_HEIGHT / 4 + cell_height / 2
    pygame.draw.circle(screen, RED, (int(agent_x), int(agent_y)), 10)

# Main game loop
running = True
agent_pos = len(line_world.states) // 2  # Start the agent at the center of the Line World
optimal_policy,optimal_value = line_world.execute()  # Get the optimal policy from policy iteration
print(optimal_policy)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with black
    screen.fill(BLACK)

    # Draw the Line World cells with the current Policy and Value Function and the agent
    draw_line_world(agent_pos, optimal_policy, optimal_value)

    # Update the display
    pygame.display.flip()

    # Add a delay to slow down the animation (adjust as needed)
    time.sleep(1)

    # Move the agent based on the optimal policy
    action = optimal_policy[agent_pos]
    if action == 0:
        agent_pos -= 1
    else:
        agent_pos += 1

    # Make sure the agent stays within the bounds of the Line World
    agent_pos = max(0, min(agent_pos, len(line_world.states) - 1))

# Quit Pygame
pygame.quit()

