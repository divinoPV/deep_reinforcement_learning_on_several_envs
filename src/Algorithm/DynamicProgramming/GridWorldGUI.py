import sys
import pygame
import time

# Ajouter le chemin d'accès au répertoire parent de "src" pour que Python puisse le trouver
sys.path.append('/Users/annadiaw/Desktop/deep_reinforcement_learning_on_several_envs')

# Le chemin d'importation pour GridWorldDynamicProgramming
from src.Algorithm.DynamicProgramming.GridWorld import GridWorldDynamicProgramming


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
pygame.display.set_caption("Grid World - Dynamic Programming")

# Define font
font = pygame.font.Font(None, 24)  # Adjust font size here

# Create the GridWorld environment
grid_world = GridWorldDynamicProgramming()  # Set the size here
size = 5

# Initialize the optimal policy and value function with default action 0 and value 0.0 for all states
optimal_policy = {state_idx: 0 for state_idx in range(size * size)}
optimal_value = {state_idx: 0.0 for state_idx in range(size * size)}

# Function to update the optimal policy and value function after the dynamic programming algorithm
def update_optimal_policy_value(policy, value_function):
    global optimal_policy, optimal_value
    optimal_policy = policy
    optimal_value = value_function
    for state, action in optimal_policy.items():
        print(f"State: {state} -> Action: {action}")


def draw_grid_world(agent_pos, policy, value_function):
    cell_width = SCREEN_WIDTH / size
    cell_height = SCREEN_HEIGHT / size
    for i in range(size):
        for j in range(size):
            state_x = i * cell_width
            state_y = j * cell_height
            pygame.draw.rect(screen, WHITE, (state_x, state_y, cell_width, cell_height), 2)

            # Display the Value Function value for each state
            state_idx = (i, j)

            value = value_function[state_idx]
            value_text = f"V(s={state_idx}): {value:.2f}"
            value_text = font.render(value_text, True, WHITE)

            # Center the text within the cell
            value_text_rect = value_text.get_rect(center=(state_x + cell_width / 2, state_y + cell_height / 2))
            screen.blit(value_text, value_text_rect)

            # Display the Policy for each state
            if state_idx in policy:
                state_actions = policy[state_idx]
                max_action_value = max(state_actions.values())
                optimal_actions = [action for action, value in state_actions.items() if value == max_action_value]
                action_text = f"Action(s={state_idx}): {optimal_actions}"
            else:
                action_text = f"Action(s={state_idx}): No Policy"
            action_text = font.render(action_text, True, WHITE)

            print(f"Agent Position: {agent_pos}, Chosen Action: {optimal_actions}")

            # Center the text within the cell
            action_text_rect = action_text.get_rect(center=(state_x + cell_width / 2, state_y + cell_height / 1.5))
            screen.blit(action_text, action_text_rect)

    # Draw the agent
    agent_x = agent_pos[0] * cell_width + cell_width / 2
    agent_y = agent_pos[1] * cell_height + cell_height / 2
    pygame.draw.circle(screen, RED, (int(agent_x), int(agent_y)), 10)


def grid_world_gui():
    global optimal_policy, optimal_value
    print(optimal_policy)

    running = True
    agent_pos = (0, 0)  # Starting the agent at the top-left corner

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)

        draw_grid_world(agent_pos, optimal_policy, optimal_value)
        pygame.display.flip()

        time.sleep(0.5)  # Delay to visualize the agent's movements

        # Determine the agent's next position based on the optimal policy
        current_state_idx = agent_pos
        action = optimal_policy[current_state_idx]

        if action == 0:  # Up
            agent_pos = (max(0, agent_pos[0] - 1), agent_pos[1])
        elif action == 1:  # Down
            agent_pos = (min(size - 1, agent_pos[0] + 1), agent_pos[1])
        elif action == 2:  # Left
            agent_pos = (agent_pos[0], max(0, agent_pos[1] - 1))
        elif action == 3:  # Right
            agent_pos = (agent_pos[0], min(size - 1, agent_pos[1] + 1))

    pygame.quit()

if __name__ == "__main__":
    # Execute the dynamic programming algorithm and update the optimal policy and value function
    policy, value_function = grid_world.execute() 
    update_optimal_policy_value(policy, value_function)
    
    # Démarrez la GUI
    grid_world_gui()