import logging

from typing import Optional

import numpy as np
import pygame


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Player:
    def __init__(self, id: int, mode: str) -> None:
        self.id = id
        self.mode = mode
        self.score = 0
        self.is_winner = False
    
    def play(self, available_actions, state_id=None, policy=None) -> Optional[int]:
        action_id = None
        
        if self.mode == 'Humain':
            while True:
                action_id = input("Où pourriez-vous bien vous mettre ?")

                if action_id not in available_actions:
                    return None
        
        elif self.mode == 'Random':
            action_id = np.random.choice(available_actions)
        else:
            if state_id is not None and policy is not None:
                action_id = max(policy[state_id], key=policy[state_id].get)
        
        return action_id


class MDP:
    def __init__(self) -> None:
        self.size = 3
        self.actions = np.arange(self.size * self.size)
        self.board = np.zeros((self.size, self.size))
        self.players = [Player(1, 'Random'), Player(2, 'Algorithm')]
    
    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.players[0].score = 0
        self.players[1].score = 0
        self.players[0].is_winner = False
        self.players[1].is_winner = False
    
    def is_game_over(self) -> bool:
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.players[0 if self.players[0].id == self.board[0][0] else 1].is_winner = True

            return True
        elif self.board[2][0] == self.board[1][1] == self.board[0][2] != 0:
            self.players[0 if self.players[0].id == self.board[2][0] else 1].is_winner = True

            return True

        for i in range(self.size):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.players[0 if self.players[0].id == self.board[i][0] else 1].is_winner = True

                return True
            elif self.board[0][i] == self.board[1][i] == self.board[2][i] != 0:
                self.players[0 if self.players[0].id == self.board[0][i] else 1].is_winner = True

                return True

        if len(self.get_actions_left()) == 0:
            return True
        
        return False
    
    def set_board_with_action(self, player_id: int, action_id: int):
        self.board[action_id // self.size][action_id % self.size] = player_id
    
    def score(self) -> float:
        score = 0
        nb_coups = (self.board == 2).sum()

        if self.players[0].is_winner:
            # Because it's easier as a first-time player
            self.players[0].score += (score := 2 if nb_coups <= 3 else 1)

            return score
        
        if self.players[1].is_winner:
            # Because it's harder as a second-time player
            self.players[1].score += (score := 4 if nb_coups <= 3 else 2)
            
            return score
        
        return score

    def get_actions_left(self):
        return np.array([cpt for cpt, val in enumerate(self.board.ravel()) if val == 0])

    def calculate_state(self) -> int:
        return int(
            sum(
                val * (self.size ** (i * self.size + j))
                for i, row in enumerate(self.board)
                for j, val in enumerate(row)
            )
        )


class TicTacToe:
    def __init__(self):
        self.env = MDP()
        self.logger = logging.getLogger(__name__)

        pygame.init()

        self.WINDOW_WIDTH = 900
        self.WINDOW_HEIGHT = 900
        self.LINE_WIDTH = 5
        self.BLOCK_HEIGHT = 300
        self.BLOCK_WIDTH = 300
        self.CIRCLE_RADIUS = 100
        self.CIRCLE_THICKNESS = 10
        self.CROSS_THICKNESS = 10
        self.CROSS_BRANCH_LENGHT = 100
        self.GRID_SIZE = 3
        self.LINE_COLOR = (255, 255, 255)
        self.CROSS_COLOR = (0, 255, 0)
        self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("TicTacToe!")
        self.current_player = 1
        self.method_player1 = None
        self.method_player2 = None
        self.algorithm = {}
        self.human_play = False

    def get_action(self, x, y):
        for action in filter(
            lambda action: x == action % self.env.size and y == action // self.env.size,
            self.env.get_actions_left(),
        ):
            return action
        
        return None

    def draw_grid(self):
        for row in range(self.GRID_SIZE):
            pygame.draw.line(
                self.window,
                self.LINE_COLOR,
                (0, row * self.BLOCK_HEIGHT),
                (self.WINDOW_WIDTH, row * self.BLOCK_HEIGHT),
                self.LINE_WIDTH,
            )
        
        for col in range(self.GRID_SIZE):
            pygame.draw.line(
                self.window,
                self.LINE_COLOR,
                (col * self.BLOCK_WIDTH, 0),
                (col * self.BLOCK_WIDTH, self.WINDOW_HEIGHT),
                self.LINE_WIDTH,
            )
    
    def draw_items(self):
        for row in range(self.env.size):
            for col in range(self.env.size):
                center_x = col * self.BLOCK_WIDTH + self.BLOCK_WIDTH // 2
                center_y = row * self.BLOCK_HEIGHT + self.BLOCK_HEIGHT // 2
                
                if self.env.board[row][col] == 1:
                    pygame.draw.circle(
                        self.window,
                        (255, 0, 0),
                        (center_x, center_y),
                        self.CIRCLE_RADIUS,
                        self.CIRCLE_THICKNESS,
                    )
                elif self.env.board[row][col] == 2:
                    pygame.draw.line(
                        self.window,
                        self.CROSS_COLOR,
                        (
                            center_x - self.CROSS_BRANCH_LENGHT,
                            center_y - self.CROSS_BRANCH_LENGHT,
                        ),
                        (
                            center_x + self.CROSS_BRANCH_LENGHT,
                            center_y + self.CROSS_BRANCH_LENGHT,
                        ),
                        self.CROSS_THICKNESS,
                    )
                    
                    pygame.draw.line(
                        self.window,
                        self.CROSS_COLOR,
                        (
                            center_x - self.CROSS_BRANCH_LENGHT,
                            center_y + self.CROSS_BRANCH_LENGHT,
                        ),
                        (
                            center_x + self.CROSS_BRANCH_LENGHT,
                            center_y - self.CROSS_BRANCH_LENGHT,
                        ),
                        self.CROSS_THICKNESS,
                    )
    
    def draw_panel(self):
        self.window.fill((0, 0, 0))
        self.draw_grid()
        self.draw_items()
        
        pygame.display.flip()
    
    def draw_selection_ui(self, options):
        font = pygame.font.Font(None, 30)
        buttons = {}
        
        self.window.blit(
            font.render("Sélectionner une méthode :", True, (0, 0, 0)),
            (50, 50),
        )
        
        text_y = 100
        
        for option in options:
            button_rect = pygame.Rect(50, text_y, 200, 30)
            buttons[option] = button_rect
            
            text = font.render(option, True, (0, 0, 0))
            self.window.blit(text, (60, text_y + 5))
            text_y += 40
        
        return buttons
    
    def draw_buttons(self, buttons):
        for button in buttons.values():
            pygame.draw.rect(self.window, (250, 250, 250), button)
            pygame.draw.rect(self.window, (0, 0, 0), button, 2)
    
    def handle_selection(self, selection):
        if self.method_player1 is None:
            self.method_player1 = selection
            print("Le premier joueur est ", selection)
        elif self.method_player2 is None:
            self.method_player2 = selection
            print("Le second joueur est ", selection)
    
    def select_method(self):
        buttons = self.draw_selection_ui([
            "Humain",
            "Monte Carlo ES",
            "Off-policy Monte Carlo Control",
            "On-policy first visit Monte Carlo Control",
        ])
        
        while self.method_player1 is None or self.method_player2 is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for option, button in buttons.items():
                        if button.collidepoint(pygame.mouse.get_pos()):
                            selection = option
                            self.handle_selection(selection)
            
            self.human_play = self.method_player1 == "Humain" or self.method_player2 == "Humain"
            self.window.fill((255, 255, 255))
            self.draw_buttons(buttons)
            
            pygame.display.flip()
