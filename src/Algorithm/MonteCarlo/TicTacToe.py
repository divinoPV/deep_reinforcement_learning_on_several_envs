from collections import defaultdict
from random import random, choice, choices

import pygame

from src.do_not_touch.result_structures import PolicyAndActionValueFunction
from src.env.TicTacToe import TicTacToe


class TicTacToeMonteCarlo(TicTacToe):
    def __init__(self):
        super().__init__()

        self.playing = False
    
    def monte_carlo_es(
        self,
        num_episodes: int = 30_000,
    ) -> PolicyAndActionValueFunction:
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        actions = self.env.get_actions_left()
        Q = defaultdict(lambda: {a: random() for a in actions})
        pi = defaultdict(lambda: {a: random() for a in actions})
        
        for i in range(num_episodes):
            self.env.reset()

            s0 = self.env.calculate_state()
            a0 = choice(self.env.get_actions_left())

            self.env.set_board_with_action(self.env.players[1].id, a0)

            rand_action = None

            while rand_action is None:
                rand_action = self.env.players[0].play(self.env.get_actions_left())

            self.env.set_board_with_action(self.env.players[0].id, rand_action)

            s_history = [s0]
            a_history = [a0]
            s_p_history = [self.env.calculate_state()]
            r_history = [self.env.score()]

            while not self.env.is_game_over():
                s = self.env.calculate_state()
                pis = [pi[s][a] for a in self.env.get_actions_left()]
                av_actions = self.env.get_actions_left()
                a = (choices(av_actions, weights=pis)[0], choice(av_actions))[max(pis) == 0.0]

                self.env.set_board_with_action(self.env.players[1].id, a)
                
                if not self.env.is_game_over():
                    rand_action = self.env.players[0].play(self.env.get_actions_left())
                    self.env.set_board_with_action(self.env.players[0].id, rand_action)
                
                self.env.is_game_over()
                s_history.append(s)
                a_history.append(a)
                s_p_history.append(self.env.calculate_state())
                r_history.append(self.env.score())

            G = 0
            discount = 0.999
            
            for t in reversed(range(len(s_history))):
                G = discount * G + r_history[t]
                s_t = s_history[t]
                a_t = a_history[t]
                
                appear = False

                for t_p in range(t - 1):
                    if s_history[t_p] == s_t and a_history[t_p] == a_t:
                        appear = True
                        break

                if appear:
                    continue
                
                returns_sum[(s_t, a_t)] += G
                returns_count[(s_t, a_t)] += 1.0
                Q[s_t][a_t] = returns_sum[(s_t, a_t)] / returns_count[(s_t, a_t)]
                pi[s_t] = {a: 0.0 for a in Q[s_t].keys()}
                best_action = max(Q[s_t], key=Q[s_t].get)
                pi[s_t][best_action] = 1.0
        
        return PolicyAndActionValueFunction(pi, Q)
    
    def epsilon_greedy_policy(self, Q, epsilon, state, A):
        actions = self.env.get_actions_left()
        if random() > epsilon:
            if Q[state]:
                best_action = max(Q[state], key=Q[state].get)
            else:
                best_action = choice(actions)
            
            A[state] = {
                a: 1.0 if a == best_action else 0
                for a in actions
            }
        else:
            A[state] = {
                a: 1.0 / len(actions) for a in actions
            }
        
        return A[state]
    
    def on_policy_first_visit_monte_carlo_control(
        self,
        num_episodes: int = 30_000,
        epsilon: float = 0.3,
    ) -> PolicyAndActionValueFunction:
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        actions = self.env.get_actions_left()
        Q = defaultdict(lambda: {a: random() for a in actions})
        pi = defaultdict(lambda: {a: random() for a in actions})

        for i_episode in range(1, num_episodes + 1):
            self.env.reset()

            pair_history = []
            s_history = []
            a_history = []
            s_p_history = []
            r_history = []
            
            while not self.env.is_game_over():
                state = self.env.calculate_state()
                pi[state] = self.epsilon_greedy_policy(Q, epsilon, state, pi)

                a = choices(
                    list(pi[state].keys()),
                    weights=list(pi[state].values()),
                )[0]

                self.env.set_board_with_action(self.env.players[1].id, a)

                if not self.env.is_game_over():
                    rand_action = self.env.players[0].play(self.env.get_actions_left())
                    self.env.set_board_with_action(self.env.players[0].id, rand_action)
                
                r = self.env.score()
                s_history.append(state)
                a_history.append(a)
                s_p_history.append(self.env.calculate_state())
                r_history.append(r)
                pair_history.append(((state, a), r))
            
            for ((s, a), r) in pair_history:
                first_idx = next(
                    i
                    for i, (s_a, r)
                    in enumerate(pair_history)
                    if s_a == (s, a)
                )

                G = sum([r for ((s, a), r) in pair_history[first_idx:]])
                returns_sum[(s, a)] += G
                returns_count[(s, a)] += 1.0
                Q[s][a] = returns_sum[(s, a)] / returns_count[(s, a)]

        return PolicyAndActionValueFunction(pi, Q)
    
    @staticmethod
    def create_target_policy(Q):
        def policy_fn(state):
            A = {a: 0.0 for a in Q[state].keys()}
            best_action = max(Q[state], key=Q[state].get)
            A[best_action] = 1.0

            return A
        
        return policy_fn

    def off_policy_monte_carlo_control(
        self,
        num_episodes: int = 30_000,
    ) -> PolicyAndActionValueFunction:
        actions = self.env.get_actions_left()
        Q = defaultdict(lambda: {a: random() for a in actions})
        C = defaultdict(lambda: {a: 0.0 for a in actions})
        pi = defaultdict(lambda: {a: random() for a in actions})
        target_policy = self.create_target_policy(Q)
        
        for i_episode in range(1, num_episodes + 1):
            self.env.reset()
            
            a0 = choices(
                self.env.get_actions_left(),
                weights=[pi[self.env.calculate_state()].get(a, 0) for a in self.env.get_actions_left()],
            )[0]

            self.env.set_board_with_action(self.env.players[1].id, a0)
            
            rand_action = self.env.players[0].play(self.env.get_actions_left())
            self.env.set_board_with_action(self.env.players[0].id, rand_action)
            
            s_history = [self.env.calculate_state()]
            a_history = [a0]
            s_p_history = [self.env.calculate_state()]
            r_history = [self.env.score()]
            
            while not self.env.is_game_over():
                s = self.env.calculate_state()
                pis = [pi[s].get(a, 0) for a in self.env.get_actions_left()]
                av_actions = self.env.get_actions_left()
                a = choices(av_actions, weights=pis)[0] if max(pis) > 0.0 else choice(av_actions)
                
                self.env.set_board_with_action(self.env.players[1].id, a)
                
                if not self.env.is_game_over():
                    self.env.set_board_with_action(
                        self.env.players[0].id,
                        self.env.players[0].play(self.env.get_actions_left()),
                    )
                
                s_history.append(s)
                a_history.append(a)
                s_p_history.append(self.env.calculate_state())
                r_history.append(self.env.score())
            
            G, W, discount = 0.0, 1.0, 0.999
            
            for t in range(len(s_history))[::-1]:
                state, action, reward = s_history[t], a_history[t], r_history[t]
                G = discount * G + reward
                C[state].setdefault(action, 0.0)
                Q[state].setdefault(action, 0.0)
                Q[state][action] += (W / (C[state][action] + 1)) * (G - Q[state][action])
                best_action = max(Q[state], key=Q[state].get)
                
                if action != best_action:
                    break
                
                W *= (target_policy(state).get(action, 0) / pi[state].get(action, 0))
        
        return PolicyAndActionValueFunction({state: target_policy(state) for state in Q.keys()}, Q)

    def set_algorithm(self):
        self.algorithm = {
            "first": {
                "Humain": None,
                "Monte Carlo ES": TicTacToeMonteCarlo().monte_carlo_es,
                "Off-policy Monte Carlo Control": TicTacToeMonteCarlo().off_policy_monte_carlo_control,
                "On-policy first visit Monte Carlo Control": TicTacToeMonteCarlo().on_policy_first_visit_monte_carlo_control,
            },
            "second": {
                "Humain": None,
                "Monte Carlo ES": TicTacToeMonteCarlo().monte_carlo_es,
                "Off-policy Monte Carlo Control": TicTacToeMonteCarlo().off_policy_monte_carlo_control,
                "On-policy first visit Monte Carlo Control": TicTacToeMonteCarlo().on_policy_first_visit_monte_carlo_control,
            },
        }
        
        if self.method_player1 != "Humain":
            self.algorithm["first"] = self.algorithm["first"][self.method_player1]()
            
        if self.method_player2 != "Humain":
            self.algorithm["second"] = self.algorithm["second"][self.method_player2]()
    
    def run(self):
        self.select_method()
        self.set_algorithm()
        self.playing = True

        while self.playing:
            just_started = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.logger.info("Au revoir! :'(")
                    self.playing = False

                    return
            
            while not self.env.is_game_over():
                current_method = (self.method_player2, self.method_player1)[self.current_player == 1]
                current_player_str = ("second", "first")[self.current_player == 1]
                state = self.env.calculate_state()
                
                if just_started:
                    self.draw_panel()
                    just_started = False

                    continue
                    
                if current_method == "Humain":
                    action = None
                    
                    while action is None:
                        for event in pygame.event.get():
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                action = self.get_action(
                                    event.pos[0] // self.BLOCK_WIDTH,
                                    event.pos[1] // self.BLOCK_HEIGHT,
                                )
                    
                    self.env.set_board_with_action(self.current_player, action)
                    self.current_player = 3 - self.current_player

                    continue
                else:
                    if state not in self.algorithm[current_player_str].pi:
                        self.algorithm[current_player_str].pi[state] = {
                            action: random()
                            for action
                            in self.env.get_actions_left()
                        }
                    
                    probabilities = [
                        self.algorithm[current_player_str].pi[state][action]
                        for action
                        in self.env.get_actions_left()
                    ]
                
                if sum(probabilities) <= 0.0:
                    action = choice(self.env.get_actions_left())
                else:
                    action = choices(
                        self.env.get_actions_left(),
                        weights=probabilities,
                    )[0]
                
                self.env.set_board_with_action(self.current_player, action)
                self.current_player = 3 - self.current_player
                self.draw_panel()
            
            if self.env.players[1].is_winner:
                self.logger.info("Le joueur 1 a mit la pâté au joueur 2!")
            elif self.env.players[0].is_winner:
                self.logger.info("Le joueur 2 a mit la pâté au joueur 1!")
            else:
                self.logger.info("Oh mon dieu! :O un match nul...")
            
            self.env.reset()
            self.draw_panel()

        pygame.quit()

    def execute(self):
        print(f"Environment \033[1m{self.__class__.__name__}\033[0m")
        print("\n")
        
        print("\t \033[1mMonte Carlo ES\033[0m")
        print("\t", self.monte_carlo_es())
        print("\n")

        print("\t \033[1mOn Policy First Visit Monte Carlo Control\033[0m")
        print("\t", self.on_policy_first_visit_monte_carlo_control())
        print("\n")
        
        # print("\t \033[1mOff Policy Monte Carlo Control\033[0m")
        # print("\t", self.off_policy_monte_carlo_control())
        # print("\n")
