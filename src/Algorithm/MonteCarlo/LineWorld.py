import random

from typing import List, Tuple, Optional

import numpy as np

from src.do_not_touch.result_structures import (
    PolicyAndActionValueFunction,
    Policy,
)
from src.env.LineWorld import LineWorld


class LineWorldMonteCarlo(LineWorld):
    def __init__(self, size: int = 7):
        super().__init__(size)

    def generate_episode(self, policy: Policy) -> List[Tuple[int, int, float]]:
        """ Generate an episode following policy pi.
        """

        episode = []
        state = random.choice(self.states[1:-1])
        action = random.choice(self.actions)

        while state not in [0, 6]:
            reward = (
                self.rewards[1]
                if (state == 1 and action == 0) or (state == 5 and action == 1)
                else self.rewards[2]
            )

            episode.append((state, action, reward))
            state = state - 1 if action == 0 else state + 1

            action = (
                np
                .random
                .choice(
                    self.actions,
                    p=[
                        policy[state][a]
                        for a
                        in self.actions
                    ],
                )
                if state not in [0, 6]
                else None
            )

        return episode

    def monte_carlo_es(self, num_episodes=1000, gamma=1.0) -> PolicyAndActionValueFunction:
        q = {s: {a: 0.0 for a in self.actions} for s in self.states}
        policy = {s: {a: 1 / len(self.actions) for a in self.actions} for s in self.states}
        returns = {s: {a: [] for a in self.actions} for s in self.states}

        for _ in range(num_episodes):
            episode = self.generate_episode(policy)
            G = 0

            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = gamma * G + reward

                if not any((state == s and action == a) for s, a, _ in episode[0:t]):
                    returns[state][action].append(G)
                    q[state][action] = np.mean(returns[state][action])
                    best_action = max(policy[state], key=policy[state].get)

                    for a in policy[state]:
                        policy[state][a] = 1.0 if a == best_action else 0.0

        return PolicyAndActionValueFunction(policy, q)
    
    @staticmethod
    def transition(state, action) -> Tuple[int, float]:
        if state == 0 or state == 6:
            return state, 0.0

        if action == 0:
            return state - 1, -1.0 if state == 1 else 0.0

        if action == 1:
            return state + 1, 1.0 if state == 5 else 0.0
    
    def on_policy_first_visit_monte_carlo_control(
        self,
        num_episodes: int = 1000,
        gamma: float = 1.0,
        breaker_limit: int = 100_000,
    ) -> Optional[PolicyAndActionValueFunction]:
        Q = {s: {a: 0.0 for a in self.actions} for s in self.states}
        returns = {s: {a: [] for a in self.actions} for s in self.states}
        policy = {s: {a: 1 / len(self.actions) for a in self.actions} for s in self.states}
        
        for _ in range(num_episodes):
            state = random.choice(self.states[1:-1])
            episode = []

            breaker = 0
            while state not in [0, 6]:
                probs = list(policy[state].values())
                action = random.choices(self.actions, probs)[0]
                next_state, reward = self.transition(state, action)
                episode.append((state, action, reward))
                state = next_state
                breaker += 1
                
                if breaker == breaker_limit:
                    return None
            
            G = 0.0
            first_visit_check = {s: {a: False for a in self.actions} for s in self.states}

            for state, action, reward in reversed(episode):
                G = gamma * G + reward

                if not first_visit_check[state][action]:
                    first_visit_check[state][action] = True
                    returns[state][action].append(G)
                    Q[state][action] = sum(returns[state][action]) / len(returns[state][action])
                    best_action = max(Q[state], key=Q[state].get)

                    for a in policy[state]:
                        policy[state][a] = 1.0 if a == best_action else 0.0
        
        return PolicyAndActionValueFunction(policy, Q)
    
    def off_policy_monte_carlo_control(
        self,
        num_episodes: int = 1000,
        gamma: float = 1.0,
        breaker_limit: int = 100_000,
    ) -> Optional[PolicyAndActionValueFunction]:
        Q = {s: {a: 0.0 for a in self.actions} for s in self.states}
        policy_function = {s: {a: 1 / len(self.actions) for a in self.actions} for s in self.states}
        returns = {s: {a: [] for a in self.actions} for s in self.states}

        for episode in range(num_episodes):
            state = np.random.choice(self.states)
            action = np.random.choice(self.actions)
            episode = []
            
            breaker = 0
            while state not in [0, 6]:
                next_state, reward = self.transition(state, action)
                episode.append((state, action, reward))
                state = next_state
                action = np.random.choice(self.actions, p=[policy_function[state][a] for a in self.actions])
                breaker += 1
                
                if breaker == breaker_limit:
                    return None

            G = 0

            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = gamma * G + reward

                if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                    returns[state][action].append(G)
                    Q[state][action] = np.mean(returns[state][action])

                    policy_function[state] = {
                        a: (a == np.argmax([Q[state][a] for a in self.actions]))
                        for a
                        in self.actions
                    }

        return PolicyAndActionValueFunction(policy_function, Q)

    def execute(self):
        print(f"Environment \033[1m{self.__class__.__name__}\033[0m")
        print("\n")
        
        print("\t \033[1mMonte Carlo ES\033[0m")
        print("\t", self.monte_carlo_es())
        print("\n")
        
        print("\t \033[1mOn Policy First Visit Monte Carlo Control\033[0m")
        print("\t", self.on_policy_first_visit_monte_carlo_control())
        print("\n")
        
        print("\t \033[1mOff Policy Monte Carlo Control\033[0m")
        print("\t", self.off_policy_monte_carlo_control())
        print("\n")
