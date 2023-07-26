import numpy as np

from src.do_not_touch.result_structures import PolicyAndActionValueFunction

from src.env.GridWorld import GridWorld


class GridWorldMonteCarlo(GridWorld):
    def __init__(self, size: int = 5):
        super().__init__(size)

    def monte_carlo_es(self, num_episodes=1000) -> PolicyAndActionValueFunction:
        pi = {state: {action: 1 / len(self.actions) for action in self.actions} for state in self.states}
        q = {state: {action: 0 for action in self.actions} for state in self.states}
        returns = {state: {action: [] for action in self.actions} for state in self.states}
        
        for _ in range(num_episodes):
            state = tuple(np.random.choice(range(5), size=2))
            action = np.random.choice(self.actions)
            episode = []

            while state not in [(0, 4), (4, 4)]:
                reward = self.transition_probability(state, action, state, 1)
                episode.append((state, action, reward))
                state = tuple(np.random.choice(range(5), size=2))
                action = np.random.choice(self.actions)
            
            g = 0

            for state, action, reward in reversed(episode):
                g = reward + g

                if (state, action) not in [(x[0], x[1]) for x in episode[::-1]]:
                    returns[state][action].append(g)
                    q[state][action] = np.mean(returns[state][action])
                    best_action = max(q[state], key=q[state].get)

                    for a in pi[state].keys():
                        pi[state][a] = 1 if a == best_action else 0
        
        return PolicyAndActionValueFunction(pi, q)
    
    def on_policy_first_visit_monte_carlo_control(self) -> PolicyAndActionValueFunction:
        pass
    
    def off_policy_monte_carlo_control(self) -> PolicyAndActionValueFunction:
        pass
    
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
