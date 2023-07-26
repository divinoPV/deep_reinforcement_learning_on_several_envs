from src.do_not_touch.mdp_env_wrapper import Env1


class Secret1:
    def __init__(self):
        self.env = Env1()
        self.states = self.env.states().tolist()
        self.actions = self.env.actions().tolist()
        self.rewards = self.env.rewards().tolist()
        self.transition_probability = self.env.transition_probability

    def random_probability(self, state, action) -> float:
        return (
            1.0 / self.env.data.ActionsLength(),
            0.0,
        )[self.env.is_state_terminal(state)]
