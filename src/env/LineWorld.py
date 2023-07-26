class LineWorld:
    """ Creates a Line World of 7 cells (leftmost and rightmost are
        terminal, with -1 and 1 reward respectively).
    """

    def __init__(self, size: int = 7):
        self.states = [i for i in range(size)]
        self.actions = [0, 1]
        self.rewards = [-1.0, 0.0, 1.0]

    @staticmethod
    def transition_probability(state, action, state_policy, reward) -> float:
        assert state >= 0 and state <= 6
        assert state_policy >= 0 and state_policy <= 6
        assert action >= 0 and action <= 1
        assert reward >= 0 and reward <= 2

        if state == 0 or state == 6:
            return 0.0

        if (
            (
                state + 1 == state_policy and
                action == 1 and
                reward == 1 and
                state != 5
            ) or
            (
                state + 1 == state_policy and
                action == 1 and
                reward == 2 and
                state == 5
            ) or
            (
                state - 1 == state_policy and
                action == 0 and
                reward == 1 and
                state != 1
            ) or
            (
                state - 1 == state_policy and
                action == 0 and
                reward == 0 and
                state == 1
            )
        ):
            return 1.0
        
        return 0.0
    
    @staticmethod
    def random_probability(state, action) -> float:
        if state == 0 or state == 6:
            return 0.0
        
        return 0.5
