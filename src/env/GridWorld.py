class GridWorld:
    """ Creates a Grid World of 5x5 cells (upper rightmost and lower
        rightmost are terminal, with -1 and 1 reward respectively).
    """

    def __init__(self, size: int = 5):
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.actions = [0, 1, 2, 3]
        self.rewards = [-1.0, 0.0, 1.0]

    @staticmethod
    def transition_probability(state, action, sub_state, reward):
        assert 0 <= state[0] <= 4 and 0 <= state[1] <= 4
        assert 0 <= sub_state[0] <= 4 and 0 <= sub_state[1] <= 4
        assert 0 <= action <= 3
        assert 0 <= reward <= 2

        if state in [(0, 4), (4, 4)]:
            return 0.0

        if (
            (state[0], state[1] - 1) == sub_state and
            action == 0 and
            reward == 1
        ):  # left
            return 1.0

        if (
            (state[0], state[1] + 1) == sub_state and
            action == 1 and
            (
                (
                    reward == 1 and state != (0, 3) and
                    state != (3, 4)
                ) or
                (reward == 0 and state == (0, 3)) or
                (reward == 2 and state == (4, 3))
            )
        ):  # right
            return 1.0

        if (
            (state[0] - 1, state[1]) == sub_state and
            action == 2 and
            (
                (reward == 1 and state != (1, 4)) or
                (reward == 0 and state == (1, 4))
            )
        ):  # up
            return 1.0

        if (
            (state[0] + 1, state[1]) == sub_state and
            action == 3 and
            (
                (reward == 1 and state != (3, 4)) or
                (reward == 2 and state == (3, 4))
            )
        ):  # down
            return 1.0

        return 0.0

    @staticmethod
    def random_probability(state, action) -> float:
        if state == (4, 4) or state == (0, 4):
            return 0.0

        return 0.25
