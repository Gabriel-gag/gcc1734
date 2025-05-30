from environment import Environment

# =============================
# Blackjack Environment
# =============================
class BlackjackEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        idx = 1
        self.state_to_id_dict = {}
        self.id_to_state_dict = {}
        for i in range(self.env.observation_space[0].n):
            for j in range(self.env.observation_space[1].n):
                for k in range(self.env.observation_space[2].n):
                    s = (i + 1, j + 1, bool(k))
                    self.state_to_id_dict[s] = idx
                    self.id_to_state_dict[idx] = s
                    idx += 1

    def get_num_states(self):
        return self.env.observation_space[0].n * self.env.observation_space[1].n * self.env.observation_space[2].n

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        return self.state_to_id_dict[state]

    def get_random_action(self):
        return self.env.action_space.sample()


# =============================
# CliffWalking Environment
# =============================
class CliffWalkingEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)

    def get_num_states(self):
        return self.env.observation_space.n

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        return state  # O estado já é um inteiro no CliffWalking

    def get_random_action(self):
        return self.env.action_space.sample()


# =============================
# FrozenLake Environment
# =============================
class FrozenLakeEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)

    def get_num_states(self):
        return self.env.observation_space.n

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        return state  # O estado já é um inteiro no FrozenLake

    def get_random_action(self):
        return self.env.action_space.sample()
