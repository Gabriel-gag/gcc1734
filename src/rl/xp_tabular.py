import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from tql import QLearningAgentTabular
from rl_enviroments import (
    BlackjackEnvironment,
    CliffWalkingEnvironment,
    FrozenLakeEnvironment
)

# =========================
# Hiperparâmetros
# =========================
DECAY_RATE = 0.001         # Controla decaimento do epsilon (exploração)
LEARNING_RATE = 0.1        # Taxa de aprendizado (alpha)
GAMMA = 0.9                # Fator de desconto
EPISODES = 5000            # Número de episódios de treinamento

# ===========================================================
# Blackjack
# ===========================================================
print("\n================ Blackjack ===================")

gym_env = gym.make('Blackjack-v1', sab=True)
env = BlackjackEnvironment(gym_env)

agent_blackjack = QLearningAgentTabular(
    env=env,
    decay_rate=DECAY_RATE,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA
)

agent_blackjack.train(num_episodes=EPISODES)
agent_blackjack.save("q_agent_blackjack.pkl")

# ===========================================================
# Cliff Walking
# ===========================================================
print("\n================ Cliff Walking ===================")

gym_env = gym.make('CliffWalking-v0')
env = CliffWalkingEnvironment(gym_env)

agent_cliff = QLearningAgentTabular(
    env=env,
    decay_rate=DECAY_RATE,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA
)

agent_cliff.train(num_episodes=EPISODES)
agent_cliff.save("q_agent_cliff.pkl")

# ===========================================================
# Frozen Lake (com ajuste de limite de passos)
# ===========================================================
print("\n================ Frozen Lake ===================")

gym_env = gym.make('FrozenLake-v1', is_slippery=True)
# Aumenta o limite de passos para evitar 'truncated'
gym_env = TimeLimit(gym_env.env, max_episode_steps=500)

env = FrozenLakeEnvironment(gym_env)

agent_frozen = QLearningAgentTabular(
    env=env,
    decay_rate=DECAY_RATE,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA
)

agent_frozen.train(num_episodes=EPISODES)
agent_frozen.save("q_agent_frozen.pkl")

# ============================
# Fim dos experimentos
# ============================
print("\nTodos os experimentos foram concluídos com sucesso!")
