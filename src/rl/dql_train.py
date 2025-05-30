import argparse
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mountain_car_environment import MountainCarEnvironment  # Importa o seu ambiente personalizado
from dql import QLearningAgentTabularDiscrete

# Dicionário de ambientes
environments_dict = {
    "MountainCar-v0": {
        "bins": (20, 20),  # Número de bins para discretização
        "learning_rate": 0.9,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MountainCar-v0", help="Nome do ambiente")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Número de episódios")
    args = parser.parse_args()

    # Verificação se o ambiente está no dicionário
    if args.env_name not in environments_dict:
        raise ValueError(f"Ambiente {args.env_name} não encontrado no dicionário de ambientes.")

    env = MountainCarEnvironment(bins=environments_dict[args.env_name]["bins"])

    # Configurações específicas do ambiente
    env_config = environments_dict[args.env_name]

    agent = QLearningAgentTabularDiscrete(
        env.env,
        bins=env_config["bins"],
        learning_rate=env_config["learning_rate"],
        gamma=env_config["gamma"],
        epsilon=env_config["epsilon"],
        epsilon_min=env_config["epsilon_min"],
        epsilon_decay=env_config["epsilon_decay"]
    )

    # Lista para armazenar recompensas de cada episódio
    rewards = []
    epsilons = []

    # Treinamento do agente
    for episode in range(args.num_episodes):
        state, _ = env.env.reset()  # Chamar o método reset no ambiente real
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.env.step(action)  # Chamar step no ambiente real
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Armazenar a recompensa total do episódio
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # Salvando o agente treinado
    agent.save("dql_mountaincar.npy")

    # Plotar a curva de aprendizado suavizada
    plt.plot(savgol_filter(rewards, 101, 3))  # Ajuste a janela de suavização e o grau do polinômio conforme necessário
    plt.title(f"Curva de aprendizado suavizada ({args.env_name})")
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa total')
    plt.savefig(args.env_name + "-dql-learning_curve.png")
    plt.close()

    # Plotar o decaimento do epsilon
    plt.plot(epsilons)
    plt.title(f"Decaimento do valor de $\\epsilon$ ({args.env_name})")
    plt.xlabel('Episódio')
    plt.ylabel('$\\epsilon$')
    plt.savefig(args.env_name + "-dql-epsilons.png")
    plt.close()
