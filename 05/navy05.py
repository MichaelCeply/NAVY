import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time

env = gym.make("CartPole-v1", max_episode_steps=2000)

# diskretizace stavu
NUM_BUCKETS = (3, 3, 8, 6)
NUM_ACTIONS = env.action_space.n

# uprava hranic
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[0] = [-2.4, 2.4]  # poloha voziku
STATE_BOUNDS[1] = [-0.5, 0.5]  # rychlost voziku
STATE_BOUNDS[2] = [-math.radians(12), math.radians(12)]  # uhel tyce - zmenseni rozsahu
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]  # uhlova rychlost tyce

# inicializace q-table
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,)) + 5.0

# learning params
MIN_LEARNING_RATE = 0.1
MIN_EPSILON = 0.01
DISCOUNT = 0.95
DECAY = 200


# diskretizace stauv
def discretize(obs):
    x, x_dot, theta, theta_dot = obs

    # omezeni na predem definovane hranice
    x = max(STATE_BOUNDS[0][0], min(STATE_BOUNDS[0][1], x))
    x_dot = max(STATE_BOUNDS[1][0], min(STATE_BOUNDS[1][1], x_dot))
    theta = max(STATE_BOUNDS[2][0], min(STATE_BOUNDS[2][1], theta))
    theta_dot = max(STATE_BOUNDS[3][0], min(STATE_BOUNDS[3][1], theta_dot))

    # diskretizace uhlu - upraveno pro vyssi presnost okolo stredu
    if abs(theta) < math.radians(6):
        theta_idx = int(
            (theta + math.radians(6)) / (2 * math.radians(6)) * (NUM_BUCKETS[2] // 2)
        )
    else:
        if theta > 0:
            theta_idx = NUM_BUCKETS[2] // 2 + int(
                (theta - math.radians(6))
                / (math.radians(12) - math.radians(6))
                * (NUM_BUCKETS[2] // 2)
            )
        else:
            theta_idx = int(
                (theta + math.radians(12))
                / (math.radians(12) - math.radians(6))
                * (NUM_BUCKETS[2] // 2)
            )

    # diskretizace - zbytek
    x_idx = int(
        (x - STATE_BOUNDS[0][0])
        / (STATE_BOUNDS[0][1] - STATE_BOUNDS[0][0])
        * (NUM_BUCKETS[0] - 1)
    )
    x_dot_idx = int(
        (x_dot - STATE_BOUNDS[1][0])
        / (STATE_BOUNDS[1][1] - STATE_BOUNDS[1][0])
        * (NUM_BUCKETS[1] - 1)
    )
    theta_dot_idx = int(
        (theta_dot - STATE_BOUNDS[3][0])
        / (STATE_BOUNDS[3][1] - STATE_BOUNDS[3][0])
        * (NUM_BUCKETS[3] - 1)
    )

    x_idx = max(0, min(NUM_BUCKETS[0] - 1, x_idx))
    x_dot_idx = max(0, min(NUM_BUCKETS[1] - 1, x_dot_idx))
    theta_idx = max(0, min(NUM_BUCKETS[2] - 1, theta_idx))
    theta_dot_idx = max(0, min(NUM_BUCKETS[3] - 1, theta_dot_idx))

    return (x_idx, x_dot_idx, theta_idx, theta_dot_idx)


# vyber akce
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state])


# aktualizace q-table
def update_q(state, action, reward, new_state, alpha):
    best_future_q = np.max(q_table[new_state])
    current_q = q_table[state + (action,)]

    q_table[state + (action,)] = current_q + alpha * (
        reward + DISCOUNT * best_future_q - current_q
    )


# vypocet epsilon+learning rate podle epochy
def get_epsilon(t):
    return max(MIN_EPSILON, min(1.0, 1.0 - math.log10((t + 1) / DECAY)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY)))


# vypocet odmeny
def shape_reward(obs, reward):
    x, x_dot, theta, theta_dot = obs

    r = reward

    # pokuta za velky uhel
    angle_penalty = (theta / math.radians(12)) ** 2
    r -= angle_penalty * 0.1

    # pokuta za blizkost okraje
    edge_penalty = min(1.0, abs(x) / 2.4) * 0.1
    r -= edge_penalty

    return r


EPISODES = 2000
MAX_TRAINING_ROUNDS = 5  # pocet treninkovych kol
best_reward = 0
best_q_table = None

for training_round in range(MAX_TRAINING_ROUNDS):
    print(f"\nTraining round {training_round + 1}/{MAX_TRAINING_ROUNDS}")

    # reset q table
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,)) + 5.0

    rewards = []
    max_round_reward = 0

    for episode in range(EPISODES):
        obs, _ = env.reset()
        current_state = discretize(obs)

        alpha = get_learning_rate(episode)
        epsilon = get_epsilon(episode)

        total_reward = 0
        done = False

        while not done:
            action = choose_action(current_state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            shaped_reward = shape_reward(next_obs, reward)
            new_state = discretize(next_obs)

            update_q(current_state, action, shaped_reward, new_state, alpha)

            current_state = new_state
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

        # ulozeni nejlepsi odmeny
        if total_reward > max_round_reward:
            max_round_reward = total_reward

        if episode % 100 == 0:
            print(f"Ep {episode}: Reward = {total_reward}, Max = {max_round_reward}")

        # early stopping
        if total_reward > 900:
            print(f"Good solution found in {episode} with reward {total_reward}")
            # stability test
            stability_count = 0
            for _ in range(5):
                test_obs, _ = env.reset()
                test_state = discretize(test_obs)
                test_reward = 0
                test_done = False

                while not test_done:
                    test_action = np.argmax(q_table[test_state])
                    test_obs, test_r, test_term, test_trunc, _ = env.step(test_action)
                    test_state = discretize(test_obs)
                    test_reward += test_r
                    test_done = test_term or test_trunc

                if test_reward > 500:
                    stability_count += 1
            # pokud alespon 3 modely byly stablini, ukoncime drive
            if stability_count >= 3:
                print(f"Stable model found!")
                break

    # test
    test_rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        state = discretize(obs)
        test_reward = 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize(obs)
            test_reward += reward
            done = terminated or truncated

        test_rewards.append(test_reward)

    avg_test_reward = sum(test_rewards) / len(test_rewards)
    print(f"Avg test revard: {avg_test_reward:.1f}")

    # ulozeni nejlpsiho modelu
    if avg_test_reward > best_reward:
        best_reward = avg_test_reward
        best_q_table = np.copy(q_table)
        print(f"New best reward: {best_reward:.1f}")

# nacteni nejlepsiho modelu
q_table = best_q_table
print(f"\nBest avg reward: {best_reward:.1f}")

# vizualizace
print("\nVisualization of trained model!\n")
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=1000)

# testovani nejlepsiho modelu
test_episodes = 5
for test_ep in range(test_episodes):
    obs, _ = env.reset()
    current_state = discretize(obs)
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[current_state])
        obs, reward, terminated, truncated, _ = env.step(action)
        current_state = discretize(obs)
        total_reward += reward
        time.sleep(0.01)
        done = terminated or truncated

    print(f"Test {test_ep+1}: Steps: {total_reward}")

env.close()
