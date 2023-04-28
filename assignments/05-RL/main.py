"""
"""

import gymnasium as gym
from customagent import Agent

# %matplotlib inline

SHOW_ANIMATIONS = True
lrs = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 10e-4]
# lrs = [3e-4]
steps = []
for i in range(1):
    # LR = lrs[0]
    # env = gym.make("LunarLander-v2", render_mode="human" if SHOW_ANIMATIONS else "none")
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=42)

    agent = Agent(
        action_space=env.action_space, observation_space=env.observation_space
    )

    total_reward = 0
    last_n_rewards = []
    scores = []
    # 100000
    max_episodes = 5 * 100000
    for epi in range(max_episodes):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation)
        # print(reward)
        # print(terminated)
        # print(truncated)
        # print(info)
        # exit()
        agent.learn(observation, reward, terminated, truncated)
        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            last_n_rewards.append(total_reward)
            n = min(30, len(last_n_rewards))
            avg = sum(last_n_rewards[-n:]) / n
            improvement_emoji = "🔥" if (total_reward > avg) else "😢"
            print(
                f"{improvement_emoji} Finished with reward {int(total_reward)}.\tAverage of last {n}: {int(avg)}"
            )
            if avg > 0:
                print("🎉 Nice work! You're ready to submit the leaderboard! 🎉")
                # print(epi)
                steps.append(epi)
                break
            total_reward = 0
        scores.append(total_reward)
    env.close()

# plt.figure(figsize=(8, 6))
# plt.plot(range(epi), scores)
# plt.title("Performance of Random Agent")
# plt.xlabel("Episodes")
# plt.ylabel("Score")
# plt.show()
# exit()
# print(steps)