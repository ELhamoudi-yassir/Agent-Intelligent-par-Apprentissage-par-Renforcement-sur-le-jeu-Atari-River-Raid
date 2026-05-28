import gymnasium as gym
import ale_py

# Enregistre ALE pour que Gymnasium trouve le "Namespace ALE"
gym.register_envs(ale_py)

# On crée l'environnement pour River Raid
env = gym.make("ALE/Riverraid-v5", render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # L'IA joue au hasard
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()