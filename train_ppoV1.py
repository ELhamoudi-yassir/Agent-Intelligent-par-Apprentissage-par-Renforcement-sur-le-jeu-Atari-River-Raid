import ale_py
import gymnasium as gym

from stable_baselines3 import PPO

print("Création environnement...")

env = gym.make(
    "ALE/Riverraid-v5",
    render_mode=None
)

print("Création modèle PPO...")

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    gamma=0.99,
    n_steps=128,
    batch_size=64,
    ent_coef=0.01,
    verbose=1
)

print("Début apprentissage...")

model.learn(total_timesteps=50000)

print("Sauvegarde modèle...")

model.save("riverraid_ppo_v1")

env.close()

print("FIN")