import ale_py
import gymnasium as gym

from stable_baselines3 import PPO


# =========================================================
# CRÉATION ENVIRONNEMENT
# =========================================================

env = gym.make(
    "ALE/Riverraid-v5",
    render_mode="human"
)

# =========================================================
# CHARGEMENT MODÈLE PPO
# =========================================================

model = PPO.load("riverraid_ppo_v3")

# =========================================================
# RESET ENVIRONNEMENT
# =========================================================

obs, info = env.reset()

# =========================================================
# BOUCLE TEST PPO
# =========================================================

while True:

    # PPO choisit une action
    action, _states = model.predict(obs)

    # Exécute action dans le jeu
    obs, recompense, morte, tronque, info = env.step(action)

    # =====================================================
    # AFFICHAGE DES REWARDS
    # =====================================================

    if recompense > 0:
        print("Reward détectée :", recompense)

    # =====================================================
    # SI PARTIE TERMINÉE
    # =====================================================

    if morte or tronque:

        print("Fin épisode")

        # Reset nouvelle partie
        obs, info = env.reset()

# =========================================================
# FERMETURE ENVIRONNEMENT
# =========================================================

env.close()