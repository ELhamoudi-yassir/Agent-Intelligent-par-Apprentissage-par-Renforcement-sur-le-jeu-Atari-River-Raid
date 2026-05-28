import ale_py
import gymnasium as gym

from gymnasium import Wrapper
from stable_baselines3 import PPO


# =========================================================
# CUSTOM REWARD WRAPPER
# =========================================================
# Ce wrapper permet de modifier les rewards du jeu
# avant qu'elles soient envoyées à PPO.
#
# Objectif :
#  - bonus fuel
# - encourager la survie
# - pénaliser la mort

# =========================================================

class CustomRewardWrapper(Wrapper):

    # Initialisation du wrapper
    def __init__(self, env):
        super().__init__(env)

    # Fonction appelée à chaque action PPO
    def step(self, action):

        # Exécute l'action dans le jeu
        # et récupère les informations du jeu
        obs, recompense, morte, tronque, info = self.env.step(action)
      
        # BONUS fuel

        if recompense == 30: # si IA a toucher le fuel ajoute bonus
           recompense+=3 

        # BONUS SURVIE 

        if not morte :#si IA survie en donne une recomponse 
            recompense += 0.2

        # PÉNALITÉ MORT
        if morte : #si IA et morte en donne une penalite
            recompense -= 20

      
        # Retourne les nouvelles données à PPO
        return    obs, recompense, morte, tronque, info 



# CRÉATION ENVIRONNEMENT


env = gym.make(
    "ALE/Riverraid-v5",
    render_mode=None
)

# Application du wrapper custom
env = CustomRewardWrapper(env)


# =========================================================
# CRÉATION MODÈLE PPO
# =========================================================

model = PPO(

    # CNN pour analyser les images Atari
    "CnnPolicy",

    env,

    # Vitesse d'apprentissage
    learning_rate=0.00025,

    # Importance des rewards futures
    gamma=0.99,

    # Nombre d'étapes avant update PPO
    n_steps=128,

    # Taille batch apprentissage
    batch_size=64,

    # Encourage exploration
    ent_coef=0.01,

    verbose=1
)

# =========================================================
# ENTRAÎNEMENT PPO avec reward + carburant
# =========================================================

model.learn(total_timesteps=500000)



# SAUVEGARDE MODÈLE


model.save("riverraid_ppo_v3")



# FERMETURE ENVIRONNEMENT


env.close()