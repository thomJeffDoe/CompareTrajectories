from compare_trajectories.data_handler import DataHandler
from compare_trajectories.train import train_autoencoder, training_plot
from compare_trajectories.utils import load_data_numpy, save_data_numpy, get_common_images, compare_obs
from compare_trajectories.trajectories_generation import extract_rewarding_states
import tensorflow.keras as keras
import os

ENV_NAME = "MineRLObtainDiamondVectorObf-v0"
REBUILD_DATA = True
path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
path_trajectories = os.path.join(path,"data/trajectories")
path_model = os.path.join(path,"data/autoencoder")


def train():
    history, model = train_autoencoder()
    training_plot(history)


def get_latent_vectors():
    povs = load_data_numpy(path_trajectories,"povs.npy")
    model = keras.models.load_model(path_model)
    latent_vectors = model.predict_latent(povs)
    save_data_numpy(latent_vectors,path_trajectories,'latents.npy')
    return latent_vectors


def main():
    data_handler = DataHandler(env_name=ENV_NAME,rebuild_data=REBUILD_DATA)
    latent_vectors = get_latent_vectors()
    povs, latents, actions, rewards, indexes = extract_rewarding_states(
        data_handler.rewards,
        data_handler.actions,
        data_handler.povs,
        latent_vectors
    )
    povs_merged = get_common_images(compare_obs(latents,indexes), indexes)
    



