import numpy as np
from compare_trajectories.utils import save_data_numpy
import os

path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
path_trajectories = os.path.join(path,"data/trajectories")


def extract_rewarding_states(
    rewards,
    actions,
    povs,
    latent_vectors
):
    assert len(rewards)==len(actions)==len(povs)==len(latent_vectors)
    rewards_filtered = []
    actions_filtered = []
    latent_vectors_filtered = []
    povs_filtered = []
    indexes = []
    old_index = 0
    for trajectory_index in range(len(rewards)):
        index = list(np.where(np.array(rewards[trajectory_index]) > 0.0)[0])
        rew_filt = []
        pov_filt = []
        action_filt = []
        latent_filt = []
        indexes.append(list(np.array(index) + old_index))
        old_index += len(povs[trajectory_index])
        for i in index:
            rew_filt.append(rewards[trajectory_index][:i + 1])
            pov_filt.append(povs[trajectory_index][i])
            action_filt.append(actions[trajectory_index][i])
            latent_filt.append(latent_vectors[trajectory_index][i])
        rewards_filtered.append(rew_filt)
        actions_filtered.append(action_filt)
        latent_vectors_filtered.append(latent_filt)
        povs_filtered.append(pov_filt)
    save_data_numpy(
        rewards_filtered,
        path_trajectories,
        "rewards_filtered.npy"
    )
    save_data_numpy(
        indexes,
        path_trajectories,
        "indexes.npy"
    )
    save_data_numpy(
        actions_filtered,
        path_trajectories,
        "actions_filtered.npy"
    )
    save_data_numpy(
        latent_vectors_filtered,
        path_trajectories,
        "latent_vectors_filtered.npy"
    )
    save_data_numpy(
        povs_filtered,
        path_trajectories,
        "povs_filtered.npy"
    )
    return povs_filtered, latent_vectors_filtered, actions_filtered, rewards_filtered, indexes



