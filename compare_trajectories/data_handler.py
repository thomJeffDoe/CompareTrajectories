import minerl
import logging
from tqdm import tqdm
from compare_trajectories.utils import save_data_numpy, load_data_numpy
from compare_trajectories.kmeans import apply_kmeans
import os

path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.basicConfig(level=logging.INFO)


class DataHandler:
    def __init__(self, env_name, rebuild_data=False, size=120):
        self.env_name = env_name
        self.size = size
        if rebuild_data:
            self.actions, self.povs, self.rewards = self.load_data()
            self.actions = self.discretize_actions()
            save_data_numpy(
                self.actions, os.path.join(path, "data/trajectories"), "actions.npy"
            )
            save_data_numpy(
                self.povs, os.path.join(path, "data/trajectories"), "povs.npy"
            )
            save_data_numpy(
                self.rewards, os.path.join(path, "data/trajectories"), "rewards.npy"
            )
        else:
            self.actions = load_data_numpy(
                os.path.join(path, "data/trajectories"), "actions.npy"
            )
            self.povs = load_data_numpy(
                os.path.join(path, "data/trajectories"), "povs.npy"
            )
            self.rewards = load_data_numpy(
                os.path.join(path, "data/trajectories"), "rewards.npy"
            )

    def download_data(self):
        """
        Download the minerl training data.
        Requires that the MINERL_DATA_ROOT
        env variable has properly been set
        Parameter
        ---------
            - env_name : `str`
                the name of the minerl environment
                to download
        """
        logging.info("Downloading Data")
        minerl.data.download(environment=self.env_name)
        logging.info("Data Downloaded")

    def load_data(self):
        try:
            data = minerl.data.make(self.env_name, num_workers=1)
            trajectory_names = data.get_trajectory_names()
        except KeyError:
            raise ValueError("Wrong Environment name")
        except FileNotFoundError:
            self.download_data()
            data = minerl.data.make(self.env_name, num_workers=1)
            trajectory_names = data.get_trajectory_names()
        actions = []
        povs = []
        rewards = []
        for index, trajectory_name in enumerate(tqdm(trajectory_names)):
            trajectory = data.load_data(
                trajectory_name, skip_interval=10, include_metadata=False
            )
            action, pov, reward = self.isolate(trajectory)
            actions.append(action)
            povs.append(action)
            rewards.append(action)
            if len(actions) >= self.size:
                break
        return actions, povs, rewards

    @staticmethod
    def isolate(raw_trajectory):
        rewards = []
        actions = []
        povs = []
        for step in raw_trajectory:
            rewards.append(step[2])
            povs.append(step[0]["pov"])
            actions.append(step[1]["vector"])
        return actions, povs, rewards

    def discretize_actions(self):
        actions_flatten = [i for j in self.actions for i in j]
        predictions = apply_kmeans(actions_flatten)
        count_actions_per_traj = 0
        actions_predicted = []
        for idx in range(len(self.actions)):
            actions_predicted.append(
                predictions[
                    count_actions_per_traj : count_actions_per_traj
                    + len(self.actions[idx])
                ]
            )
            count_actions_per_traj = len(self.actions[idx])
        return actions_predicted


