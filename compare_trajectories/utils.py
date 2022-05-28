import numpy as np
import logging
import os


def save_data_numpy(data,path,filename):
    logging.info(f"Saving {filename} ...")
    np.save(os.path.join(path, filename), data)


def load_data_numpy(path,filename):
    logging.info(f"Loading {filename} ...")
    return np.load(os.path.join(path, filename), allow_pickle=True)