import numpy as np
import logging
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


def save_data_numpy(data, path, filename):
    logging.info(f"Saving {filename} ...")
    np.save(os.path.join(path, filename), data)


def load_data_numpy(path, filename):
    logging.info(f"Loading {filename} ...")
    return np.load(os.path.join(path, filename), allow_pickle=True)


def get_common_images(common_elements, index_povs):
    aliases = {}
    for i in index_povs:
        for common in common_elements:
            if i in common:
                aliases[i] = min(common)
                break
    return aliases


def compare_obs(latent_povs,index_povs, threshold=0.85):
    """
    get lists of identical observations based on
    cosine distance of latent representations
    """
    cosines = cosine_similarity(latent_povs, latent_povs)
    commons = []
    with tqdm(total=len(index_povs),position=0,leave=True,desc='Cosines') as pbar:
        for idx_out, x in zip(index_povs,cosines):
            common = []
            for idx_in, y in zip(index_povs,x):
                if y >= threshold:
                    common.append(idx_in)
            if len(common) > 0:
                common.append(idx_out)
            commons.append(common)
            pbar.update(1)
    common_elements = merge_common(commons)
    return list(common_elements)


def merge_common(lists):
    neigh = defaultdict(set)
    visited = set()
    for each in tqdm(lists,desc='Merge1',position=0,leave=True):
        for item in each:
            neigh[item].update(each)

    def comp(node, neigh=neigh, visited=visited, vis=visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node

    for node in tqdm(neigh,desc='Merge2',position=0,leave=True):
        if node not in visited:
            yield sorted(comp(node))