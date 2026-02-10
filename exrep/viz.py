from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    R = dendrogram(linkage_matrix, **kwargs)
    return R

def get_cluster_leaves(cluster_ids: Sequence[int], clustering):
    # this is O(n^2) but it's fine for n <= 10000
    subtrees = [[] for _ in range(len(clustering.children_))]
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_subtree = []
        for child_idx in merge:
            if child_idx < n_samples:
                current_subtree.append(child_idx)   # leaf node
            else:
                current_subtree.extend(subtrees[child_idx - n_samples])
        subtrees[i] = np.array(current_subtree)
    
    cluster_labels = np.zeros(n_samples)
    for cluster in cluster_ids:
        cluster_labels[subtrees[cluster - n_samples]] = cluster
    
    return cluster_labels, subtrees

def get_cluster_children(clustering, cluster_ids: Sequence[int], targets=Sequence[int]):
    subtrees = {
        c: [] for c in cluster_ids
    }
    n_samples = len(clustering.labels_)
    for cluster in cluster_ids:
        q = [cluster]
        while q:
            node = q.pop()
            if node in targets:
                subtrees[cluster].append(node)
            else:
                q.extend(clustering.children_[node - n_samples].tolist())
    assert sum(len(v) for v in subtrees.values()) == len(targets), "Only found %d/%d targets" % (sum(len(v) for v in subtrees.values()), len(targets))
    return subtrees

def viz_clusters(clusters, cluster_labels, crops_dataset, preview_size=8, shuffle=False):
    plt.close('all')
    for cluster_id in clusters:
        indices = np.where(cluster_labels == cluster_id)[0]
        if shuffle:
            rng = np.random.default_rng()
            indices = rng.permutation(indices)
        fig, axs = plt.subplots(1, preview_size, figsize=(20, 2))
        fig.suptitle(cluster_id)
        for ax, i in zip(axs, indices):
            display_im = crops_dataset[int(i)]["patches"].resize((80, 80))
            ax.imshow(display_im)
            ax.axis("off")