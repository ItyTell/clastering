# k-means clustering algorithm implementation in Python

import numpy as np

def k_means(dots:list, k:int, max_iter:int=300):
    """
    Perform k-means clustering on a python list of points with max number of iteration and condition that if the centers dont move it stops - they will never move."""

    dots = np.array(dots)
    centers = dots[np.random.choice(dots.shape[0], k, replace=False)]

    for i in range(max_iter):
        distances = np.linalg.norm(dots[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([dots[labels == j].mean(axis=0) for j in range(k)])

        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels