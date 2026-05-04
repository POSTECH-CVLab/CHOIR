"""Data augmentation transforms for point cloud processing."""

from typing import Literal

import numpy as np
import torch
import torch_cluster


class Compose:
    """Compose several transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


def _knn(queries: np.ndarray, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute k-nearest neighbors (numpy).

    Args:
        queries: (M, 3)
        points: (N, 3)
        k: number of neighbors.

    Returns:
        knn_points: (M, k, 3)
        knn_indices: (M, k)
    """
    rel = np.expand_dims(queries, 1) - np.expand_dims(points, 0)
    squared_dist = np.sum(rel**2, axis=-1)
    indices = np.argsort(squared_dist, axis=-1)[:, :k]
    knn_points = points[indices.ravel()].reshape(len(queries), k, -1)
    return knn_points, indices


class KnnPatchRemoval:
    """Remove k-nearest-neighbor patches from point clouds for augmentation."""

    def __init__(self, k: int, num_patches: int = 1):
        self.k = k
        self.num_patches = num_patches

    def __call__(self, coords: np.ndarray, feats=None, labels=None):
        N = len(coords)
        assert N > self.k * self.num_patches

        query_indices = np.random.choice(N, self.num_patches, replace=False)
        queries = coords[query_indices]

        _, knn_indices = _knn(queries, coords, self.k)
        mask = np.ones(N, dtype=bool)
        mask[knn_indices.ravel()] = False

        return_args = [coords[mask]]
        if feats is not None:
            return_args.append(feats[mask])
        if labels is not None:
            return_args.append(labels[mask])
        return tuple(return_args)


class PointSampling:
    """Sample a fixed number of points from the point cloud."""

    def __init__(self, num_points: int, sample_alg: Literal["fixed", "random", "fps"] = "random"):
        self.num_points = num_points
        self.sample_alg = sample_alg

    def __call__(self, coords: np.ndarray, feats=None, labels=None):
        N = len(coords)
        assert N > self.num_points

        if self.sample_alg == "fps":
            ratio = self.num_points / N
            sampled_indices = torch_cluster.fps(torch.Tensor(coords), ratio=ratio).numpy()
            assert len(sampled_indices) == self.num_points
        elif self.sample_alg == "random":
            sampled_indices = np.random.choice(N, size=self.num_points, replace=False)
        else:  # fixed
            sampled_indices = np.arange(self.num_points)

        return_args = [coords[sampled_indices]]
        if feats is not None:
            return_args.append(feats[sampled_indices])
        if labels is not None:
            return_args.append(labels[sampled_indices])
        return tuple(return_args)


class GaussianNoise:
    """Add Gaussian noise to point coordinates."""

    def __init__(self, noise_amp: float = 0.025):
        self.noise_amp = noise_amp

    def __call__(self, coords: np.ndarray, feats=None, labels=None):
        coords = coords + self.noise_amp * np.random.randn(*coords.shape)
        return_args = [coords]
        if feats is not None:
            return_args.append(feats)
        if labels is not None:
            return_args.append(labels)
        return tuple(return_args)
