from typing import Literal

import numpy as np
import torch
import torch_cluster


class Compose(object):
    r"""Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


def knn(queries, points, k, return_indices=False):
    r"""
    Args:
        queries: (M, 3), numpy.ndarray
        points: (N, 3), numpy.ndarray
        k: int
    Returns:
        knn_points: (M, k, 3)
        (optional) knn_indices: (M, k)
    """
    assert k <= len(points)
    rel = np.expand_dims(queries, 1) - np.expand_dims(points, 0) # (M, N, 3)
    squared_dist = np.sum(rel ** 2, axis=-1, keepdims=False) # (M, N)
    indices = np.argsort(squared_dist, axis=-1)[:, :k]

    knn_points = points[indices.ravel()].reshape(len(queries), k, -1)

    returns = [knn_points]
    if return_indices:
        returns.append(indices)
    
    return tuple(returns)


class KnnPatchRemoval(object):
    r"""kNN-based patch removal"""

    def __init__(self, k, num_patches):
        self.k = k
        self.num_patches = num_patches

    def __call__(self, coords, feats=None, labels=None):
        N = len(coords)
        assert N > self.k * self.num_patches

        query_indices = np.random.choice(range(N), self.num_patches, replace=False)
        queries = coords[query_indices]

        _, knn_indices = knn(queries, coords, self.k, return_indices=True)
        mask = np.ones(N, dtype=bool)
        mask[knn_indices.ravel()] = False

        return_args = [coords[mask]]
        if feats is not None:
            return_args.append(feats[mask])
        if labels is not None:
            return_args.append(labels[mask])

        return tuple(return_args)


class PointSampling(object):
    r""" Sample the fixed number of points randomly"""

    def __init__(self, num_points, sample_alg: Literal["fixed", "random", "fps"]):
        self.num_points = num_points
        self.sample_alg = sample_alg

    def __call__(self, coords, feats=None, labels=None):
        N = len(coords)
        assert N > self.num_points

        if self.sample_alg == "fps":            
            ratio = self.num_points / N
            sampled_indices = torch_cluster.fps(torch.Tensor(coords), ratio=ratio).numpy()
            assert len(sampled_indices) == self.num_points, f"# samples {len(sampled_indices)} != {self.num_points}"
        elif self.sample_alg == "random": 
            sampled_indices = np.random.choice(N, size=self.num_points, replace=False)
        else:
            sampled_indices = np.arange(self.num_points)

        return_args = [coords[sampled_indices]]
        if feats is not None:
            return_args.append(feats[sampled_indices])
        if labels is not None:
            return_args.append(labels[sampled_indices])

        return tuple(return_args)


class GaussianNoise(object):
    r""" Gaussian Noise (Ref: https://github.com/orenkatzir/VN-SPD/blob/5b2bed613585b731296409ed0ffed093e2923a97/models/shape_pose_model.py#L91)"""

    def __init__(self, noise_amp):
        self.noise_amp = noise_amp

    def __call__(self, coords, feats=None, labels=None):
        coords_translated = coords + self.noise_amp*np.random.randn(*coords.shape)

        return_args = [coords_translated]
        if feats is not None:
            return_args.append(feats)
        if labels is not None:
            return_args.append(labels)

        return tuple(return_args)