import unittest as ut

import numpy as np
import pytest
from helper import *


# mrmesh uses float32 for vertex coordinates
# however, you could also use float64
def test_numpy_pointsbuild1():
    verts = np.ndarray(
        shape=(4, 3),
        dtype=np.float32,
        buffer=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    pc = mrmeshnumpy.pointCloudFromPoints(verts)
    assert pc.validPoints.count() == 4
    assert pc.normals.vec.size() == 0

    # mrmesh uses float32 for vertex coordinates
    # however, you could also use float64

    norms = np.ndarray(
        shape=(4, 3),
        dtype=np.float32,
        buffer=np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )

    pc = mrmeshnumpy.pointCloudFromPoints(verts, norms)
    assert pc.validPoints.count() == 4
    assert pc.normals.vec.size() == 4


def test_numpy_points_mesh():
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Prepare for MeshLib PointCloud
    verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
    # Create MeshLib PointCloud from np ndarray
    pc = mrmeshnumpy.pointCloudFromPoints(verts)
    # Remove duplicate points
    settings = mrmesh.UniformSamplingSettings()
    settings.distance = 1e-3
    pc.validPoints = mrmesh.pointUniformSampling(pc, settings)
    pc.invalidateCaches()

    # Triangulate it
    triangulatedPC = mrmesh.triangulatePointCloud(pc)
    assert triangulatedPC.topology.findHoleRepresentiveEdges().size() == 0
    assert mrmesh.getAllComponents(triangulatedPC).size() == 1
    assert triangulatedPC.volume() > 0

def test_numpy_grid_sampling():
    # Start from numpy arrays
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Prepare for MeshLib PointCloud
    verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
    # Create MeshLib PointCloud from np ndarray
    pc = mrmeshnumpy.pointCloudFromPoints(verts)
    assert(pc.validPoints.count() == 10000)
    # Sample points
    samples = mrmesh.pointGridSampling(pc, 0.1)
    assert(samples.count() == 1524)

    # Other sampling
    samples2 = mrmesh.pointIterativeSampling(pc, 1000)
    assert(samples2.count() == 1000)

    # Keep only sampled points
    pc.validPoints = samples
    pc.pack()
    # All remaining points are valid now

    # Convert back into numpy array
    pts = mrmeshnumpy.toNumpyArray(pc.points)
