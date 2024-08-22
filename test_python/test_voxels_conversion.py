import unittest as ut

import numpy as np
import pytest
import math
from helper import *


def test_voxels_conversion():
    # Create array containing one octant with spherical distribution
    mul = 1  # Data size multiplier, set to larger value to test on larger arrays
    np_array = np.zeros((12 * mul, 15 * mul, 20 * mul))
    max_value = 1000 * mul * mul
    center = (np_array.shape[0] / 12, np_array.shape[1] / 3, np_array.shape[2] / 3)
    for k in range(np_array.shape[2]):
        for j in range(np_array.shape[1]):
            for i in range(np_array.shape[0]):
                x, y, z = i - center[0], j - center[1], k - center[2]
                np_array[i, j, k] = 0.0 if x < 0 or y < 0 or z < 0 else \
                        max_value - (x * x + y * y + z * z)

    # Create initial volume object
    volume0 = mrmeshnumpy.simpleVolumeFrom3Darray(np_array)
    volume0.voxelSize = mrmesh.Vector3f(0.5, 1, 1)
    volume0.min, volume0.max = 0.0, max_value

    # Test conversion functions
    vdb_volume = mrmesh.simpleVolumeToVdbVolume(volume0)
    simple_volume = mrmesh.vdbVolumeToSimpleVolume(vdb_volume)
    grid = mrmesh.simpleVolumeToDenseGrid(simple_volume)
    # Basic results validation
    assert volume0.dims == vdb_volume.dims == simple_volume.dims
    assert volume0.voxelSize == vdb_volume.voxelSize == simple_volume.voxelSize
    assert volume0.min == vdb_volume.min == simple_volume.min
    assert volume0.max == vdb_volume.max == simple_volume.max

    # Test mesh build functions
    radius = 8 * mul
    mesh1 = mrmesh.gridToMesh(vdb_volume, isoValue = max_value - radius * radius)
    mesh2 = mrmesh.gridToMesh(grid, voxelSize = volume0.voxelSize, isoValue = max_value - radius * radius)
    # Basic checks
    for mesh in (mesh1, mesh2):
        assert len(mrmesh.getAllComponents(mesh1)) == 1
        assert mesh.topology.numValidFaces() + mesh.topology.numValidVerts() - \
                mesh.topology.computeNotLoneUndirectedEdges() == 2  # Euler characteristic
    # Check by roughly comparing mesh volume to expectation (volume of 1/8 sphere)
    vol = radius ** 3 * math.pi / 6 * (volume0.voxelSize.x * volume0.voxelSize.y * volume0.voxelSize.z)
    for mesh in (mesh1, mesh2):
        assert 0.9 < mesh.volume() / vol < 1.1
