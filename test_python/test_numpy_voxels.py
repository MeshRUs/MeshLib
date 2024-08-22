import unittest as ut

import numpy as np
import pytest
from helper import *


def test_numpy_voxels():
    sphere = mrmesh.makeSphere(mrmesh.SphereParams())
    params = mrmesh.MeshToDistanceVolumeParams()
    voxelSize = 0.01
    box = sphere.computeBoundingBox()
    expansion = mrmesh.Vector3f.diagonal(3 * voxelSize)
    params.vol.origin = box.min - expansion
    params.vol.voxelSize = mrmesh.Vector3f.diagonal(0.01)
    dimensionsF = (box.max + expansion - params.vol.origin) / voxelSize
    params.vol.dimensions = mrmesh.Vector3i(
        int(dimensionsF.x), int(dimensionsF.y), int(dimensionsF.z)
    ) + mrmesh.Vector3i.diagonal(1)
    params.dist.signMode = mrmesh.SignDetectionMode.HoleWindingRule
    params.dist.maxDistSq = 3 * voxelSize
    volume = mrmesh.meshToDistanceVolume(sphere, params)
    npArray = mrmeshnumpy.getNumpy3Darray(volume)
    assert npArray.shape[0] == params.vol.dimensions.x
    assert npArray.shape[1] == params.vol.dimensions.y
    assert npArray.shape[2] == params.vol.dimensions.z
