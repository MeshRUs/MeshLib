import shutil

import pytest
from helper import *


def test_save_voxels():
    mesh = mrmesh.makeTorus()
    mtvParams = mrmesh.MeshToVolumeParams()
    mtvParams.type = mrmesh.MeshToVolumeParamsType.Signed
    mtvParams.surfaceOffset = 3
    meshBox = mesh.computeBoundingBox()
    boxSize = meshBox.max - meshBox.min
    mtvParams.voxelSize = boxSize / 27.0
    voxels = mrmesh.meshToVolume(mesh, mtvParams)

    vsParams = mrmesh.VoxelsSaveSavingSettings()
    shutil.rmtree("save_voxels_dir_test", ignore_errors=True) # Remove existing directory, if any. Otherwise `mkdir()` fails.
    os.mkdir("save_voxels_dir_test")
    vsParams.path = "save_voxels_dir_test"
    vsParams.slicePlane = mrmesh.SlicePlane.XY
    mrmesh.saveAllSlicesToImage(voxels, vsParams)
    shutil.rmtree("save_voxels_dir_test")
