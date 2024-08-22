import pytest
from helper import *



def test_general_offset_mesh():
    cube = mrmesh.makeCube()
    try:
        cube_offset = mrmesh.generalOffsetMesh(cube, 0.1)
    except ValueError as e:
        print(e)
        assert False

    assert cube_offset.volume() / cube.volume() > 1.0


def test_thicken_mesh():
    cube = mrmesh.makeCube()
    removeFaces = mrmesh.FaceBitSet()
    removeFaces.resize(2, False)
    removeFaces.set(mrmesh.FaceId(0), True)
    removeFaces.set(mrmesh.FaceId(1), True)
    cube.topology.deleteFaces(removeFaces)
    cube.invalidateCaches()

    params = mrmesh.OffsetParameters()
    params.signDetectionMode = mrmesh.SignDetectionMode.Unsigned
    params.voxelSize = mrmesh.suggestVoxelSize(cube, 5e6)
    try:
        cube_thicken = mrmesh.thickenMesh(cube, 0.1, params)
    except ValueError as e:
        print(e)
        assert False
    assert cube_thicken.topology.findHoleRepresentiveEdges().size() == 2

def test_double_offset_mesh():
    cube = mrmesh.makeCube()
    params = mrmesh.OffsetParameters()
    params.voxelSize = mrmesh.suggestVoxelSize(cube, 5e6)
    try:
        cube_offset = mrmesh.doubleOffsetMesh(cube, -0.2, 0.2)
    except ValueError as e:
        print(e)
        assert False

    assert cube_offset.volume() / cube.volume() < 1.0
