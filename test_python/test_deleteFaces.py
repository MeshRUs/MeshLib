import pytest
from helper import *


def test_delete_faces():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)
    oldFaceBS = torus.topology.getValidFaces()
    torus.topology.deleteFaces(faceBitSetToDelete)
    deletedBitSet = oldFaceBS - torus.topology.getValidFaces()
    assert deletedBitSet.count() == 1
    assert deletedBitSet.test(mrmesh.FaceId(1))

    # delete_faces(...) method of builtins.PyCapsule instance
    # delete_faces(arg0: mrmesh.MeshTopology, arg1: mrmesh.FaceBitSet) ->
    # None
