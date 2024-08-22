import pytest
from helper import *


def is_equal_vector_3(a, b):
    diff = a - b
    return diff.length() < 1.0e-6


# TEST 1
def decimate_1(size, pos1, pos2, pos3):
    mesh = mrmesh.makeCube(size, pos1)
    settings = mrmesh.DecimateSettings()

    result = mrmesh.decimateMesh(mesh, settings)

    assert result.vertsDeleted == 0
    assert result.facesDeleted == 0
    # assert( result.errorIntroduced == 0 )

    assert mesh.topology.getValidVerts().size() == 8
    assert mesh.topology.getValidVerts().count() == 8
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0


# TEST 2
def decimate_2(size, pos1, pos2, pos3):
    meshA = mrmesh.makeCube(size, pos1)
    meshB = mrmesh.makeCube(size, pos2)

    bOperation = mrmesh.BooleanOperation.Intersection
    bResMapper = mrmesh.BooleanResultMapper()
    bResult = mrmesh.boolean(meshA, meshB, bOperation, None, bResMapper)

    mesh = bResult.mesh
    settings = mrmesh.DecimateSettings()

    result = mrmesh.decimateMesh(mesh, settings)

    assert is_equal_vector_3(
        mesh.computeBoundingBox(mesh.topology.getValidFaces(), mrmesh.AffineXf3f()).min,
        pos1,
    )
    assert is_equal_vector_3(
        mesh.computeBoundingBox(mesh.topology.getValidFaces(), mrmesh.AffineXf3f()).max,
        pos3,
    )

    assert result.vertsDeleted == 6
    assert result.facesDeleted == 12
    # assert( result.errorIntroduced == 0 )
    assert mesh.topology.getValidVerts().size() == 14
    assert mesh.topology.getValidVerts().count() == 8
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0
    mesh.pack()


def test_deciamte():
    size = mrmesh.Vector3f.diagonal(2)
    pos1 = mrmesh.Vector3f.diagonal(0)
    pos2 = mrmesh.Vector3f.diagonal(-1)
    pos3 = mrmesh.Vector3f.diagonal(1)
    decimate_1(size, pos1, pos2, pos3)
    decimate_2(size, pos1, pos2, pos3)


def test_remesh():
    mesh = mrmesh.makeTorus(2, 1, 10, 10, None)

    settings = mrmesh.RemeshSettings()
    settings.maxEdgeSplits = 1000

    nb_verts_before = mesh.topology.getValidVerts().size()

    result = mrmesh.remesh(mesh, settings)

    # Remeshing is successful
    assert result == True

    # Assume edge splits are executed
    assert mesh.points.vec.size() > nb_verts_before
    assert mesh.topology.getValidVerts().size() > nb_verts_before

    # No holes are introduced
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0
