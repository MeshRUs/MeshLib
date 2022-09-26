from helper import *
import pytest

def isEqualVector3(a, b):
    diff = a - b
    return diff.length() < 1.e-6


# TEST 1
def decimate1(size, pos1, pos2, pos3):
    mesh = mrmesh.makeCube(size, pos1)
    settings = mrmesh.DecimateSettings()

    result = mrmesh.decimateMesh( mesh, settings )

    assert( result.vertsDeleted == 0 )
    assert( result.facesDeleted == 0 )
    # assert( result.errorIntroduced == 0 )

    assert( mesh.topology.getValidVerts().size() == 8 )
    assert( mesh.topology.getValidVerts().count() == 8 )
    assert( mesh.topology.findHoleRepresentiveEdges().size() == 0 )


# TEST 2
def decimate2(size, pos1, pos2, pos3):
    meshA = mrmesh.makeCube(size, pos1)
    meshB = mrmesh.makeCube(size, pos2)

    bOperation = mrmesh.BooleanOperation.Intersection
    bResMapper = mrmesh.BooleanResultMapper()
    bResult = mrmesh.boolean( meshA, meshB, bOperation, None, bResMapper )

    mesh = bResult.mesh
    settings = mrmesh.DecimateSettings()

    result = mrmesh.decimateMesh( mesh, settings )

    assert( isEqualVector3( mesh.computeBoundingBox(mesh.topology.getValidFaces(), mrmesh.AffineXf3f() ).min , pos1 ) )
    assert( isEqualVector3( mesh.computeBoundingBox(mesh.topology.getValidFaces(), mrmesh.AffineXf3f() ).max , pos3 ) )

    assert( result.vertsDeleted == 6 )
    assert( result.facesDeleted == 12 )
    # assert( result.errorIntroduced == 0 )

    assert( mesh.topology.getValidVerts().size() == 14 )
    assert( mesh.topology.getValidVerts().count() == 8 )
    assert( mesh.topology.findHoleRepresentiveEdges().size() == 0 )


def test_deciamte():
    size = mrmesh.Vector3f.diagonal( 2 )
    pos1 = mrmesh.Vector3f.diagonal( 0 )
    pos2 = mrmesh.Vector3f.diagonal( -1 )
    pos3 = mrmesh.Vector3f.diagonal( 1 )
    decimate1(size, pos1, pos2, pos3)
    decimate2(size, pos1, pos2, pos3)