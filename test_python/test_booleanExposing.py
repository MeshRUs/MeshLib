import pytest
from helper import *


def is_equal_vector3(a, b):
    diff = a - b
    return diff.length() < 1.0e-6



def test_unite_may_meshes():
    size = mrmesh.Vector3f.diagonal(2)
    poses = [
        mrmesh.Vector3f.diagonal(-1),
        mrmesh.Vector3f.diagonal(0),
        mrmesh.Vector3f.diagonal(1),
        mrmesh.Vector3f.diagonal(2),
    ]
    meshes = []
    vecMeshes = mrmesh.vectorConstMeshPtr()
    vecMeshes.resize(len(poses))
    for i in range(len(poses)):
        meshes.append(mrmesh.makeCube(size, poses[i]))
        vecMeshes[i] = meshes[i]
    params = mrmesh.UniteManyMeshesParams()
    params.nestedComponentsMode = mrmesh.NestedComponenetsMode.Remove
    resMesh = mrmesh.uniteManyMeshes(vecMeshes)
    assert resMesh.topology.numValidFaces() > 0
    assert resMesh.topology.findHoleRepresentiveEdges().size() == 0

def test_intersection_contours():
    size = mrmesh.Vector3f.diagonal(2)
    pos1 = mrmesh.Vector3f.diagonal(0)
    pos2 = mrmesh.Vector3f.diagonal(-1)
    pos3 = mrmesh.Vector3f.diagonal(1)

    meshA = mrmesh.makeCube(size, pos1)
    meshB = mrmesh.makeCube(size, pos2)

    conv = mrmesh.getVectorConverters(meshA,meshB)
    intersections = mrmesh.findCollidingEdgeTrisPrecise(meshA,meshB,conv.toInt)
    orderedIntersections = mrmesh.orderIntersectionContours(meshA.topology,meshA.topology,intersections)
    aConts = mrmesh.getOneMeshIntersectionContours(meshA,meshB,orderedIntersections,True,conv)
    bConts = mrmesh.getOneMeshIntersectionContours(meshA,meshB,orderedIntersections,False,conv)
    assert aConts[0].intersections.size() > 0
    assert aConts[0].closed
    assert aConts[0].intersections.size() == bConts[0].intersections.size()
    assert bConts[0].closed
