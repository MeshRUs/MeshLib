import pytest
from helper import *


def createTestMesh() -> mrmesh.Mesh:
    # test simple triangulation
    triangulation = mrmesh.Triangulation()
    triangulation.vec.resize(2)
    assert len(triangulation.vec[0]) == 3
    for i in range(len(triangulation.vec[0])):
        triangulation.vec[0][i] = mrmesh.VertId(i)
        assert triangulation.vec[0][i].get() == i
    i = 0
    for v in triangulation.vec[0]:
        assert v.get() == i
        i = i + 1
    triangulation.vec[1] = mrmesh.ThreeVertIds([
        mrmesh.VertId(0), mrmesh.VertId(2), mrmesh.VertId(3)
    ])

    mesh = mrmesh.Mesh()
    mesh.points.vec.resize(4)
    mesh.points.vec[0] = mrmesh.Vector3f(0, 0, 0)
    mesh.points.vec[1] = mrmesh.Vector3f(1, 0, 0)
    mesh.points.vec[2] = mrmesh.Vector3f(1, 1, 0)
    mesh.points.vec[3] = mrmesh.Vector3f(0, 1, 0)
    mesh.topology = mrmesh.topologyFromTriangles(triangulation)

    # Basic assertions to verify the mesh is created correctly
    assert mesh.topology.numValidFaces() == 2
    assert mesh.topology.numValidVerts() == 4

    return mesh


def test_basicHalfEdgeOperations():
    mesh = createTestMesh()

    # Edge 0: 0 -> 1
    edge = mrmesh.EdgeId(0)

    assert mesh.topology.hasEdge(edge)
    assert not mesh.topology.hasEdge(mrmesh.EdgeId(100))

    assert mesh.topology.dest(edge).get() == 1  # Goes to vertex 1
    assert mesh.topology.org(edge).get() == 0  # Comes from vertex 0

    assert mesh.topology.prev(edge).get() == 9  # Previous edge is 9
    assert mesh.topology.next(edge).get() == 5  # Next edge is 1

    assert mesh.topology.left(edge).get() == 0  # Left face is 0
    assert mesh.topology.right(edge).get() == -1  # Has no right face

    assert mesh.topology.getOrgDegree(edge) == 3
    assert mesh.topology.getVertDegree(mrmesh.VertId(0)) == 3
    assert mesh.topology.getLeftDegree(edge) == 3
    assert mesh.topology.getFaceDegree(mrmesh.FaceId(0)) == 3
