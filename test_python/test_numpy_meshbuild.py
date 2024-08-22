import unittest as ut

import numpy as np
import pytest
from helper import *


def test_numpy_meshbuild():
    faces = np.ndarray(
        shape=(2, 3),
        dtype=np.int32,
        buffer=np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32),
    )

    # mrmesh uses float32 for vertex coordinates
    # however, you could also use float64
    verts = np.ndarray(
        shape=(4, 3),
        dtype=np.float32,
        buffer=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    mesh = mrmeshnumpy.meshFromFacesVerts(faces, verts)

    assert mesh.topology.getValidFaces().count() == 2
    a = mrmesh.VertId(0)
    b = mrmesh.VertId(0)
    c = mrmesh.VertId(0)
    mesh.topology.getTriVerts(mrmesh.FaceId(0), a, b, c)
    assert a.get() == 0
    assert b.get() == 1
    assert c.get() == 2

    assert mesh.points.vec.size() == 4
    np.testing.assert_almost_equal(mesh.points.vec[0].z, 0.0)
    np.testing.assert_almost_equal(mesh.points.vec[2].x, 1.0)


def test_numpy_makeManifold():

    # Create a non-manifold mesh
    verts = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0], [0.5, 0.5, 0]])
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 3, 1], [0, 4, 1]])

    # Build the mesh allowing non-manifold edges
    m1 = mrmeshnumpy.meshFromFacesVerts(faces, verts, duplicateNonManifoldVertices = False )
    assert m1.topology.getValidFaces().count() == 3

    # Build the mesh again without allowing manifold edges
    settings = mrmesh.MeshBuilderSettings()
    settings.allowNonManifoldEdge = False

    m2 = mrmeshnumpy.meshFromFacesVerts(faces, verts, settings, duplicateNonManifoldVertices = False)
    assert m2.topology.getValidFaces().count() == 2

    # Build the mesh againd with duplication (True by default)
    m3 = mrmeshnumpy.meshFromFacesVerts(faces, verts)
    assert m3.topology.getValidFaces().count() == 4
