import unittest as ut

import numpy as np
import pytest
from helper import *


def test_numpy_conversoin():
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

    vertNorms = mrmesh.computePerVertNormals(mesh)
    faceNorms = mrmesh.computePerFaceNormals(mesh)

    vertNormsNp = mrmeshnumpy.toNumpyArray(vertNorms)
    faceNormsNp = mrmeshnumpy.toNumpyArray(faceNorms)

    assert vertNormsNp.shape[0] == len(vertNorms.vec)
    assert faceNormsNp.shape[0] == len(faceNorms.vec)
