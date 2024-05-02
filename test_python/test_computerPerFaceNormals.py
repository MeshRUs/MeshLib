import pytest
from helper import *


def test_compute_per_face_normals():
    torus = mrmesh.makeTorus(2, 1, 10, 10)
    normals = mrmesh.computePerFaceNormals(torus)
