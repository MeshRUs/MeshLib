import unittest as ut

import numpy as np
import pytest
from helper import *


# mrmesh uses float32 for vertex coordinates
# however, you could also use float64
def test_numpy_UVSphere():
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 100j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    mesh = mrmeshnumpy.meshFromUVPoints(x, y, z)
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0
    assert mrmesh.getAllComponents(mesh).size() == 1
    assert mesh.volume() > 0


def test_numpy_UVTorus():
    r1 = 2.0
    r2 = 1.5

    u, v = np.mgrid[0 : 2 * np.pi : 4j, 0 : 2 * np.pi : 10j]
    x = (r1 - r2 * np.cos(u)) * np.cos(v)
    y = (r1 - r2 * np.cos(u)) * np.sin(v)
    z = r2 * np.sin(u)

    mesh = mrmeshnumpy.meshFromUVPoints(x, y, z)
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0
    assert mrmesh.getAllComponents(mesh).size() == 1
    assert mesh.volume() > 0
