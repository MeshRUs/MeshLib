import pytest
from helper import *

def test_contour_stitch():
    """
    Stest exposing of `stitchContours` and `cutAlongEdgeLoop` functions.
    """

    mesh = mrmesh.makeCube()
    topology = mesh.topology
    ueCntA = topology.computeNotLoneUndirectedEdges()

    c0 = mrmesh.vectorEdges()
    c0.append(mrmesh.EdgeId(0))
    c0.append(mrmesh.EdgeId(2))
    c0.append(mrmesh.EdgeId(4))

    c1 = mrmesh.cutAlongEdgeLoop( mesh.topology, c0 )

    ueCntB = topology.computeNotLoneUndirectedEdges()
    assert ueCntB == ueCntA + 3

    mrmesh.stitchContours( mesh.topology, c0, c1 )
    ueCntC = topology.computeNotLoneUndirectedEdges()
    assert ueCntC == ueCntA
