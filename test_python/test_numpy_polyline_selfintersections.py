import unittest as ut

import numpy as np
import pytest
from helper import *


# mrmesh uses float32 for vertex coordinates
# however, you could also use float64
def test_numpy_polyline_selfintersections():
    points = np.ndarray(
        shape=(4, 2),
        dtype=np.float32,
        buffer=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, -0.5]], dtype=np.float32
        ),
    )
    polyline = mrmeshnumpy.polyline2FromPoints(points)
    selfInters = mrmesh.findSelfCollidingEdges(polyline)
    assert selfInters.size() == 1
    segm1 = polyline.edgeSegment(mrmesh.EdgeId(selfInters[0].aUndirEdge))
    segm2 = polyline.edgeSegment(mrmesh.EdgeId(selfInters[0].bUndirEdge))
    inter = mrmesh.intersection(segm1, segm2)
    assert abs(inter.x - 2.0 / 3.0) < 0.001 and inter.y == 0
