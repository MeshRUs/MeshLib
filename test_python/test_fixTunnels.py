import pytest
from helper import *


def test_fix_tunnels():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)

    settings = mrmesh.DetectTunnelSettings()
    settings.maxTunnelLength = 100500

    tunnelFaces = mrmesh.detectTunnelFaces(torus, settings)

    # one circle with 2-faces width
    assert tunnelFaces.count() == 20
