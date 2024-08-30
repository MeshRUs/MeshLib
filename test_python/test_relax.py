import pytest
from helper import *


def test_relax():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshRelaxParams()
    params.iterations = 5
    res = mrmesh.relax(torus, params)

    assert res


def test_relax_keep_volume():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    keep_volume_torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshRelaxParams()
    params.iterations = 5
    res = mrmesh.relaxKeepVolume(keep_volume_torus, params)

    assert res

def test_relax_approx():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshApproxRelaxParams()
    params.iterations = 5
    res = mrmesh.relaxApprox(torus, params)

    assert res

def test_smooth_region_boundary():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    keep_volume_torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    smooth_region = mrmesh.FaceBitSet()
    smooth_region.resize( 6, True )

    # This just checks that the function exists and can be called.
    mrmesh.smoothRegionBoundary(keep_volume_torus, smooth_region)

def test_straighten_boundary():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)

    torus.topology.deleteFaces(faceBitSetToDelete)

    holes = torus.topology.findHoleRepresentiveEdges()
    mrmesh.straightenBoundary(torus, holes[0], 13, 5)

def test_relax_inside():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    faceBitSetToDelete = mrmesh.FaceBitSet()
    faceBitSetToDelete.resize(5, False)
    faceBitSetToDelete.set(mrmesh.FaceId(1), True)
    torus.deleteFaces(faceBitSetToDelete)

    bdVerts = mrmesh.getBoundaryVerts(torus.topology)
    mrmesh.expand(torus.topology, bdVerts, 5);
    region = torus.topology.getValidVerts() - bdVerts;

    params = mrmesh.MeshApproxRelaxParams()
    params.region = region
    res = mrmesh.relaxApprox(torus, params)
    assert( res )

test_relax_inside()
