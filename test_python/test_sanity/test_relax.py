from  module_helper import *



def test_relax():
    R1 = 2
    R2_1 = 1
    R2_2 = 2.5
    torus = mrmesh.makeTorusWithSpikes(R1, R2_1, R2_2, 10, 12, None)

    params = mrmesh.MeshRelaxParams()
    params.iterations = 5
    res = mrmesh.relax(torus, params)

    assert (res)
