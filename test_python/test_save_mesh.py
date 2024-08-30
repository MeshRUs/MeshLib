import pytest
from helper import *


def test_save_load_mesh():
    torus = mrmesh.makeTorus(2, 1, 10, 10, None)
    fs = open("test_save_load_mesh.mrmesh", "wb")
    try:
        mrmesh.saveMesh(torus, fs, "*.mrmesh")
    except ValueError as e:
        print(e)
        assert False

    fs.close()
    fr = open("test_save_load_mesh.mrmesh", "rb")
    try:
        torus2 = mrmesh.loadMesh(fr, "*.mrmesh")
    except ValueError as e:
        print(e)
        assert False

    fr.close()
    assert torus2 == torus
    os.remove("test_save_load_mesh.mrmesh")
