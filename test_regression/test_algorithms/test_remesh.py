from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlib

import pytest

@pytest.mark.smoke
@pytest.mark.parametrize("case", [
    {"name": "maxEdgeSplits", "params": {"maxEdgeSplits": 1000}},
    {"name": "finalRelaxIters", "params": {"maxEdgeSplits": 1000, "finalRelaxIters": 1}},
    {"name": "maxAngleChangeAfterFlip", "params": {"maxEdgeSplits": 1000, "maxAngleChangeAfterFlip": 50}},
    {"name": "maxBdShift", "params": {"maxEdgeSplits": 1000, "maxBdShift": 0.4}},
])
def test_remesh(tmp_path, case):
    """
    Test remesh algorithm
    """
    #  Load input mesh
    case_name = case["name"]
    input_folder = Path(test_files_path) / "algorithms" / "remesh"
    mesh = mlib.loadMesh(input_folder / "input.ctm")

    # Remesh
    params = case["params"]
    settings = mlib.RemeshSettings()
    for key, value in params.items():
        setattr(settings, key, value)

    result = mlib.remesh(mesh, settings)

    # === Verification
    assert result == True
    assert mesh.topology.findHoleRepresentiveEdges().size() == 0
    mlib.saveMesh(mesh, tmp_path / f"{case_name}.ctm")  # used to store
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mlib.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh)
