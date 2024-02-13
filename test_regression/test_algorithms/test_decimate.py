from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh
import meshlib.mrmeshpy as mlpy

import pytest


@pytest.mark.parametrize("dec_params", [
    {"name": "maxError_0.05",
        "params": {
            "maxError": 0.05
        }},
    {"name": "maxError_0.25",
        "params": {
            "maxError": 0.25
        }},
    {"name": "target_triangles_200",
        "params": {
            "maxDeletedFaces": 200,  # tuned to R0003C_V4-16aug19 mesh
            "maxError": 0.05
        }},
    {"name": "maxEdgeLen_245",
        "params": {
            "maxEdgeLen": 245,
            "maxError": 0.25  # tuned to R0003C_V4-16aug19 mesh
        }},
    {"name": "maxEdgeLen_0.4",
        "params": {
            "maxEdgeLen": 0.4,  # tuned to R0003C_V4-16aug19 mesh
            "maxError": 0.20
        }},
    {"name": "maxTriangleAspectRatio_20",
     "params": {
         "maxTriangleAspectRatio": 20,
         "maxError": 0.25
     }},
    {"name": "maxTriangleAspectRatio_1",
     "params": {
         "maxTriangleAspectRatio": 1,
         "maxError": 0.25
     }},
    {"name": "stabilizer_0.001",
        "params": {
            "stabilizer": 0.001,
            "maxError": 0.05
        }},
    {"name": "stabilizer_0",
        "params": {
            "stabilizer": 0,
            "maxError": 0.05

        }},
    {"name": "strategy_ShortestEdgeFirst",
     "params": {
         "strategy": "ShortestEdgeFirst",
         "maxError": 0.25
     }},
    {"name": "strategy_MinimizeError",
     "params": {
         "strategy": "MinimizeError",
         "maxError": 0.25
     }},
    {"name": "touchBdVertices_true",
     "params": {
         "touchBdVertices": True,
         "maxError": 0.25
     }},
    {"name": "touchBdVertices_false",
     "params": {
         "touchBdVertices": False,
         "maxError": 0.25
     }},
    {"name": "optimizeVertexPos_false",
     "params": {
         "optimizeVertexPos": False,
         "maxError": 0.25

     }},
    {"name": "optimizeVertexPos_true",
     "params": {
         "optimizeVertexPos": True,
         "maxError": 0.25
     }},
])
def test_decimate(tmp_path, dec_params):
    """
    Test boolean algorithm with all operation types
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "decimate" / "R0003C_V4-16aug19"
    case_name = dec_params["name"]
    mesh = mlpy.loadMesh(input_folder / "input.mrmesh")

    # Setup decimate parameters
    settings = mlpy.DecimateSettings()
    for key in dec_params["params"].keys():
        if key == "strategy":
            settings.strategy = mlpy.DecimateStrategy.__members__[dec_params["params"]["strategy"]]
        else:
            settings.__setattr__(key, dec_params["params"][key])
    settings.packMesh = True
    mlpy.decimateMesh(mesh, settings)

    # === Verification
    mlpy.saveMesh(mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mlpy.loadMesh(ref_mesh_path)
    #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(mesh, ref_mesh,
                                  verts_thresh=0.01)  # diff vs reference usually about 5 verts on 800 overall
    with check:
        self_col_tri = mlpy.findSelfCollidingTriangles(mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
