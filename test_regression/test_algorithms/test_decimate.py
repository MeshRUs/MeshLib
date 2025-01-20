from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("dec_params", [
    pytest.param({"name": "maxError_0.05",
                  "params": {
                      "maxError": 0.05
                  }},
                 id="maxError_0.05"),
    pytest.param({"name": "maxError_0.25",
                  "hasAbiIssuesOnUbuntuArm2004Mrbind": True,
                  "params": {
                      "maxError": 0.2
                  }},
                 id="maxError_0.25"),
    pytest.param({"name": "target_triangles_200",
                  "params": {
                      "maxDeletedFaces": 200,  # tuned to R0003C_V4-16aug19 mesh
                      "maxError": 0.05
                  }}, id="target_triangles_200"),
    pytest.param({"name": "maxEdgeLen_245",
                  "params": {
                      "maxEdgeLen": 245,
                      "maxError": 0.03  # tuned to R0003C_V4-16aug19 mesh
                  }}, id="maxEdgeLen_245"),
    pytest.param({"name": "maxEdgeLen_0.4",
                  "params": {
                      "maxEdgeLen": 0.4,  # tuned to R0003C_V4-16aug19 mesh
                      "maxError": 0.05
                  }}, id="maxEdgeLen_0.4"),
    pytest.param({"name": "maxTriangleAspectRatio_5",
                  "hasAbiIssuesOnUbuntuArm2004Mrbind": True,
                  "params": {
                      "maxTriangleAspectRatio": 5,
                      "maxError": 0.15
                  }}, id="maxTriangleAspectRatio_5"),
    pytest.param({"name": "stabilizer_0.001",
                  "params": {
                      "stabilizer": 0.001,
                      "maxError": 0.03
                  }}, id="stabilizer_0.001"),
    pytest.param({"name": "stabilizer_0",
                  "params": {
                      "stabilizer": 0,
                      "maxError": 0.03
                  }}, id="stabilizer_0"),
    pytest.param({"name": "strategy_ShortestEdgeFirst",
                  "params": {
                      "strategy": "ShortestEdgeFirst",
                      "maxError": 0.05
                  }}, id="strategy_ShortestEdgeFirst"),
    pytest.param({"name": "strategy_MinimizeError",
                  "params": {
                      "strategy": "MinimizeError",
                      "maxError": 0.05
                  }}, id="strategy_MinimizeError"),
    pytest.param({"name": "touchBdVertices_true",
                  "params": {
                      "touchNearBdEdges": True,
                      "maxError": 0.05
                  }}, id="touchBdVertices_true"),
    pytest.param({"name": "touchBdVertices_false",
                  "params": {
                      "touchNearBdEdges": False,
                      "maxError": 0.05
                  }}, id="touchBdVertices_false"),
    pytest.param({"name": "optimizeVertexPos_false",
                  "params": {
                      "optimizeVertexPos": False,
                      "maxError": 0.05

                  }}, id="optimizeVertexPos_false"),
    pytest.param({"name": "optimizeVertexPos_true",
                  "params": {
                      "optimizeVertexPos": True,
                      "maxError": 0.05
                  }}, id="optimizeVertexPos_true"),
])
def test_decimate(tmp_path, dec_params):
    """
    Test decimate algorithm with all settings, avaliable in UI
    """

    # Those tests fail (seemingly due to an unknown ABI incompatibility) on Arm Ubuntu 20.04 when the MRBind bindings
    # are compiled with Clang 18 AND MeshLib is compiled with Clang 12 or older (14 is fine on Ubuntu 22.04, 13 wasn't tested).
    # So instead we build the whole MeshLib with Clang 18 (only on ubuntu 20.04 arm) when building the wheels,
    # but when building ML we simply disable the offending tests (makes no sense to use Clang 18 for that,
    # since the library users will then face this ABI incompatibility).
    if os.getenv("MR_REGRESSION_TESTS_UBUNTUARM2004_MRBIND_ABI_ISSUES","0") == "1" and dec_params.get("hasAbiIssuesOnUbuntuArm2004Mrbind", False):
        print('Skipping this configuration on Ubuntu Arm 20.04')
        return

    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "decimate" / "R0003C_V4-16aug19"
    case_name = dec_params["name"]
    mesh = mrmeshpy.loadMesh(input_folder / "input.mrmesh")

    # Setup decimate parameters
    settings = mrmeshpy.DecimateSettings()
    settings.maxError = 0.001
    for key in dec_params["params"].keys():
        if key == "strategy":
            settings.strategy = mrmeshpy.DecimateStrategy.__members__[dec_params["params"]["strategy"]]
        else:
            settings.__setattr__(key, dec_params["params"][key])
    settings.packMesh = True
    mrmeshpy.decimateMesh(mesh, settings)

    # === Verification
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)
    #  check meshes similarity (for extra details on fail)
    with check:
        compare_meshes_similarity(mesh, ref_mesh,
                                  verts_thresh=0.01, edges_thresh=0.01)  # diff vs reference usually about 5 verts on 800 overall
    with check:
        self_col_tri = mrmeshpy.findSelfCollidingTriangles(mesh).size()
        assert self_col_tri == 0, f"Mesh should have no self-colliding triangles, actual value is {self_col_tri}"
    with check:
        degen_faces = mrmeshpy.findDegenerateFaces(mesh).count()
        assert degen_faces == 0, f"Mesh should have no degenerate faces, actual value is {degen_faces}"
