from module_helper import *
from pathlib import Path
from pytest_check import check
from constants import test_files_path
from helpers.meshlib_helpers import compare_meshes_similarity, compare_mesh

import pytest


@pytest.mark.smoke
@pytest.mark.parametrize("input_case", ["torus", "crocodile"])
def test_make_bridge_edge(input_case, tmp_path):
    """
    Test makeBridgeEdge algorithm
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "bridge" / input_case
    case_name = f"bridge_edge_{input_case}"
    mesh = mrmeshpy.loadMesh(input_folder / "input.ctm")

    # make bridge edge
    topl = mesh.topology
    edges_num_before = topl.edgeSize()
    faces_num_before = topl.numValidFaces()

    edges = topl.findHoleRepresentiveEdges()
    mrmeshpy.makeBridgeEdge(topl, edges[0], edges[1])

    # === Verification
    with check:
        assert topl.edgeSize() - edges_num_before == 2, "Edges number should be increased on 2"
    with check:
        assert topl.numValidFaces() == faces_num_before, "Faces number should be the same"
    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.mrmesh")
    ref_mesh_path = input_folder / f"{case_name}.mrmesh"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh, edges_thresh=0)


@pytest.mark.smoke
@pytest.mark.parametrize("input_case", ["torus", "crocodile"])
def test_make_bridge(input_case, tmp_path):
    """
    Test makeBridge algorithm
    """
    #  Load input meshes
    input_folder = Path(test_files_path) / "algorithms" / "bridge" / input_case
    case_name = f"bridge_{input_case}"
    mesh = mrmeshpy.loadMesh(input_folder / "input.ctm")

    # make bridge edge
    topl = mesh.topology
    edges = topl.findHoleRepresentiveEdges()
    edges_num_before = topl.edgeSize()
    faces_num_before = topl.numValidFaces()
    mrmeshpy.makeBridge(topl, edges[0], edges[1])

    # === Verification
    with check:
        assert mesh.topology.edgeSize() - edges_num_before == 6, "Edges number should be increased on 6"
    with check:
        assert topl.numValidFaces() - faces_num_before == 2, "Faces number should be increased on 2"

    mrmeshpy.saveMesh(mesh, tmp_path / f"{case_name}.ctm")
    ref_mesh_path = input_folder / f"{case_name}.ctm"
    ref_mesh = mrmeshpy.loadMesh(ref_mesh_path)

    with check:
        compare_meshes_similarity(mesh, ref_mesh, edges_thresh=0)
