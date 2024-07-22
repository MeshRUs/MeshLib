from constants import DEFAULT_RHAUSDORF_THRESHOLD
from helpers.file_helpers import get_reference_files_list
from module_helper import *
from pytest_check import check
import meshlib.mrmeshpy as mrmesh
from pathlib import Path


def relative_hausdorff(mesh1: mrmesh.Mesh or mrmesh.PointCloud or Path or str,
                       mesh2: mrmesh.Mesh or mrmesh.PointCloud or Path or str):
    """
    Calculate Hausdorff distance between two meshes, normalized on smallest bounding box diagonal.
    1.0 means that the meshes are equal, 0.0 means that they are completely different.
    The value is in range [0.0, 1.0]

    :param mesh1: first mesh or path to it
    :param mesh2: second mesh or path to it
    """
    if isinstance(mesh1, str) or isinstance(mesh1, Path):
        mesh1 = mrmesh.loadMesh(str(mesh1))
    if isinstance(mesh2, str) or isinstance(mesh2, Path):
        mesh2 = mrmesh.loadMesh(str(mesh2))
    distance = mrmesh.findMaxDistanceSq(mesh1, mesh2) ** 0.5
    diagonal = min(mesh1.getBoundingBox().diagonal(), mesh2.getBoundingBox().diagonal())
    val = 1.0 - (distance / diagonal)
    val = 0.0 if val < 0.0 else val  # there are some specific cases when metric can be below zero,
    # but exact values have no practical meaning, any value beyond zero means "completely different"
    return val


def compare_meshes_similarity(mesh1: mrmesh.Mesh, mesh2: mrmesh.Mesh,
                              rhsdr_thresh=DEFAULT_RHAUSDORF_THRESHOLD,
                              vol_thresh=0.005,
                              area_thresh=0.005,
                              verts_thresh=0.005,
                              skip_volume=False):
    """
    Compare two meshes and assert them to similarity by different params.
    Similarity calcs as (mesh1.param - mesh2.param) / min(mesh1.param, mesh2.param) < param_threshold
    If one of difference is more than threshold - fail on assert
    :param mesh1: first mesh
    :param mesh2: second mesh
    :param rhsdr_thresh: relative Hausdorff distance threshold
    :param vol_thresh: volume difference threshold
    :param area_thresh: area difference threshold
    :param verts_thresh: vertices difference threshold
    :param skip_volume: skip volume check - for open meshes
    """
    with check:
        #  check on meshes relative Hausdorff distance
        rhsdr = relative_hausdorff(mesh1, mesh2)
        assert rhsdr > rhsdr_thresh, f"Relative hausdorff is lower than threshold: {rhsdr}, threshold is {rhsdr_thresh}"
    if not skip_volume:
        with check:
            #  check on meshes volume
            assert abs(mesh1.volume() - mesh2.volume()) / min(mesh1.volume(), mesh2.volume()) < vol_thresh, (
                f"Volumes of result and reference differ too much, \nvol1={mesh1.volume()}\nvol2={mesh2.volume()}\n"
                f"relative threshold is {vol_thresh}")
    with check:
        #  check on meshes area
        assert abs(mesh1.area() - mesh2.area()) / min(mesh1.area(), mesh2.area()) < area_thresh, (
            f"Areas of result and reference differ too much, \narea1={mesh1.area()}\narea2={mesh2.area()}\n"
            f"relative threshold is {area_thresh}")
        #  check on meshes vertices number
    with check:
        if mesh1.topology.numValidVerts() - mesh2.topology.numValidVerts() != 0:
            assert abs(mesh1.topology.numValidVerts() - mesh2.topology.numValidVerts()) / min(mesh1.topology.numValidVerts(),
                                                                        mesh2.topology.numValidVerts()) < verts_thresh, (
                f"Vertex numbers of result and reference differ too much, \n"
                f"verts1={mesh1.topology.numValidVerts()}\nverts2={mesh2.topology.numValidVerts()}\n"
                f"relative threshold is {verts_thresh}")


def compare_mesh(mesh1: mrmesh.Mesh or Path or str, ref_file_path: Path, multi_ref=True):
    """
    Compare mesh by full equality with multiple reference files by content
    :param mesh1: mesh to compare
    :param ref_file_path: reference file
    :param multi_ref: if True, it compares file with multiple references, otherwise with single reference
    """
    if isinstance(mesh1, str) or isinstance(mesh1, Path):
        mesh1 = mrmesh.loadMesh(mesh1)
    if multi_ref:
        ref_files = get_reference_files_list(ref_file_path)
    else:
        ref_files = [ref_file_path]
    for ref_file in ref_files:
        if mesh1 == mrmesh.loadMesh(ref_file):
            return True
    return False


def compare_points_similarity(points_a: mrmesh.PointCloud or Path or str,
                              points_b: mrmesh.PointCloud or Path or str,
                              rhsdr_thresh=DEFAULT_RHAUSDORF_THRESHOLD,
                              verts_thresh=0.005,
                              testname: str = None):
    """
    Compare two meshes and assert them to similarity by different params.
    Similarity calcs as (mesh1.param - mesh2.param) / min(mesh1.param, mesh2.param) < param_threshold
    If one of difference is more than threshold - fail on assert
    :param points_a: first point cloud
    :param points_b: second point cloud
    :param rhsdr_thresh: relative Hausdorff distance threshold
    :param verts_thresh: vertices difference threshold
    :param testname: name of test, will be printed on fail
    """
    test_report = f"Testname is {testname}\n" if testname else ""

    # load points if required
    if isinstance(points_a, str) or isinstance(points_a, Path):
        points_a = mrmesh.loadPoints(str(points_a))
    if isinstance(points_b, str) or isinstance(points_b, Path):
        points_b = mrmesh.loadPoints(str(points_b))

    num_p_a = points_a.validPoints.size()
    num_p_b = points_b.validPoints.size()

    # checks for empty clouds
    if num_p_a == 0 and num_p_b == 0:
        return
    if num_p_a == 0 or num_p_b == 0:
        assert False, f"{test_report}One of point clouds is empty, while other is not"

    # check on meshes relative Hausdorff distance
    with check:
        rhsdr = relative_hausdorff(points_a, points_b)
        assert rhsdr > rhsdr_thresh, f"{test_report}Relative hausdorff is lower than threshold: {rhsdr}, threshold is {rhsdr_thresh}"
    # check points number
    with check:
        if num_p_a - num_p_b != 0:
            assert abs(num_p_a - num_p_b) / min(num_p_a, num_p_b) < verts_thresh, (
                f"{test_report}Vertex numbers of result and reference differ too much, \n"
                f"verts1={num_p_a}\nverts2={num_p_b}\n"
                f"relative threshold is {verts_thresh}")
