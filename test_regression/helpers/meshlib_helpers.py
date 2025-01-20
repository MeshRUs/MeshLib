from constants import DEFAULT_RHAUSDORF_THRESHOLD
from helpers.file_helpers import get_reference_files_list
from module_helper import *
from pytest_check import check
from pathlib import Path



def relative_hausdorff(mesh1: mrmeshpy.Mesh or mrmeshpy.PointCloud or Path or str,
                       mesh2: mrmeshpy.Mesh or mrmeshpy.PointCloud or Path or str):
    """
    Calculate Hausdorff distance between two meshes, normalized on smallest bounding box diagonal.
    1.0 means that the meshes are equal, 0.0 means that they are completely different.
    The value is in range [0.0, 1.0]

    :param mesh1: first mesh or path to it
    :param mesh2: second mesh or path to it
    """
    if isinstance(mesh1, str) or isinstance(mesh1, Path):
        mesh1 = mrmeshpy.loadMesh(str(mesh1))
    if isinstance(mesh2, str) or isinstance(mesh2, Path):
        mesh2 = mrmeshpy.loadMesh(str(mesh2))
    distance = mrmeshpy.findMaxDistanceSq(mesh1, mesh2) ** 0.5
    diagonal = min(mesh1.getBoundingBox().diagonal(), mesh2.getBoundingBox().diagonal())
    val = 1.0 - (distance / diagonal)
    val = 0.0 if val < 0.0 else val  # there are some specific cases when metric can be below zero,
    # but exact values have no practical meaning, any value beyond zero means "completely different"
    return val


def compare_meshes_similarity(mesh1: mrmeshpy.Mesh, mesh2: mrmeshpy.Mesh,
                              rhsdr_thresh=DEFAULT_RHAUSDORF_THRESHOLD,
                              vol_thresh=0.005,
                              area_thresh=0.005,
                              verts_thresh=0.005,
                              edges_thresh=0.005,
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
    if isinstance(mesh1, str) or isinstance(mesh1, Path):
        mesh1 = mrmeshpy.loadMesh(str(mesh1))
    if isinstance(mesh2, str) or isinstance(mesh2, Path):
        mesh2 = mrmeshpy.loadMesh(str(mesh2))

    with check:
        #  check on meshes relative Hausdorff distance
        rhsdr = relative_hausdorff(mesh1, mesh2)
        assert rhsdr > rhsdr_thresh, f"Relative hausdorff is lower than threshold: {rhsdr}, threshold is {rhsdr_thresh}"
    if not skip_volume:
        with check:
            #  check on meshes volume
            vol_1 = mesh1.volume()
            vol_2 = mesh2.volume()
            assert abs(vol_1 - vol_2) / min(vol_1, vol_2) < vol_thresh, (
                f"Volumes of result and reference differ too much, \nvol1={vol_1}\nvol2={vol_2}\n"
                f"relative threshold is {vol_thresh}")
    with check:
        #  check on meshes area
        area_1 = mesh1.area()
        area_2 = mesh2.area()
        assert abs(area_1 - area_2) / min(area_1, area_2) < area_thresh, (
            f"Areas of result and reference differ too much, \narea1={area_1}\narea2={area_2}\n"
            f"relative threshold is {area_thresh}")
        #  check on meshes vertices number
    with check:
        verts_1 = mesh1.topology.numValidVerts()
        verts_2 = mesh2.topology.numValidVerts()
        if verts_1 != verts_2:
            assert abs(verts_1 - mesh2.topology.numValidVerts()) / min(verts_1,
                                                                        verts_2) < verts_thresh, (
                f"Vertex numbers of result and reference differ too much, \n"
                f"verts1={verts_1}\nverts2={verts_2}\n"
                f"relative threshold is {verts_thresh}")
    with check:
        edges_1 = mesh1.topology.computeNotLoneUndirectedEdges()
        edges_2 = mesh2.topology.computeNotLoneUndirectedEdges()
        if edges_1 != edges_2:
            assert abs(edges_1 - edges_2) / min(edges_1, edges_2) < edges_thresh, (
                f"Edge numbers of result and reference differ too much, \n"
                f"edges1={edges_1}\nedges2={edges_2}\n"
                f"relative threshold is {edges_thresh}")


def compare_mesh(mesh1: mrmeshpy.Mesh or Path or str, ref_file_path: Path, multi_ref=True):
    """
    Compare mesh by full equality with multiple reference files by content
    :param mesh1: mesh to compare
    :param ref_file_path: reference file
    :param multi_ref: if True, it compares file with multiple references, otherwise with single reference
    """
    if isinstance(mesh1, str) or isinstance(mesh1, Path):
        mesh1 = mrmeshpy.loadMesh(mesh1)
    if multi_ref:
        ref_files = get_reference_files_list(ref_file_path)
    else:
        ref_files = [ref_file_path]
    for ref_file in ref_files:
        if mesh1 == mrmeshpy.loadMesh(ref_file):
            return True
    return False


def compare_points_similarity(points_a: mrmeshpy.PointCloud or Path or str,
                              points_b: mrmeshpy.PointCloud or Path or str,
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
        points_a = mrmeshpy.loadPoints(str(points_a))
    if isinstance(points_b, str) or isinstance(points_b, Path):
        points_b = mrmeshpy.loadPoints(str(points_b))

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


def compare_voxels(voxels_a: mrmeshpy.VdbVolume or Path or str,
                              voxels_b: mrmeshpy.VdbVolume or Path or str,
                              testname: str = None,
                              ):
    """
    Checks that two voxels have same properties and that means they can be treated as identical
    :param voxels_a: first voxel
    :param voxels_b: second voxel
    :param testname: name of test, will be printed on fail
    """
    test_report = f"Testname is {testname}\n" if testname else ""
    # load voxels if required
    if isinstance(voxels_a, str) or isinstance(voxels_a, Path):
        voxels_a = mrmeshpy.loadVoxelsRaw(Path(voxels_a))
    if isinstance(voxels_b, str) or isinstance(voxels_b, Path):
        voxels_b = mrmeshpy.loadVoxelsRaw(Path(voxels_b))
    with check:
        for dim in ["x", "y", "z"]:
            val_a = voxels_a.voxelSize.__getattribute__(dim)
            val_b = voxels_b.voxelSize.__getattribute__(dim)
            # dcm format sometimes has very small difference in voxel sizes, so we need to check it with threshold
            assert (val_a - val_b) / val_a < 0.00001, (
                    f"{test_report}Voxel sizes are differs for dimension {dim}, \n"
                    f"voxel_a:{val_a}\nvoxel_b:{val_b}\n")
        assert voxels_a.min == voxels_b.min, (
            f"{test_report}voxels.min of voxels are differs, \n"
            f"voxel_a:{voxels_a.min}\nvoxel_b:{voxels_b.min}\n")
        assert voxels_a.max == voxels_b.max, (
            f"{test_report}voxels.min of voxels are differs, \n"
            f"voxel_a:{voxels_a.max}\nvoxel_b:{voxels_b.max}\n")
        assert voxels_a.dims == voxels_b.dims, (
            f"{test_report}voxels.dims of voxels are differs, \n"
            f"voxel_a:{voxels_a.dims}\nvoxel_b:{voxels_b.dims}\n")


def compare_lines(lines_a: mrmeshpy.Polyline3 or Path or str,
                  lines_b: mrmeshpy.Polyline3 or Path or str,
                  testname: str = None,
                  ):
    """
    Checks that two lines have same properties and that means they can be treated as identical
    :param lines_a: first set of lines
    :param lines_b: second set of lines
    :param testname: name of test, will be printed on fail
    """
    test_report = f"Testname is {testname}\n" if testname else ""
    # load lines from file if required
    if isinstance(lines_a, str) or isinstance(lines_a, Path):
        lines_a = mrmeshpy.loadLines(Path(lines_a))
    if isinstance(lines_b, str) or isinstance(lines_b, Path):
        lines_b = mrmeshpy.loadLines(Path(lines_b))
    with check:
        assert lines_a.getBoundingBox().diagonal() == lines_b.getBoundingBox().diagonal(), (
                f"{test_report}Diagonals of bounding boxes are differs, \n"
                f"lines_a:{lines_a.voxelSize}\nlines_b:{lines_b.voxelSize}\n")
        assert lines_a.totalLength() == lines_b.totalLength(), (
            f"{test_report}total length of lines are differs, \n"
            f"lines_a:{lines_a.min}\nlines_b:{lines_b.min}\n")


def compare_distance_maps(dm_a, dm_b, threshold=0.001):
    assert dm_a.resX() == dm_b.resX() and dm_a.resY() == dm_b.resY(), \
        f"resolution is different {dm_a.resX(), dm_a.resY()} vs {dm_a.resX(), dm_a.resY()}"
    with check:
        for i in range(dm_a.resX()):
            for j in range(dm_a.resY()):
                a = dm_a.get(i, j)
                b = dm_b.get(i, j)
                if a != b:
                    assert abs(a - b) / max(a, b) < threshold, (
                        f"values are different at {i}, {j}, {a}, {b}")
