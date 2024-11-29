import pytest
from module_helper import *

from constants import test_files_path
from pathlib import Path
from helpers.meshlib_helpers import compare_mesh
from pytest_check import check

import shutil
import runpy

SAMPLES_FOLDER = Path("../examples/python-examples")


def run_code_sample(code_path: str, args: list):
    """
    Function to run code sample with CLI args
    :param code_path: path to code sample
    :param args: arguments to pass to code sample
    """
    # Preserving original args
    original_sys_argv = sys.argv.copy()
    # setting new args for code to execute
    sys.argv = [code_path] + args
    try:
        runpy.run_path(code_path)
    finally:
        # restoring original args
        sys.argv = original_sys_argv

@pytest.mark.parametrize("doc_case",
                         [
                             pytest.param({'sample': "FreeFormDeformation.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['deformed_mesh.stl']}, id="FreeFormDeformation.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "LaplacianExample.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['deformed_mesh.stl']}, id="LaplacianExample.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "MeshBoolean.dox.py", 'input_meshes': [],
                                           'output_meshes': ['out_boolean.stl']}, id="MeshBoolean.dox.py"),
                             pytest.param({'sample': "MeshDecimate.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['decimatedMesh.stl']}, id="MeshDecimate.dox.py"),
                             pytest.param({'sample': "MeshExtrude.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['extrudedMesh.stl']}, id="MeshExtrude.dox.py"),
                             pytest.param({'sample': "MeshFillHole.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['filledMesh.stl']}, id="MeshFillHole.dox.py"),
                             pytest.param({'sample': "MeshICP.dox.py", 'input_meshes': ['meshA.stl', 'meshB.stl'],
                                           'output_meshes': ['meshA_icp.stl']}, id="MeshICP.dox.py"),
                             pytest.param({'sample': "MeshLoadSave.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['mesh.ply']}, id="MeshLoadSave.dox.py"),
                             pytest.param({'sample': "MeshModification.dox.py", 'input_meshes': [],
                                           'output_meshes': []}, id="MeshModification.dox.py"),
                             pytest.param({'sample': "MeshOffset.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['offsetMesh.stl']}, id="MeshOffset.dox.py"),
                             pytest.param({'sample': "MeshStitchHole.dox.py",
                                           'input_meshes': ['meshAwithHole.stl', 'meshBwithHole.stl'],
                                           'output_meshes': ['stitchedMesh.stl']}, id="MeshStitchHole.dox.py"),
                             pytest.param({'sample': "NoiseDenoiseExample.dox.py", 'input_meshes': ['mesh.stl'],
                                           'output_meshes': ['noised_mesh.stl', 'denoised_mesh.stl']},
                                          id="NoiseDenoiseExample.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "MeshResolveDegenerations.dox.py", 'input_meshes': ['mesh.ctm'],
                                           'output_meshes': ['mesh_fixed.ctm']},
                                          id="MeshResolveDegenerations.dox.py",
                                          marks=pytest.mark.bindingsV3),
                             pytest.param({'sample': "Numpy.dox.py", 'input_meshes': [], 'output_meshes': []},
                                          id="Numpy.dox.py"),
                             pytest.param(
                                 {'sample': "NumpyTriangulation.dox.py", 'input_meshes': [], 'output_meshes': []},
                                 id="NumpyTriangulation.dox.py"),
                             pytest.param(
                                 {'sample': "Triangulation_v2.dox.py", 'input_meshes': [], 'output_meshes': []},
                                 id="Triangulation_v2.dox.py",
                                 marks=pytest.mark.bindingsV2),
                             pytest.param(
                                 {'sample': "Triangulation_v3.dox.py", 'input_meshes': [], 'output_meshes': []},
                                 id="Triangulation_v3.dox.py",
                                 marks=pytest.mark.bindingsV3),
                             pytest.param(
                                 {'sample': "GlobalRegistration.dox.py",
                                  'input_meshes': ["cloud0.ply", "cloud1.ply", "cloud2.ply"],
                                  'output_meshes': ["out.ply"],
                                  'args': ["cloud0.ply", "cloud1.ply", "cloud2.ply", "out.ply"]},
                                 id="GlobalRegistration.dox.py",
                                 marks=pytest.mark.bindingsV3),
                         ])
@pytest.mark.smoke
def test_python_doc_samples(tmp_path, doc_case):
    """
    Test copies python files from examples to tmp_path, and executes them.
    :param tmp_path: pytest fixture for temporary directory
    :return: None
    """
    # Copy files from examples to tmp_path
    py_file = doc_case['sample']
    shutil.copy(SAMPLES_FOLDER / py_file, tmp_path / py_file)
    for mesh in doc_case['input_meshes']:
        shutil.copy(Path(test_files_path) / "doc_samples" / py_file / mesh, mesh)

    # Execute file in tmp_path
    if 'args' in doc_case:
        run_code_sample(tmp_path / py_file, doc_case['args'])
    else:
        run_code_sample(tmp_path / py_file, [])

    for out_mesh in doc_case['output_meshes']:
        shutil.copy(out_mesh, tmp_path / out_mesh)

    # Compare files in tmp_path with reference files
    for out_mesh in doc_case['output_meshes']:
        with check:
            compare_mesh(out_mesh, Path(test_files_path) / 'doc_samples' / py_file / out_mesh)
