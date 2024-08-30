import os
import sys

import pytest

working_directory = str()
# insert 0 to find mrpy.so in working directory and not in system
if (
    "MeshLibPyModulesPath" in os.environ
    and not os.environ["MeshLibPyModulesPath"] in sys.path
):
    sys.path.insert(0, os.environ["MeshLibPyModulesPath"])
    working_directory = os.environ["MeshLibPyModulesPath"]

if bool(os.environ.get('USE_MESHLIB2_PY', 0)):
    from meshlib2 import mrmeshpy as mrmesh
    from meshlib2 import mrmeshnumpy
    mrmesh.getAllComponentsVerts = mrmesh.MeshComponents_getAllComponentsVerts
    mrmesh.BooleanResMapObj = mrmesh.BooleanResultMapper_MapObject
    mrmesh.vectorConstMeshPtr = mrmesh.std_vector_const_Mesh
    mrmesh.vectorEdges = mrmesh.EdgeLoop
    mrmesh.copyMesh = mrmesh.Mesh
    mrmesh.localFindSelfIntersections = mrmesh.SelfIntersections_getFaces
    mrmesh.localFixSelfIntersections = mrmesh.SelfIntersections_fix
    mrmesh.FixSelfIntersectionSettings = mrmesh.SelfIntersections_Settings
    mrmesh.FixSelfIntersectionMethod = mrmesh.SelfIntersections_Settings_Method
    mrmesh.triangulateContours = mrmesh.PlanarTriangulation_triangulateContours
    mrmesh.TextAlignParams = mrmesh.TextMeshAlignParams
    mrmesh.MeshToVolumeParamsType = mrmesh.MeshToVolumeParams_Type
    mrmesh.VoxelsSaveSavingSettings = mrmesh.VoxelsSave_SavingSettings
    mrmesh.saveMesh = mrmesh.MeshSave_toAnySupportedFormat # There was also a manually written overload, hmm.
    mrmesh.loadMesh = mrmesh.MeshLoad_fromAnySupportedFormat
    mrmesh.saveSliceToImage = mrmesh.VoxelsSave_saveSliceToImage
    mrmesh.saveAllSlicesToImage = mrmesh.VoxelsSave_saveAllSlicesToImage
    mrmesh.LaplacianEdgeWeightsParam = mrmesh.EdgeWeights
    mrmesh.getAllComponents = mrmesh.MeshComponents_getAllComponents
    mrmesh.MeshBuilderSettings = mrmesh.MeshBuilder_BuildSettings
    mrmesh.findUndercuts = mrmesh.FixUndercuts_findUndercuts
    mrmesh.fixUndercuts = mrmesh.FixUndercuts_fixUndercuts
    mrmesh.topologyFromTriangles = mrmesh.MeshBuilder_fromTriangles
    mrmesh.uniteCloseVertices = mrmesh.MeshBuilder_uniteCloseVertices
else:
    import meshlib.mrmeshpy as mrmesh
    import meshlib.mrmeshnumpy as mrmeshnumpy
