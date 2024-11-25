import meshlib.mrmeshpy as mrmeshpy
import sys

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# Setup parameters
params = mrmeshpy.OffsetParameters()
params.voxelSize = mesh.computeBoundingBox().diagonal() * 5e-3 # offset grid precision (algorithm is voxel based)
if mrmeshpy.findRightBoundary(mesh.topology).empty():
    params.signDetectionMode = mrmeshpy.SignDetectionMode.HoleWindingRule  # use if you have holes in mesh

# Make offset mesh
offset = mesh.computeBoundingBox().diagonal() * 0.05
try:
    result_mesh = mrmeshpy.offsetMesh(mesh, offset, params)
except ValueError as e:
    print(e)
    sys.exit(1)


# Save result
mrmeshpy.saveMesh(result_mesh, "offsetMesh.stl")
