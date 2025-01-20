import sys

import meshlib.mrmeshpy as mrmeshpy

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# you can set various parameters for the resolving process; see the documentation for more info
params = mrmeshpy.ResolveMeshDegenSettings()
params.maxDeviation = 1e-5 * mesh.computeBoundingBox().diagonal()
params.tinyEdgeLength = 1e-3

changed = mrmeshpy.resolveMeshDegenerations(mesh, params)
if not changed:
    print("No changes were made")
    sys.exit(0)

mrmeshpy.saveMesh(mesh, "fixed_mesh.stl")
