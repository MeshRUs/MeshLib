import meshlib.mrmeshpy as mrmeshpy

mesh = mrmeshpy.loadMesh("mesh.ctm")

# You can set various parameters for the resolving process; see the documentation for more info
params = mrmeshpy.ResolveMeshDegenSettings()
# maximum permitted deviation
params.maxDeviation = 1e-5 * mesh.computeBoundingBox().diagonal()
# maximum length of edges to be collapsed
params.tinyEdgeLength = 1e-3

is_changed = mrmeshpy.resolveMeshDegenerations(mesh, params)

if is_changed:
    print("Mesh was changed, saving")
    mrmeshpy.saveMesh(mesh, "mesh_fixed.ctm")
else:
    print("Mesh was not changed")
