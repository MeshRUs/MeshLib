import meshlib.mrmeshpy as mrmeshpy
import sys

# Load mesh
try:
    mesh = mrmeshpy.loadMesh("mesh.stl")
except ValueError as e:
    print(e)
    sys.exit(1)

# Save mesh
try:
    mrmeshpy.saveMesh(mesh, "mesh.ply")
except ValueError as e:
    print(e)
    sys.exit(1)

