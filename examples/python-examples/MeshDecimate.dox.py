import meshlib.mrmeshpy as mrmeshpy

# Load mesh
mesh = mrmeshpy.loadMesh("mesh.stl")

# Repack mesh optimally.
# It's not necessary but highly recommended to achieve the best performance in parallel processing
mesh.packOptimally()

# Setup decimate parameters
settings = mrmeshpy.DecimateSettings()

# Decimation stop thresholds, you may specify one or both
settings.maxDeletedFaces = 1000 # Number of faces to be deleted
settings.maxError = 0.05 # Maximum error when decimation stops

# Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
# Recommended to set to number of CPU cores or more available for the best performance
settings.subdivideParts = 64

# Decimate mesh
mrmeshpy.decimateMesh(mesh, settings)

# Save result
mrmeshpy.saveMesh(mesh, "decimatedMesh.stl")
