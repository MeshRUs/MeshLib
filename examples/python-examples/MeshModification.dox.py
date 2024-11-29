import math

import meshlib.mrmeshpy as mrmeshpy

mesh = mrmeshpy.makeTorus()

relax_params = mrmeshpy.MeshRelaxParams()
relax_params.iterations = 5
mrmeshpy.relax(mesh, relax_params)

props = mrmeshpy.SubdivideSettings()
props.maxDeviationAfterFlip = 0.5
mrmeshpy.subdivideMesh(mesh, props)

plus_z = mrmeshpy.Vector3f()
plus_z.z = 1.0
rotation_xf = mrmeshpy.AffineXf3f.linear(mrmeshpy.Matrix3f.rotation(plus_z, math.pi * 0.5))
mesh.transform(rotation_xf)
