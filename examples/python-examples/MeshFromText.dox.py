import sys

import meshlib.mrmeshpy as mrmeshpy

if len(sys.argv) < 3:
    print("Usage: ./MeshFromText fontpath text", file=sys.stderr)
    sys.exit(1)

font_path = sys.argv[1]
text = sys.argv[2]

params = mrmeshpy.SymbolMeshParams()
params.text = text
params.pathToFontFile = font_path
try:
    conv_res = mrmeshpy.createSymbolsMesh(params)
except ValueError as e:
    print(f"Failed to convert text to mesh: {e}", file=sys.stderr)
    sys.exit(1)

mrmeshpy.saveMesh(conv_res, "mesh.ply")
