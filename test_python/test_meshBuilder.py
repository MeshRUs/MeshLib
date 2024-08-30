import pytest
from helper import *

def test_unite_close_vertices():
	positions = [
	mrmesh.Vector3f(0,0,0),
	mrmesh.Vector3f(0,1,0),
	mrmesh.Vector3f(1,1,0),
	mrmesh.Vector3f(1,0,0),
	mrmesh.Vector3f(0,0,1),
	mrmesh.Vector3f(0,1,1),
	mrmesh.Vector3f(1,1,1),
	mrmesh.Vector3f(1,0,1)
	]
	mesh = mrmesh.Mesh()
	mesh.points.vec.resize(36)
	mesh.points.vec[0*3+0] = positions[0]
	mesh.points.vec[0*3+1] = positions[1]
	mesh.points.vec[0*3+2] = positions[2]

	mesh.points.vec[1*3+0] = positions[2]
	mesh.points.vec[1*3+1] = positions[3]
	mesh.points.vec[1*3+2] = positions[0]

	mesh.points.vec[2*3+0] = positions[0]
	mesh.points.vec[2*3+1] = positions[4]
	mesh.points.vec[2*3+2] = positions[5]

	mesh.points.vec[3*3+0] = positions[5]
	mesh.points.vec[3*3+1] = positions[1]
	mesh.points.vec[3*3+2] = positions[0]

	mesh.points.vec[4*3+0] = positions[0]
	mesh.points.vec[4*3+1] = positions[3]
	mesh.points.vec[4*3+2] = positions[7]

	mesh.points.vec[5*3+0] = positions[7]
	mesh.points.vec[5*3+1] = positions[4]
	mesh.points.vec[5*3+2] = positions[0]

	mesh.points.vec[6*3+0] = positions[6]
	mesh.points.vec[6*3+1] = positions[5]
	mesh.points.vec[6*3+2] = positions[4]

	mesh.points.vec[7*3+0] = positions[4]
	mesh.points.vec[7*3+1] = positions[7]
	mesh.points.vec[7*3+2] = positions[6]

	mesh.points.vec[8*3+0] = positions[1]
	mesh.points.vec[8*3+1] = positions[5]
	mesh.points.vec[8*3+2] = positions[6]

	mesh.points.vec[9*3+0] = positions[6]
	mesh.points.vec[9*3+1] = positions[2]
	mesh.points.vec[9*3+2] = positions[1]

	mesh.points.vec[10*3+0] = positions[6]
	mesh.points.vec[10*3+1] = positions[7]
	mesh.points.vec[10*3+2] = positions[3]

	mesh.points.vec[11*3+0] = positions[3]
	mesh.points.vec[11*3+1] = positions[2]
	mesh.points.vec[11*3+2] = positions[6]

	tris = mrmesh.Triangulation()
	tris.vec.resize(12)
	for i in range(12):
		tris.vec[i] = mrmesh.ThreeVertIds([mrmesh.VertId(i*3 + 0),mrmesh.VertId(i*3 + 1),mrmesh.VertId(i*3 + 2)])
	mesh.topology = mrmesh.topologyFromTriangles(tris)

	assert mrmesh.getAllComponents(mesh).size() == 12
	assert mesh.topology.findHoleRepresentiveEdges().size() == 12

	mrmesh.uniteCloseVertices(mesh,0)
	assert mrmesh.getAllComponents(mesh).size() == 1
	assert mesh.topology.findHoleRepresentiveEdges().size() == 0
