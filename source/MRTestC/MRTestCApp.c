#include "MRBitSet.h"
#include "MRMesh.h"
#include "MRMeshBoolean.h"
#include "MRMeshCollidePrecise.h"
#include "MRMeshDecimate.h"
#include "MRMeshFillHole.h"
#include "MRMeshNormals.h"
#include "MRMeshComponents.h"
#include "MRMeshBuilder.h"
#include "MRFixSelfIntersections.h"
#include "MRExpandShrink.h"
#include "MRPointCloud.h"

int main( void )
{
    testArea();
    testBitSet();
    testMeshBoolean();
    testBooleanMultipleEdgePropogationSort();
    testBooleanMapper();
    testMeshCollidePrecise();
    testMeshDecimate();
    testMeshFillHole();
    testMeshFillHoleNicely();
    testMeshNormals();
    testComponentsMap();
    testLargeRegions();
    testUniteCloseVertices();
    testLargeComponents();
    testLargestComponent();
    testGetComponent();
    testFixSelfIntersections();
    testRightBoundary();
    testFindHoleComplicatingFaces();
    testExpandShrink();
    testExpandShrinkVerts();
    testShortEdges();
    testIncidentFacesFromVerts();
    testIncidentFacesFromEdges();
    testTriangulation();
}
