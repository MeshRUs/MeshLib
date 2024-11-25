#include "TestMacros.h"

#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRMeshTopology.h"
#include "MRMeshC/MRBitSet.h"
#include "MRMeshC/MRPointCloud.h"
#include "MRMeshC/MRPointCloudTriangulation.h"
#include "MRMeshC/MRVector3.h"
#include "MRMeshC/MRTorus.h"
#include "MRMeshC/MRMeshToPointCloud.h"

void testTriangulation( void )
{
    MRMakeTorusParameters torusParams;
    torusParams.primaryRadius = 2.f;
    torusParams.secondaryRadius = 1.0f;
    torusParams.primaryResolution = 32;
    torusParams.secondaryResolution = 32;

    MRMesh* mesh = mrMakeTorus( &torusParams );

    MRPointCloud* pc = mrMeshToPointCloud( mesh, true, NULL );
    const MRTriangulationParameters params = mrTriangulationParametersNew();
    MRMesh* restored = mrTriangulatePointCloud( pc, &params );
    TEST_ASSERT( mrMeshPointsNum( restored ) == 1024 );
    const MRMeshTopology* top = mrMeshTopology( restored );
    TEST_ASSERT( mrBitSetCount( ( MRBitSet* )mrMeshTopologyGetValidVerts( top ) ) == 1024 );
    MREdgePath* holes = mrMeshTopologyFindHoleRepresentiveEdges( top );
    
    TEST_ASSERT( holes->size == 0 );

    mrEdgePathFree( holes );
    mrMeshFree( mesh );
    mrMeshFree( restored );
    mrPointCloudFree( pc );
}
