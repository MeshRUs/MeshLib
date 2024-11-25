#include "TestMacros.h"
#include "MRExpandShrink.h"

#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRBitSet.h"
#include "MRMeshC/MRExpandShrink.h"
#include "MRMeshC/MRMakeSphereMesh.h"

void testExpandShrink( void )
{
    MRSphereParams params = mrSphereParamsNew();
    params.numMeshVertices = 3000;
    params.radius = 1.0f;

    MRMesh* mesh = mrMakeSphere( &params );
    const MRMeshTopology* top = mrMeshTopology( mesh );

    MRFaceId face; face.id = 0;
    MRFaceBitSet* region = mrExpandFaceRegionFromFace( top, face, 3 );

    size_t num = mrBitSetCount( (MRBitSet*)region );
    TEST_ASSERT( num ==75 );

    mrExpandFaceRegion( top, region, 3 );
    num = mrBitSetCount( ( MRBitSet* )region );
    TEST_ASSERT( num > 75 ); //platform dependent results

    mrShrinkFaceRegion( top, region, 3 );
    num = mrBitSetCount( ( MRBitSet* )region );
    TEST_ASSERT( num == 75 );

    mrFaceBitSetFree( region );
    mrMeshFree( mesh );
}

void testExpandShrinkVerts( void )
{
    MRSphereParams params = mrSphereParamsNew();
    params.numMeshVertices = 3000;
    params.radius = 1.0f;

    MRMesh* mesh = mrMakeSphere( &params );
    const MRMeshTopology* top = mrMeshTopology( mesh );

    MRVertId vert; vert.id = 0;
    MRVertBitSet* region = mrExpandVertRegionFromVert( top, vert, 3 );

    size_t num = mrBitSetCount( (MRBitSet*)region );
    TEST_ASSERT( num == 37 );

    mrExpandVertRegion( top, region, 3 );
    num = mrBitSetCount( ( MRBitSet* )region );
    TEST_ASSERT( num > 37 ); //platform dependent results

    mrShrinkVertRegion( top, region, 3 );
    num = mrBitSetCount( ( MRBitSet* )region );
    TEST_ASSERT( num == 37 );

    mrVertBitSetFree( region );
    mrMeshFree( mesh );
}