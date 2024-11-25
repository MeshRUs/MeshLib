#include "TestMacros.h"

#include "MRMesh.h"
#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRCube.h"
#include "MRMeshC/MRTorus.h"
#include "MRMeshC/MRBitSet.h"
#include "MRMeshC/MRMeshFixer.h"
#include "MRMeshC/MRRegionBoundary.h"

void testArea( void )
{
    const MRVector3f size = mrVector3fDiagonal( 1.f );
    const MRVector3f base = mrVector3fDiagonal( -0.5f );

    MRMesh* mesh = mrMakeCube( &size, &base );
    

    double area = mrMeshArea( mesh, NULL );
    TEST_ASSERT( area > 5.999f && area < 6.001f );

    MRFaceBitSet* faces = mrFaceBitSetNew( 12, true );
    for ( int i = 6; i < 12; ++i )
        mrBitSetSet( (MRBitSet*)faces, i, false );

    area = mrMeshArea( mesh, faces );
    TEST_ASSERT( area > 2.999f && area < 3.001f );

    mrMeshDeleteFaces( mesh, faces, NULL );

    area = mrMeshArea( mesh, NULL );
    TEST_ASSERT( area > 2.999f && area < 3.001f );

    mrFaceBitSetFree( faces );
    mrMeshFree( mesh );    
}

void testShortEdges( void )
{
    MRMakeTorusParameters params;
    params.primaryRadius = 1.f;
    params.secondaryRadius = 0.05f;
    params.primaryResolution = 16;
    params.secondaryResolution = 16;

    MRMesh* mesh = mrMakeTorus( &params );
    MRMeshPart mp;
    mp.mesh = mesh;
    mp.region = NULL;

    MRUndirectedEdgeBitSet* shortEdges = mrFindShortEdges( &mp, 0.1f, NULL, NULL );
    size_t num = mrBitSetCount( (MRBitSet*)shortEdges );
    TEST_ASSERT( num == 256 );
    mrUndirectedEdgeBitSetFree( shortEdges );
    mrMeshFree( mesh );
}

void testIncidentFacesFromVerts( void )
{
    const MRVector3f size = mrVector3fDiagonal( 1.f );
    const MRVector3f base = mrVector3fDiagonal( -0.5f );

    MRMesh* mesh = mrMakeCube( &size, &base );

    MRVertBitSet* verts = mrVertBitSetNew( 8, false );
    mrBitSetSet( (MRBitSet*)verts, 0, true );

    MRFaceBitSet* faces = mrGetIncidentFacesFromVerts( mrMeshTopology(mesh), verts );
    size_t num = mrBitSetCount( (MRBitSet*)faces );
    TEST_ASSERT( num == 6 );
    mrFaceBitSetFree( faces );
    mrVertBitSetFree( verts );
    mrMeshFree( mesh );
}

void testIncidentFacesFromEdges( void )
{
    const MRVector3f size = mrVector3fDiagonal( 1.f );
    const MRVector3f base = mrVector3fDiagonal( -0.5f );

    MRMesh* mesh = mrMakeCube( &size, &base );

    MRUndirectedEdgeBitSet* edges = mrUndirectedEdgeBitSetNew( 12, false );
    mrBitSetSet( ( MRBitSet* ) edges, 0, true );

    MRFaceBitSet* faces = mrGetIncidentFacesFromEdges( mrMeshTopology( mesh ), edges );
    size_t num = mrBitSetCount( ( MRBitSet* ) faces );
    TEST_ASSERT( num == 8 );
    mrFaceBitSetFree( faces );
    mrUndirectedEdgeBitSetFree( edges );
    mrMeshFree( mesh );
}