#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRVector3.h>
#include <MRMeshC/MRBox.h>
#include <MRMeshC/MRAffineXf.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MROffset.h>
#include <MRMeshC/MRString.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define APPROX_VOXEL_COUNT 10000000.f

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;

    // error messages will be stored here
    MRString* errorString = NULL;

    // Create mesh
    MRVector3f size = mrVector3fDiagonal( 1.f );
    MRVector3f base = mrVector3fDiagonal( -0.5f );
    MRMesh* mesh = mrMakeCube( &size, &base );

    // offset functions can also be applied to separate mesh components rather than to the whole mesh
    // this is not our case, so the region is set to NULL
    MRMeshPart inputMeshPart = (MRMeshPart){
        .mesh = mesh,
        .region = NULL,
    };

    // Setup parameters
    MROffsetParameters params = mrOffsetParametersNew();
    // calculate voxel size depending on desired accuracy and/or memory consumption
    params.voxelSize = mrSuggestVoxelSize( inputMeshPart, 10000000.f );
    MRAffineXf3f xf = mrAffineXf3fNew();
    MRBox3f bbox = mrMeshComputeBoundingBox( mesh, &xf );
    float offset = mrBox3fDiagonal( &bbox ) * 0.1f;

    // Make offset mesh
    MRMesh* outputMesh = mrOffsetMesh( inputMeshPart, offset, &params, &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to perform offset: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    // Save result
    mrMeshSaveToAnySupportedFormat( outputMesh, "mesh_offset.stl", &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save inputMesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_outputMesh;
    }

    rc = EXIT_SUCCESS;
out_outputMesh:
    mrMeshFree( outputMesh );
out:
    mrMeshFree( mesh );
    return rc;
}
