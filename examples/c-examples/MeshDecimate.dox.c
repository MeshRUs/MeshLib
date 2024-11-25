#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshDecimate.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;
    
    // error messages will be stored here
    MRString* errorString = NULL;

    // Load mesh
    MRMesh* mesh = mrMeshLoadFromAnySupportedFormat( "mesh.stl", &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to load mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    // Setup decimate parameters
    MRDecimateSettings params = mrDecimateSettingsNew();

    // Decimation stop thresholds, you may specify one or both
    params.maxDeletedFaces = 1000; // Number of faces to be deleted
    params.maxError = 0.05f; // Maximum error when decimation stops

    // Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
    // Recommended to set to the number of available CPU cores or more for the best performance
    params.subdivideParts = 64;

    // Decimate mesh
    MRDecimateResult result = mrDecimateMesh( mesh, &params );
    printf( "Removed %d vertices, %d faces", result.vertsDeleted, result.facesDeleted );

    // Save result
    mrMeshSaveToAnySupportedFormat( mesh, "decimated_mesh.stl", &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    rc = EXIT_SUCCESS;
out:
    mrMeshFree( mesh );
    return rc;
}
