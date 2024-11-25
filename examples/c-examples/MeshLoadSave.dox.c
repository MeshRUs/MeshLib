#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshLoad.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRMeshTopology.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    // Save mesh
    mrMeshSaveToAnySupportedFormat( mesh, "mesh.ply", &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save mesh: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_mesh;
    }

    rc = EXIT_SUCCESS;
out_mesh:
    mrMeshFree( mesh );
out:
    return rc;
}
