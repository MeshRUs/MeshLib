#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMakeSphereMesh.h>
#include <MRMeshC/MRVector3.h>
#include <MRMeshC/MRAffineXf.h>
#include <MRMeshC/MRMeshBoolean.h>
#include <MRMeshC/MRMeshSave.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;

    // error messages will be stored here
    MRString* errorString = NULL;

    // create first sphere with radius of 1 unit
    MRMakeUVSphereParameters makeParams = mrMakeUvSphereParametersNew();
    makeParams.radius = 1.f;
    makeParams.horizontalResolution = 64;
    makeParams.verticalResolution = 64;
    MRMesh* sphere1 = mrMakeUVSphere( &makeParams );

    // create second sphere by cloning the first sphere and moving it in X direction
    MRMesh* sphere2 = mrMeshCopy( sphere1 );
    MRVector3f xfTranslation = mrVector3fDiagonal( 0.f );
    xfTranslation.x = 0.7f;
    MRAffineXf3f xf = mrAffineXf3fTranslation( &xfTranslation );
    mrMeshTransform( &sphere2, &xf, NULL );

    // perform the boolean operation
    MRBooleanParameters params = mrBooleanParametersNew();
    MRBooleanResult result = mrBoolean( sphere1, sphere2, MRBooleanOperationIntersection, &params );
    if ( result.errorString )
    {
        fprintf( stderr, "Failed to perform boolean: %s", mrStringData( result.errorString ) );
        mrStringFree( errorString );
        goto out;
    }

    // save result to STL file
    mrMeshSaveToAnySupportedFormat( result.mesh, "out_boolean.stl", &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save result: %s", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_result;
    }

    rc = EXIT_SUCCESS;
out_result:
    mrMeshFree( result.mesh );
out:
    mrMeshFree( sphere2 );
    mrMeshFree( sphere1 );
    return rc;
}
