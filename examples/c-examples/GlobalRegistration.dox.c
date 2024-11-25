#include <MRMeshC/MRBox.h>
#include <MRMeshC/MRMultiwayICP.h>
#include <MRMeshC/MRPointCloud.h>
#include <MRMeshC/MRPointsLoad.h>
#include <MRMeshC/MRPointsSave.h>
#include <MRMeshC/MRString.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// print progress every 10%
int gProgress = -1;
bool onProgress( float v )
{
    int progress = (int)( 10.f * v );
    if ( progress != gProgress )
    {
        gProgress = progress;
        printf( "%d%%...\n", progress * 10 );
    }
    return true;
}

void resetProgress( void )
{
    gProgress = -1;
}

void printStats( const MRMultiwayICP* icp )
{
    size_t numSamples = mrMultiWayICPGetNumSamples( icp );
    size_t numActivePairs = mrMultiWayICPGetNumActivePairs( icp );
    printf( "Samples: %zu\n", numSamples );
    printf( "Active point pairs: %zu\n", numActivePairs );
    if ( numActivePairs > 0 )
    {
        double p2ptMetric = mrMultiWayICPGetMeanSqDistToPoint( icp, NULL );
        double p2ptInaccuracy = mrMultiWayICPGetMeanSqDistToPoint( icp, &p2ptMetric );
        printf( "RMS point-to-point distance: %f ± %f\n", p2ptMetric, p2ptInaccuracy );

        double p2plMetric = mrMultiWayICPGetMeanSqDistToPlane( icp, NULL );
        double p2plInaccuracy = mrMultiWayICPGetMeanSqDistToPlane( icp, &p2plMetric );
        printf( "RMS point-to-plane distance: %f ± %f\n", p2plMetric, p2plInaccuracy );
    }
}

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;
    if ( argc < 4 )
    {
        fprintf( stderr, "Usage: %s INPUT1 INPUT2 [INPUTS...] OUTPUT\n", argv[0] );
        goto out;
    }

    // error messages will be stored here
    MRString* errorString = NULL;

    // the global registration can be applied to meshes and point clouds
    // to simplify the sample app, we will work with point clouds only
    const int inputNum = argc - 2;
    MRPointCloud** inputs = malloc( sizeof( MRPointCloud* ) * inputNum );
    memset( inputs, 0, sizeof( MRPointCloud* ) * inputNum );
    // as ICP and MultiwayICP classes accept both meshes and point clouds,
    // the input data must be converted to special wrapper objects
    // NB: the wrapper objects hold *references* to the source data, NOT their copies
    MRMeshOrPointsXf** inputXfs = malloc( sizeof( MRMeshOrPointsXf* ) * inputNum );
    memset( inputXfs, 0, sizeof( MRMeshOrPointsXf* ) * inputNum );
    MRBox3f maxBBox = mrBox3fNew();
    for ( int i = 0; i < inputNum; ++i )
    {
        inputs[i] = mrPointsLoadFromAnySupportedFormat( argv[1 + i], &errorString );
        if ( errorString )
        {
            fprintf( stderr, "Failed to load point cloud: %s\n", mrStringData( errorString ) );
            mrStringFree( errorString );
            goto out_inputs;
        }

        // you may also set an affine transformation for each input as a second argument
        inputXfs[i] = mrMeshOrPointsXfFromPointCloud( inputs[i], NULL ); // or mrMeshOrPointsXfFromMesh for meshes

        MRBox3f bbox = mrPointCloudComputeBoundingBox( inputs[i], NULL );
        if ( !mrBox3fValid( &maxBBox ) || mrBox3fVolume( &bbox ) > mrBox3fVolume( &maxBBox ) )
            maxBBox = bbox;
    }

    // you can set various parameters for the global registration; see the documentation for more info
    MRMultiwayICPSamplingParameters samplingParams = mrMultiwayIcpSamplingParametersNew();
    // set sampling voxel size
    samplingParams.samplingVoxelSize = mrBox3fDiagonal( &maxBBox ) * 0.03f;
    // set progress callback
    samplingParams.cb = onProgress;

    MRMultiwayICP* icp = mrMultiwayICPNew( inputXfs, inputNum, &samplingParams );

    MRICPProperties params = mrICPPropertiesNew();
    mrMultiwayICPSetParams( icp, &params );

    // gather statistics
    mrMultiwayICPUpdateAllPointPairs( icp, NULL );
    printStats( icp );

    printf( "Calculating transformations...\n" );
    resetProgress();
    MRVectorAffineXf3f* xfs = mrMultiwayICPCalculateTransformations( icp, onProgress );
    printStats( icp );

    MRPointCloud* output = mrPointCloudNew();
    for ( int i = 0; i < inputNum; i++ )
    {
        const MRAffineXf3f* xf = xfs->data + i;
        for ( int j = 0; j < mrPointCloudPointsNum( inputs[i] ); j++ )
        {
            MRVector3f* point = mrPointCloudPointsRef( inputs[i] ) + j;
            mrAffineXf3fApply( xf, point );
            mrPointCloudAddPoint( output, point );
        }
    }

    mrPointsSaveToAnySupportedFormat( output, argv[argc - 1], &errorString );
    if ( errorString )
    {
        fprintf( stderr, "Failed to save point cloud: %s\n", mrStringData( errorString ) );
        mrStringFree( errorString );
        goto out_output;
    }

    rc = EXIT_SUCCESS;
out_output:
    mrPointCloudFree( output );
out_xfs:
    mrVectorAffineXf3fFree( xfs );
out_icp:
    mrMultiwayICPFree( icp );
out_inputs:
    for ( int i = 0; i < inputNum; i++ )
    {
        mrMeshOrPointsXfFree( inputXfs[i] );
        mrPointCloudFree( inputs[i] );
    }
    free( inputXfs );
    free( inputs );
out:
    return rc;
}
