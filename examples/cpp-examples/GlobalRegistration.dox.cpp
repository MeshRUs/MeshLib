#include <MRMesh/MRBox.h>
#include <MRMesh/MRMultiwayICP.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointsSave.h>

#include <iostream>

void printStats( const MR::MultiwayICP& icp )
{
    std::cout << "Samples: " << icp.getNumSamples() << std::endl
              << "Active point pairs: " << icp.getNumActivePairs() << std::endl;
    if ( icp.getNumActivePairs() > 0 )
    {
        const auto p2ptMetric = icp.getMeanSqDistToPoint();
        const auto p2plMetric = icp.getMeanSqDistToPlane();
        std::cout << "RMS point-to-point distance: " << p2ptMetric << " ± " << icp.getMeanSqDistToPoint( p2ptMetric ) << std::endl
                  << "RMS point-to-plane distance: " << p2plMetric << " ± " << icp.getMeanSqDistToPlane( p2plMetric ) << std::endl;
    }
}

int main( int argc, char* argv[] )
{
    if ( argc < 4 )
    {
        std::cerr << "Usage: " << argv[0] << " INPUT1 INPUT2 [INPUTS...] OUTPUT" << std::endl;
        return EXIT_FAILURE;
    }

    // the global registration can be applied to meshes and point clouds
    // to simplify the sample app, we will work with point clouds only
    std::vector<MR::PointCloud> inputs;
    // as ICP and MultiwayICP classes accept both meshes and point clouds,
    // the input data must be converted to special wrapper objects
    // NB: the wrapper objects hold *references* to the source data, NOT their copies
    MR::ICPObjects objects;
    MR::Box3f maxBBox;
    for ( auto i = 1; i < argc - 1; ++i )
    {
        auto pointCloud = MR::PointsLoad::fromAnySupportedFormat( argv[i] );
        if ( !pointCloud )
        {
            std::cerr << "Failed to load point cloud: " << pointCloud.error() << std::endl;
            return EXIT_FAILURE;
        }

        auto bbox = pointCloud->computeBoundingBox();
        if ( !maxBBox.valid() || bbox.volume() > maxBBox.volume() )
            maxBBox = bbox;

        inputs.emplace_back( std::move( *pointCloud ) );
        // you may also set an affine transformation for each input as a second argument
        objects.push_back( { inputs.back(), {} } );
    }

    // you can set various parameters for the global registration; see the documentation for more info
    MR::MultiwayICP icp( objects, {
        // set sampling voxel size
        .samplingVoxelSize = maxBBox.diagonal() * 0.03f,
    } );

    icp.setParams( {} );

    // gather statistics
    icp.updateAllPointPairs();
    printStats( icp );

    std::cout << "Calculating transformations..." << std::endl;
    auto xfs = icp.calculateTransformations();
    printStats( icp );

    MR::PointCloud output;
    for ( auto i = MR::ObjId( 0 ); i < inputs.size(); ++i )
    {
        const auto& input = inputs[i];
        const auto& xf = xfs[i];
        for ( const auto& point : input.points )
            output.points.emplace_back( xf( point ) );
    }

    auto res = MR::PointsSave::toAnySupportedFormat( output, argv[argc - 1] );
    if ( !res )
    {
        std::cerr << "Failed to save point cloud: " << res.error() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}