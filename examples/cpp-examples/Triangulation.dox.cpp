#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointCloudTriangulation.h>
#include <MRMesh/MRUniformSampling.h>
#include <MRVoxels/MROffset.h>

int main()
{
    // Generate point cloud
    MR::PointCloud pc;
    pc.points.reserve( 10000 );
    for ( auto i = 0; i < 100; ++i )
    {
        const auto u = MR::PI2_F * float( i ) / ( 100.f - 1.f );
        for ( auto j = 0; j < 100; ++j )
        {
            const auto v = MR::PI_F * float( j ) / ( 100.f - 1.f );

            pc.points.emplace_back(
                std::cos( u ) * std::sin( v ),
                std::sin( u ) * std::sin( v ),
                std::cos( v )
            );
        }
    }
    // Remove duplicate points
    auto vs = MR::pointUniformSampling( pc, {
        .distance = 1e-3f,
    } );
    assert( vs );
    pc.validPoints = std::move( *vs );
    pc.invalidateCaches();

    // Triangulate it
    auto triangulated = MR::triangulatePointCloud( pc );
    assert( triangulated );

    // Fix possible issues
    auto mesh = MR::offsetMesh( *triangulated, 0.f, { {
        .voxelSize = MR::suggestVoxelSize( *triangulated, 5e+6f ),
    } } );
}
