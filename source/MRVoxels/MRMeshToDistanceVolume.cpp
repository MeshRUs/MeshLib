#include "MRMeshToDistanceVolume.h"
#include "MRMesh/MRIsNaN.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRFastWindingNumber.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include <tuple>

namespace MR
{

std::optional<float> signedDistanceToMesh( const MeshPart& mp, const Vector3f& p, const DistanceToMeshOptions& op )
{
    assert( op.signMode != SignDetectionMode::OpenVDB );
    const auto proj = findProjection( p, mp, op.maxDistSq, nullptr, op.minDistSq );
    if ( op.signMode != SignDetectionMode::HoleWindingRule // for HoleWindingRule the sign can change even for too small or too large distances
        && ( proj.distSq <= op.minDistSq || proj.distSq >= op.maxDistSq ) )
        return {}; // distance is too small or too large, discard them

    float dist = std::sqrt( proj.distSq );
    switch ( op.signMode )
    {
    case SignDetectionMode::ProjectionNormal:
        if ( !mp.mesh.isOutsideByProjNorm( p, proj, mp.region ) )
            dist = -dist;
        break;

    case SignDetectionMode::WindingRule:
    {
        const Line3d ray( Vector3d( p ), Vector3d::plusX() );
        int count = 0;
        rayMeshIntersectAll( mp, ray, [&count] ( auto&& ) { ++count; return true; } );
        if ( count % 2 == 1 ) // inside
            dist = -dist;
        break;
    }

    case SignDetectionMode::HoleWindingRule:
        assert( !mp.region );
        if ( !mp.mesh.isOutside( p ) )
            dist = -dist;
        break;

    default: ; //nothing
    }
    return dist;
}

Expected<SimpleVolume> meshToDistanceVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& cParams /*= {} */ )
{
    MR_TIMER
    auto params = cParams;
    assert( params.dist.signMode != SignDetectionMode::OpenVDB );

    if ( params.dist.signMode == SignDetectionMode::HoleWindingRule )
    {
        SimpleVolume res;
        res.voxelSize = params.vol.voxelSize;
        res.dims = params.vol.dimensions;
        VolumeIndexer indexer( res.dims );
        res.data.resize( indexer.size() );

        if ( !params.fwn )
            params.fwn = std::make_shared<FastWindingNumber>( mp.mesh );
        assert( !mp.region ); // only whole mesh is supported for now
        auto basis = AffineXf3f( Matrix3f::scale( params.vol.voxelSize ), params.vol.origin + 0.5f * params.vol.voxelSize );
        constexpr float beta = 2;
        if ( auto d = params.fwn->calcFromGridWithDistances( res.data, res.dims, basis, beta,
            params.dist.maxDistSq, params.dist.minDistSq, params.vol.cb ); !d )
        {
            return unexpected( std::move( d.error() ) );
        }
        std::tie( res.min, res.max ) = parallelMinMax( res.data );
        return res;
    }

    const auto func = meshToDistanceFunctionVolume( mp, params );
    return functionVolumeToSimpleVolume( func, params.vol.cb );

}

FunctionVolume meshToDistanceFunctionVolume( const MeshPart& mp, const MeshToDistanceVolumeParams& params )
{
    assert( params.dist.signMode != SignDetectionMode::OpenVDB );

    return FunctionVolume
    {
        .data = [params, mp = MeshPart( mp.mesh )] ( const Vector3i& pos ) -> float
        {
            const auto coord = Vector3f( pos ) + Vector3f::diagonal( 0.5f );
            const auto voxelCenter = params.vol.origin + mult( params.vol.voxelSize, coord );
            auto dist = signedDistanceToMesh( mp, voxelCenter, params.dist );
            return dist ? *dist : cQuietNan;
        },
        .dims = params.vol.dimensions,
        .voxelSize = params.vol.voxelSize
    };
}

Expected<SimpleVolume> meshRegionToIndicatorVolume( const Mesh& mesh, const FaceBitSet& region,
    float offset, const DistanceVolumeParams& params )
{
    MR_TIMER
    if ( !region.any() )
    {
        assert( false );
        return unexpected( "empty region" );
    }

    SimpleVolume res;
    res.voxelSize = params.voxelSize;
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );

    AABBTree regionTree( { mesh, &region } );
    const FaceBitSet notRegion = mesh.topology.getValidFaces() - region;
    //TODO: check that notRegion is not empty
    AABBTree notRegionTree( { mesh, &notRegion } );
    
    const auto voxelSize = std::max( { params.voxelSize.x, params.voxelSize.y, params.voxelSize.z } );

    if ( !ParallelFor( size_t( 0 ), indexer.size(), [&]( size_t i )
    {
        const auto coord = Vector3f( indexer.toPos( VoxelId( i ) ) ) + Vector3f::diagonal( 0.5f );
        auto voxelCenter = params.origin + mult( params.voxelSize, coord );

        // minimum of given offset distance parameter and the distance to not-region part of mesh
        const auto distToNotRegion = std::sqrt( findProjectionSubtree( voxelCenter, mesh, notRegionTree, sqr( offset ) ).distSq );

        const auto maxDistSq = sqr( distToNotRegion + voxelSize );
        const auto minDistSq = sqr( std::max( distToNotRegion - voxelSize, 0.0f ) );
        const auto distToRegion = std::sqrt( findProjectionSubtree( voxelCenter, mesh, regionTree, maxDistSq, nullptr, minDistSq ).distSq );

        res.data[i] = distToRegion - distToNotRegion;
    }, params.cb ) )
        return unexpectedOperationCanceled();

    std::tie( res.min, res.max ) = parallelMinMax( res.data );

    return res;
}



Expected<std::array<SimpleVolume, 3>> meshToDirectionVolume( const MeshToDirectionVolumeParams& params )
{
    MR_TIMER
    VolumeIndexer indexer( params.vol.dimensions );
    std::vector<MeshProjectionResult> projs;

    auto getPoint = [&indexer, &params] ( VoxelId i )
    {
        const auto c = Vector3f( indexer.toPos( i ) ) + Vector3f::diagonal( 0.5f );
        return params.vol.origin + mult( params.vol.voxelSize, c );
    };

    {
        std::vector<Vector3f> points( indexer.size() );
        for ( auto i = VoxelId( size_t( 0 ) ); i < indexer.size(); ++i )
        {
            points[i] = getPoint( i );
        }
        params.projector->findProjections( projs, points );
    }

    std::array<SimpleVolume, 3> res;
    for ( auto& v : res )
    {
        v.voxelSize = params.vol.voxelSize;
        v.dims = params.vol.dimensions;
        v.data.resize( indexer.size() );
    }

    for ( auto i = VoxelId( size_t( 0 ) ); i < indexer.size(); ++i )
    {
        const auto d = ( getPoint( i ) - projs[i].proj.point ).normalized();
        res[0].data[i] = d.x;
        res[1].data[i] = d.y;
        res[2].data[i] = d.z;
    }

    for ( auto& v : res )
    {
        std::tie( v.min, v.max ) = parallelMinMax( v.data );
    }

    return res;
}


} //namespace MR