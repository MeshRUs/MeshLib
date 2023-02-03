#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRVDBConversions.h"
#include "MRTimer.h"
#include "MRPolyline.h"
#include "MRMeshFillHole.h"
#include "MRRegionBoundary.h"
#include "MRPch/MRSpdlog.h"
#include "MRVoxelsConversions.h"
#include "MRSharpenMarchingCubesMesh.h"

namespace
{
constexpr float autoVoxelNumber = 5e6f;
}

namespace MR
{

tl::expected<Mesh, std::string> offsetMesh( const MeshPart & mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER

    float voxelSize = params.voxelSize;
    // Compute voxel size if needed
    if ( voxelSize <= 0.0f )
    {
        auto bb = mp.mesh.computeBoundingBox( mp.region );
        auto vol = bb.volume();
        voxelSize = std::cbrt( vol / autoVoxelNumber );
    }

    bool useShell = params.type == OffsetParameters::Type::Shell;
    if ( !findRegionBoundary( mp.mesh.topology, mp.region ).empty() && !useShell )
    {
        spdlog::warn( "Cannot use offset for non-closed meshes, using shell instead." );
        useShell = true;
    }

    if ( useShell )
        offset = std::abs( offset );

    auto offsetInVoxels = offset / voxelSize;

    auto voxelSizeVector = Vector3f::diagonal( voxelSize );
    // Make grid
    auto grid = ( !useShell ) ?
        // Make level set grid if it is closed
        meshToLevelSet( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
                        params.callBack ?
                        [params]( float p )
    {
        return params.callBack( p * 0.5f );
    } : ProgressCallback{} ) :
        // Make distance field grid if it is not closed
        meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
                        params.callBack ?
                             [params]( float p )
    {
        return params.callBack( p * 0.5f );
    } : ProgressCallback{} );

    if ( !grid )
        return tl::make_unexpected( "Operation was canceled." );

    // Make offset mesh
    auto newMesh = gridToMesh( std::move( grid ), voxelSizeVector, offsetInVoxels, params.adaptivity, params.callBack ?
                             [params]( float p )
    {
        return params.callBack( 0.5f + p * 0.5f );
    } : ProgressCallback{} );

    if ( !newMesh.has_value() )
        return tl::make_unexpected( "Operation was canceled." );

    // For not closed meshes orientation is flipped on back conversion
    if ( useShell )
        newMesh->topology.flipOrientation();

    return newMesh;
}

tl::expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    if ( !findRegionBoundary( mp.mesh.topology, mp.region ).empty() )
    {
        spdlog::error( "Only closed meshes allowed for double offset." );
        return tl::make_unexpected( "Only closed meshes allowed for double offset." );
    }
    if ( params.type == OffsetParameters::Type::Shell )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return levelSetDoubleConvertion( mp, AffineXf3f(), params.voxelSize, offsetA, offsetB, params.adaptivity, params.callBack );
}

tl::expected<MR::Mesh, std::string> sharpOffsetMesh( const Mesh& mesh, float offset, const SharpOffsetParameters& params )
{
    MR_TIMER;

    auto offsetInVoxels = offset / params.voxelSize;

    ProgressCallback meshToLSCb;
    if ( params.callBack )
        meshToLSCb = [&] ( float p )
    {
        return params.callBack( p * 0.3f );
    };

    auto voxelRes = meshToLevelSet( mesh, AffineXf3f(),
        Vector3f::diagonal( params.voxelSize ),
        std::abs( offsetInVoxels ) + 2, meshToLSCb );

    if ( !voxelRes )
        return tl::make_unexpected( "Operation was canceled." );

    VdbVolume volume = floatGridToVdbVolume( voxelRes );


    VdbVolumeToMeshParams vmParams;
    vmParams.basis.A = Matrix3f::scale( params.voxelSize );
    vmParams.iso = offsetInVoxels;
    vmParams.lessInside = true;
    if ( params.callBack )
        vmParams.cb = [&] ( float p )
    {
        return params.callBack( 0.3f + 0.4f * p );
    };
    Vector<VoxelId, FaceId> map;
    vmParams.outVoxelPerFaceMap = &map;
    auto meshRes = vdbVolumeToMesh( volume, vmParams );
    if ( !meshRes )
        return tl::make_unexpected( "Operation was canceled." );

    SharpenMarchingCubesMeshSettings sharpenParams;
    sharpenParams.minNewVertDev = params.voxelSize / 25;
    sharpenParams.maxNewVertDev = 2 * params.voxelSize;
    sharpenParams.maxOldVertPosCorrection = params.voxelSize / 2;
    sharpenParams.offset = offset;
    sharpenParams.outSharpEdges = params.outSharpEdges;

    sharpenMarchingCubesMesh( mesh, *meshRes, map, sharpenParams );
    if ( params.callBack && !params.callBack( 0.99f ) )
        return tl::make_unexpected( "Operation was canceled." );

    return *meshRes;
}

tl::expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER;

    Mesh mesh;
    auto contours = polyline.topology.convertToContours<Vector3f>(
        [&points = polyline.points]( VertId v )
    {
        return points[v];
    } );

    std::vector<EdgeId> newHoles;
    newHoles.reserve( contours.size() );
    for ( auto& cont : contours )
    {
        if ( cont[0] != cont.back() )
            cont.insert( cont.end(), cont.rbegin(), cont.rend() );
        newHoles.push_back( mesh.addSeparateEdgeLoop( cont ) );
    }

    for ( auto h : newHoles )
        makeDegenerateBandAroundHole( mesh, h );

    return offsetMesh( mesh, offset, params );
}

}
#endif
