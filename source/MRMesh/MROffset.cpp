#include "MROffset.h"
#include "MRMesh.h"
#include "MRBox.h"
#include "MRFloatGrid.h"
#include "MRVDBConversions.h"
#include "MRTimer.h"
#include "MRPolyline.h"
#include "MRMeshFillHole.h"
#include "MRRegionBoundary.h"
#include "MRVoxelsConversions.h"
#include "MRSharpenMarchingCubesMesh.h"
#include "MRFastWindingNumber.h"
#include "MRVolumeIndexer.h"
#include "MRInnerShell.h"
#include "MRMeshFixer.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRPch/MRSpdlog.h"

namespace MR
{

float suggestVoxelSize( const MeshPart & mp, float approxNumVoxels )
{
    MR_TIMER
    auto bb = mp.mesh.computeBoundingBox( mp.region );
    auto vol = bb.volume();
    return std::cbrt( vol / approxNumVoxels );
}

#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
Expected<Mesh, std::string> offsetMesh( const MeshPart & mp, float offset, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    assert( params.signDetectionMode == SignDetectionMode::Unsigned 
        || params.signDetectionMode == SignDetectionMode::OpenVDB
        || params.signDetectionMode == SignDetectionMode::HoleWindingRule );

    if ( params.voxelSize <= 0 )
    {
        assert( false );
        return unexpected( "wrong voxelSize" );
    }

    float voxelSize = params.voxelSize;
    bool useShell = params.signDetectionMode == SignDetectionMode::Unsigned;
    bool signPostprocess = params.signDetectionMode == SignDetectionMode::HoleWindingRule;

    if ( useShell )
        offset = std::abs( offset );

    auto offsetInVoxels = offset / voxelSize;

    auto voxelSizeVector = Vector3f::diagonal( voxelSize );
    // Make grid
    FloatGrid grid;
    if ( !useShell && !signPostprocess ) 
    {
        // Compute signed distance grid
        grid = meshToLevelSet( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
            subprogress( params.callBack, 0.0f, 0.5f ) );
    }
    else
    {
        // Compute unsigned distance grid
        grid = meshToDistanceField( mp, AffineXf3f(), voxelSizeVector, std::abs( offsetInVoxels ) + 2,
                        subprogress( params.callBack, 0.0f, signPostprocess ? 0.33f : 0.5f ) );
        setLevelSetType( grid ); // to flip mesh normals
    }

    if ( !grid )
        return unexpectedOperationCanceled();

    if ( signPostprocess )
    {
        // Compute signs for initially unsigned distance field
        auto sp = subprogress( params.callBack, 0.33f, 0.66f );
        auto signRes = makeSignedWithFastWinding( grid, Vector3f::diagonal( voxelSize ), mp.mesh, {}, params.fwn, sp );
        if ( !signRes.has_value() )
            return unexpected( signRes.error() );
    }

    // Make offset mesh
    auto newMesh = gridToMesh( std::move( grid ), GridToMeshSettings{
        .voxelSize = voxelSizeVector,
        .isoValue = offsetInVoxels,
        .adaptivity = 0, // it does not work good, better use common decimation after offsetting
        .cb = subprogress( params.callBack, signPostprocess ? 0.66f : 0.5f, 1.0f )
    } );

    if ( !newMesh.has_value() )
        return unexpectedOperationCanceled();

    return newMesh;
}

Expected<Mesh, std::string> thickenMesh( const Mesh& mesh, float offset, const OffsetParameters& params )
{
    MR_TIMER
    const bool unsignedOffset = params.signDetectionMode == SignDetectionMode::Unsigned;
    auto res = offsetMesh( mesh, unsignedOffset ? std::abs( offset ) : offset, params );
    if ( !res )
        return res;

    auto & resMesh = res.value();

    if ( unsignedOffset )
    {
        // do not trust degenerate faces with huge aspect ratios
        auto badFaces = findDegenerateFaces( mesh, 1000 ).value();
        // do not trust only boundary degenerate faces (excluding touching the boundary only by short edge)
        BitSetParallelFor( badFaces, [&] ( FaceId f )
        {
            float perimeter = 0;
            float bdLen = 0;
            for ( EdgeId e : leftRing( mesh.topology, f ) )
            {
                auto elen = mesh.edgeLength( e );
                perimeter += elen;
                if ( mesh.topology.isBdEdge( e ) )
                    bdLen += elen;
            }
            if ( perimeter * 0.1f >= bdLen )
                badFaces.reset( f );
        } );
        const auto goodFaces = mesh.topology.getValidFaces() - badFaces;

        // for open input mesh, let us find only necessary portion on the shell
        auto innerFaces = findInnerShellFacesWithSplits( MeshPart{ mesh, &goodFaces }, resMesh, offset > 0 ? Side::Positive : Side::Negative );
        resMesh.topology.deleteFaces( resMesh.topology.getValidFaces() - innerFaces );
        resMesh.pack();
    }

    if ( offset >= 0 )
        resMesh.addPartByMask( mesh, mesh.topology.getValidFaces(), true ); // true = with flipping
    else
    {
        if ( !unsignedOffset )
            resMesh.topology.flipOrientation();
        resMesh.addPart( mesh );
    }

    resMesh.invalidateCaches();
    return res;
}
#endif

Expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params /*= {} */ )
{
    MR_TIMER
    if ( params.signDetectionMode == SignDetectionMode::Unsigned )
    {
        spdlog::warn( "Cannot use shell for double offset, using offset mode instead." );
    }
    return levelSetDoubleConvertion( mp, AffineXf3f(), params.voxelSize, offsetA, offsetB, 0, params.fwn, params.callBack );
}

Expected<Mesh, std::string> mcOffsetMesh( const Mesh& mesh, float offset,
    const OffsetParameters& params, Vector<VoxelId, FaceId> * outMap )
{
    MR_TIMER;
    auto meshToLSCb = subprogress( params.callBack, 0.0f, 0.4f );
    if ( params.signDetectionMode == SignDetectionMode::OpenVDB )
    {
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
        auto offsetInVoxels = offset / params.voxelSize;
        auto voxelRes = meshToLevelSet( mesh, AffineXf3f(),
            Vector3f::diagonal( params.voxelSize ),
            std::abs( offsetInVoxels ) + 2, meshToLSCb );
        if ( !voxelRes )
            return unexpectedOperationCanceled();

        VdbVolume volume = floatGridToVdbVolume( voxelRes );
        volume.voxelSize = Vector3f::diagonal( params.voxelSize );

        MarchingCubesParams vmParams;
        vmParams.iso = offsetInVoxels;
        vmParams.lessInside = true;
        vmParams.cb = subprogress( params.callBack, 0.4f, 1.0f );
        vmParams.outVoxelPerFaceMap = outMap;
        return marchingCubes( volume, vmParams );
#else
        assert( false );
        return unuexpected( "OpenVDB is not available" );
#endif
    }
    else
    {
        MeshToDistanceVolumeParams msParams;
        msParams.cb = meshToLSCb;
        auto box = mesh.getBoundingBox();
        auto absOffset = std::abs( offset );
        auto expansion = Vector3f::diagonal( 2 * params.voxelSize + absOffset );
        msParams.origin = box.min - expansion;
        msParams.voxelSize = Vector3f::diagonal( params.voxelSize );
        msParams.dimensions = Vector3i( ( box.max + expansion - msParams.origin ) / params.voxelSize ) + Vector3i::diagonal( 1 );
        msParams.signMode = params.signDetectionMode;
        msParams.maxDistSq = sqr( absOffset + params.voxelSize );
        msParams.minDistSq = sqr( std::max( absOffset - params.voxelSize, 0.0f ) );
        msParams.fwn = params.fwn;
        
        auto volume = meshToDistanceVolume( mesh, msParams );
        if ( !volume )
            return unexpected( std::move( volume.error() ) );

        MarchingCubesParams vmParams;
        vmParams.origin = msParams.origin;
        vmParams.iso = offset;
        vmParams.cb = subprogress( params.callBack, 0.4f, 1.0f );
        vmParams.lessInside = true;
        vmParams.outVoxelPerFaceMap = outMap;
        return marchingCubes( std::move( *volume ), vmParams );
    }
}

Expected<Mesh, std::string> mcShellMeshRegion( const Mesh& mesh, const FaceBitSet& region, float offset,
    const BaseShellParameters& params, Vector<VoxelId, FaceId> * outMap )
{
    MR_TIMER

    DistanceVolumeParams dvParams;
    dvParams.cb = subprogress( params.callBack, 0.0f, 0.5f );
    auto box = mesh.getBoundingBox();
    auto absOffset = std::abs( offset );
    auto expansion = Vector3f::diagonal( 2 * params.voxelSize + absOffset );
    dvParams.origin = box.min - expansion;
    dvParams.voxelSize = Vector3f::diagonal( params.voxelSize );
    dvParams.dimensions = Vector3i( ( box.max + expansion - dvParams.origin ) / params.voxelSize ) + Vector3i::diagonal( 1 );

    auto volume = meshRegionToIndicatorVolume( mesh, region, offset, dvParams );
    if ( !volume )
        return unexpectedOperationCanceled();

    MarchingCubesParams vmParams;
    vmParams.origin = dvParams.origin;
    vmParams.iso = 0;
    vmParams.cb = subprogress( params.callBack, 0.5f, 1.0f );
    vmParams.lessInside = true;
    vmParams.outVoxelPerFaceMap = outMap;
    return marchingCubes( std::move( *volume ), vmParams );
}

Expected<MR::Mesh, std::string> sharpOffsetMesh( const Mesh& mesh, float offset, const SharpOffsetParameters& params )
{
    MR_TIMER
    OffsetParameters mcParams = params;
    mcParams.callBack = subprogress( params.callBack, 0.0f, 0.7f );
    Vector<VoxelId, FaceId> map;
    auto res = mcOffsetMesh( mesh, offset, mcParams, &map );
    if ( !res.has_value() )
        return res;

    SharpenMarchingCubesMeshSettings sharpenParams;
    sharpenParams.minNewVertDev = params.voxelSize * params.minNewVertDev;
    sharpenParams.maxNewRank2VertDev = params.voxelSize * params.maxNewRank2VertDev;
    sharpenParams.maxNewRank3VertDev = params.voxelSize * params.maxNewRank3VertDev;
    sharpenParams.maxOldVertPosCorrection = params.voxelSize * params.maxOldVertPosCorrection;
    sharpenParams.offset = offset;
    sharpenParams.outSharpEdges = params.outSharpEdges;

    sharpenMarchingCubesMesh( mesh, res.value(), map, sharpenParams );
    if ( !reportProgress( params.callBack, 0.99f ) )
        return unexpectedOperationCanceled();

    return res;
}

#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
Expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params /*= {} */ )
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

    // Type::Shell is more efficient in this case
    OffsetParameters p = params;
    p.signDetectionMode = SignDetectionMode::Unsigned;

    return offsetMesh( mesh, offset, p );
}
#endif

}
