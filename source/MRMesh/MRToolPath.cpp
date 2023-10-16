#include "MRToolPath.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )

#include "MR2to3.h"
#include "MRSurfacePath.h"
#include "MRFixUndercuts.h"
#include "MROffset.h"
#include "MRBox.h"
#include "MRExtractIsolines.h"
#include "MRSurfaceDistance.h"
#include "MRMeshDirMax.h"
#include "MRParallelFor.h"
#include "MRObjectGcode.h"
#include "MRExpected.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRRegionBoundary.h"
#include "MRMeshDecimate.h"
#include "MRBitSetParallelFor.h"
#include "MRDistanceMap.h"
#include "MRMeshComponents.h"
#include "MRPolylineProject.h"
#include "MRContoursCut.h"
#include "MRFillContourByGraphCut.h"
#include "MRInnerShell.h"
#include "MRRingIterator.h"
#include "MREdgePaths.h"

#include "MRParallelFor.h"
#include "MRPch/MRTBB.h"
#include <sstream>
#include <span>

namespace MR
{

Vector2f rotate90( const Vector2f& v )
{
    return { v.y, -v.x };
}

Vector2f rotateMinus90( const Vector2f& v )
{
    return { -v.y, v.x };
}

bool calcCircleCenter( const Vector2f& p0, const Vector2f& p1, const Vector2f& p2, Vector2f& center )
{
    const auto dif1 = p1 - p0;
    const auto dif2 = p2 - p0;

    const auto proj1 = dot( dif1, p0 + p1 );
    const auto proj2 = dot( dif2, p0 + p2 );
    const auto det = cross( dif1, p2 - p1 ) * 2;

    if ( fabs( det ) < 1e-10 )
        return false;

    // calc center coords
    center.x = ( dif2.y * proj1 - dif1.y * proj2 ) / det;
    center.y = ( dif1.x * proj2 - dif2.x * proj1 ) / det;

    return true;
}

// get coordinate along specified axis
float coord( const GCommand& command, Axis axis )
{
    return ( axis == Axis::X ) ? command.x :
        ( axis == Axis::Y ) ? command.y : command.z;
}
// get projection to the plane orthogonal to the specified axis
Vector2f project( const GCommand& command, Axis axis )
{
    return ( axis == Axis::X ) ? Vector2f{ command.y, command.z } :
        ( axis == Axis::Y ) ? Vector2f{ command.x, command.z } : Vector2f{ command.x, command.y };
}

Expected<Mesh, std::string> preprocessMesh( const Mesh& inputMesh, const ToolPathParams& params, bool needToDecimate )
{
    Mesh meshCopy = inputMesh;

    if ( !params.flatTool )
    {
        OffsetParameters offsetParams;
        offsetParams.voxelSize = params.voxelSize;
        offsetParams.callBack = subprogress( params.cb, 0.0f, 0.15f );
        const auto offsetRes = offsetMesh( inputMesh, params.millRadius, offsetParams );
        if ( !offsetRes )
            return unexpectedOperationCanceled();

        meshCopy = *offsetRes;
    }

    if ( params.xf )
        meshCopy.transform( *params.xf );

    if ( !reportProgress( params.cb, 0.15f ) )
        return unexpectedOperationCanceled();
    
    FixUndercuts::fixUndercuts( meshCopy, Vector3f::plusZ(), params.voxelSize );
    
    if ( !reportProgress( params.cb, 0.20f ) )
        return unexpectedOperationCanceled();

    if ( needToDecimate )
    {
        const auto decimateResult = decimateMesh( meshCopy, { .progressCallback = subprogress( params.cb, 0.20f, 0.25f ) } );
        if ( decimateResult.cancelled )
            return unexpectedOperationCanceled();
    }
        
    return meshCopy;
}

// compute surface path between given edge points
void addSurfacePath( std::vector<GCommand>& gcode, const Mesh& mesh, const MeshEdgePoint& start, const MeshEdgePoint& end )
{
    const auto sp = computeSurfacePath( mesh, start, end );
    if ( !sp.has_value() || sp->empty() )
    {
        const auto p = mesh.edgePoint( end );
        gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
        return;
    }

    if ( sp->size() == 1 )
    {
        const auto p = mesh.edgePoint( sp->front() );
        gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
    }
    else
    {
        Polyline3 transit;
        transit.addFromSurfacePath( mesh, *sp );
        const auto transitContours = transit.contours().front();
        for ( const auto& p : transitContours )
            gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
    }

    const auto p = mesh.edgePoint( end );
    gcode.push_back( { .x = p.x, .y = p.y, .z = p.z } );
}

// computes all sections of modified mesh along the given axis
// if a selected area is specified in the original mesh, then only points projected on it will be taken into consideration
std::vector<PlaneSections> extractAllSections( const Mesh& mesh, const Box3f& box, Axis axis, float sectionStep, int steps, BypassDirection bypassDir, ProgressCallback cb )
{
    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };

    std::vector<PlaneSections> sections( steps );

    const int axisIndex = int( axis );
    constexpr Vector3f normals[3] = { {1, 0, 0}, {0, 1, 0}, {0, 0 ,1} };
    const Vector3f normal = normals[axisIndex];
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );

    tbb::parallel_for( tbb::blocked_range<int>( 0, steps ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int step = range.begin(); step < range.end(); ++step )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            const float currentCoord = plane.d - sectionStep * step;

            auto stepSections = extractPlaneSections( mesh, Plane3f{ plane.n, currentCoord } );

            if ( bypassDir == BypassDirection::Clockwise )
            {
                sections[step] = std::move( stepSections );
                continue;
            }

            sections[step].reserve( stepSections.size() );

            for ( auto& section : stepSections )
            {
                std::reverse( section.begin(), section.end() );
                sections[step].push_back( std::move( section ) );
            }
        }

        if ( cb )
            numDone += range.size();

        if ( cb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !cb( float( numDone ) / float( steps ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( !keepGoing.load( std::memory_order_relaxed ) || !reportProgress( cb, 1.0f ) )
        return {};

    return sections;
}

struct ExtractIsolinesResult
{
    std::list<SurfacePath> sortedIsolines;
    Mesh meshAfterCut;
    Vector<std::vector<FaceId>, FaceId> old2NewMap;
    FaceBitSet region;
};

struct ExtractIsolinesParams
{
    const FaceBitSet* region;
    // if is not null, isolines will be based on this surface path
    const std::vector<EdgeLoop>& startSurfacePaths;
    // if is not null, coordinates of the start contours will be stored here
    Contours3f* startContours;
    // if is not null, coordinates of the start contours will be stored here
    Contours3f* contoursBeforeCutMesh;
    // if is not null, coordinates of the contours failed to build isolines will be stored here
    Contours3f* failedContours;
    // if is not null, coordinates of the start vertices will be stored here
    std::vector<Vector3f>* startVertices;
    // distance between isolines
    float sectionStep;
    // which direction isolines should be passed in
    BypassDirection bypassDir;
    //
    bool fromCenterToBoundary;
    // callback for reporting on progress
    ProgressCallback cb;    
};

ExtractIsolinesResult extractAllIsolines( const Mesh& mesh, const ExtractIsolinesParams& params )
{
    ExtractIsolinesResult res;
    MR::VertScalars allDistances;
    // if startContours is provided add a new contour
    Contour3f* startContour = nullptr;
    Contour3f* contourBeforeCutMesh = nullptr;
    

    res.meshAfterCut = mesh;

    HashMap<VertId, float> allStartVerticesWithDists;

    VertBitSet vertRegion;

    for ( const auto& startSurfacePath : params.startSurfacePaths )
    {
        if ( params.startContours )
        {
            params.startContours->emplace_back();
            startContour = &params.startContours->back();
        }

        if ( params.contoursBeforeCutMesh )
        {
            params.contoursBeforeCutMesh->emplace_back();
            contourBeforeCutMesh = &params.contoursBeforeCutMesh->back();
        }

        //Polyline3 startPolyline;
        //startPolyline.addFromSurfacePath( res.meshAfterCut, startSurfacePath );
        HashMap<VertId, float> startVerticesWithDists;

        for ( const auto& ep : startSurfacePath )
        {
            const auto orgVertId = res.meshAfterCut.topology.org( ep );
            const auto dstVertId = res.meshAfterCut.topology.dest( ep );
            if ( !orgVertId.valid() || !dstVertId.valid() )
                continue;

            vertRegion.autoResizeSet( orgVertId );
            vertRegion.autoResizeSet( dstVertId );
            // if removed then warning C4686: 'MR::findProjectionOnPolyline2': possible change in behavior, change in UDT return calling convention
            static PolylineProjectionResult3 unused;

            /*auto proj = findProjectionOnPolyline(res.meshAfterCut.points[orgVertId], startPolyline);
            float dist = sqrt( proj.distSq );
            startVerticesWithDists.insert_or_assign( orgVertId, dist );
            if ( params.startVertices )
                params.startVertices->push_back( { res.meshAfterCut.points[orgVertId] } );

            proj = findProjectionOnPolyline( res.meshAfterCut.points[dstVertId], startPolyline );
            dist = sqrt( proj.distSq );
            startVerticesWithDists.insert_or_assign( dstVertId, dist );*/
            if ( params.startVertices )
                params.startVertices->push_back( { res.meshAfterCut.points[dstVertId] } );

            if ( startContour )
                startContour->push_back( res.meshAfterCut.edgePoint( { ep, 0.0f } ) );
        }

        allStartVerticesWithDists.insert( startVerticesWithDists.begin(), startVerticesWithDists.end() );
        auto distances = computeSurfaceDistances( res.meshAfterCut, startVerticesWithDists );

        
        const auto innerFaces = fillContourLeftByGraphCut( res.meshAfterCut.topology, startSurfacePath, edgeCurvMetric( res.meshAfterCut ) );
        BitSetParallelFor( getInnerVerts( res.meshAfterCut.topology, &innerFaces ), [&] ( VertId v ) 
        {
            distances[v] = -distances[v];
        } );

        const auto shiftedIsoline = extractIsolines( res.meshAfterCut.topology, distances, params.sectionStep ).front();
        if ( contourBeforeCutMesh )
        for ( const auto& ep : shiftedIsoline )
        {
            contourBeforeCutMesh->push_back( res.meshAfterCut.edgePoint( ep ) );
        }
        //distances = computeSurfaceDistances( res.meshAfterCut, startVerticesWithDists );

        /*startPolyline = Polyline3{};
        startPolyline.addFromSurfacePath( res.meshAfterCut, shiftedIsoline );
        startVerticesWithDists.clear();

        for ( const auto& ep : shiftedIsoline )
        {
            const auto orgVertId = res.meshAfterCut.topology.org( ep.e );
            const auto dstVertId = res.meshAfterCut.topology.dest( ep.e );
            if ( !orgVertId.valid() || !dstVertId.valid() )
                continue;

            auto proj = findProjectionOnPolyline( res.meshAfterCut.points[orgVertId], startPolyline );
            float dist = sqrt( proj.distSq );
            startVerticesWithDists.insert_or_assign( orgVertId, dist );
          

            proj = findProjectionOnPolyline( res.meshAfterCut.points[dstVertId], startPolyline );
            dist = sqrt( proj.distSq );
            startVerticesWithDists.insert_or_assign( dstVertId, dist );
        }

       
        auto offsetDistances = computeSurfaceDistances( res.meshAfterCut, startVerticesWithDists, FLT_MAX );

        if ( params.region )
        {
            auto offsetRegion = *params.region;
            dilateRegion( res.meshAfterCut, offsetRegion, params.sectionStep );

            BitSetParallelFor( getInnerVerts( res.meshAfterCut.topology, params.region ), [&] ( VertId v )
            {
                offsetDistances[v] = -offsetDistances[v];
            } );
        }        

        auto smoothedPath = extractIsolines( res.meshAfterCut.topology, offsetDistances, -params.sectionStep ).front();*/

       
    }

    allDistances = computeSurfaceDistances( res.meshAfterCut, vertRegion, FLT_MAX );

    const float topExcluded = FLT_MAX;
    const auto [min, max] = parallelMinMax( allDistances.vec_, &topExcluded );
    
    size_t numIsolines = size_t( ( max - min ) / params.sectionStep );
    if ( numIsolines == 0 )
        return res;

    const auto& topology = res.meshAfterCut.topology;
    const auto firstIsolines = extractIsolines( topology, allDistances, params.sectionStep );
    const size_t groupCount = firstIsolines.size();

    
    std::vector<std::list<SurfacePath>::iterator> groupStarts;
    groupStarts.reserve( groupCount );
    std::vector<Vector3f> groupStartPositions;
    groupStartPositions.reserve( groupCount );

    for ( const auto& isoline : firstIsolines )
    {
        res.sortedIsolines.push_front( isoline );
        groupStarts.push_back( res.sortedIsolines.begin() );
        groupStartPositions.push_back( res.meshAfterCut.edgePoint( groupStarts.back()->front() ) );
    }

    for ( size_t i = 1; i < numIsolines - 1; ++i )
    {
        auto isolines = extractIsolines( topology, allDistances, params.sectionStep * ( i + 1 ) );
        if ( params.bypassDir == BypassDirection::CounterClockwise )
        {
            for ( auto& isoLine : isolines )
            {
                std::reverse( isoLine.begin(), isoLine.end() );
            }
        }

        for ( const auto& isoLine : isolines )
        {
            float minDist = FLT_MAX;
            size_t nearestIdx = groupCount;

            for ( auto ep : isoLine )
            {
                for ( size_t j = 0; j < groupCount; ++j )
                {
                    const float dist = ( res.meshAfterCut.edgePoint( ep ) - groupStartPositions[j] ).lengthSq();
                    if ( dist < minDist )
                    {
                        nearestIdx= j;
                        minDist = dist;
                    }
                }
            }

            groupStarts[nearestIdx] = res.sortedIsolines.insert( groupStarts[nearestIdx], isoLine );
            groupStartPositions[nearestIdx] = res.meshAfterCut.edgePoint( groupStarts[nearestIdx]->front() );            
        }        
    }

    if ( params.fromCenterToBoundary )
    {
        for ( size_t i = 0; i < groupCount - 1; ++i )
            std::reverse( groupStarts[i], groupStarts[i + 1] );
        
        std::reverse( groupStarts.back(), res.sortedIsolines.end() );
    }

    if ( !reportProgress( params.cb, 1.0f ) )
        return {};

    return res;
}

using V3fIt = std::vector<Vector3f>::const_iterator;
using Intervals = std::vector<std::pair<V3fIt, V3fIt>>;

// compute intervals of the slice which are projected on the selected area in the mesh (skip if not specified)
Intervals getIntervals( const MeshPart& mp, const MeshPart* offset, const V3fIt startIt, const V3fIt endIt, const V3fIt beginVec, const V3fIt endVec, bool moveForward, float toolRadius )
{
    Intervals res;
    if ( startIt == endIt )
        return res;

    auto startInterval = moveForward ? startIt : endIt;
    auto endInterval = startInterval;

    // if the point is projected on the selected area expand the current interval (forward or backward in depending of the parameter )
    // otherwise add interval to the result
    const auto processPoint = [&] ( V3fIt it )
    {
        const float maxDistSq = 2 * toolRadius * toolRadius;
        const auto mpr = offset ? offset->mesh.projectPoint( *it, maxDistSq ) : mp.mesh.projectPoint( *it, maxDistSq );

        bool isInsideSelection = mpr.has_value();

        if ( isInsideSelection )
        {
            const auto faceId = offset ? offset->mesh.topology.left( mpr->mtp.e ) : mp.mesh.topology.left( mpr->mtp.e );

            isInsideSelection = offset ?
                ( !offset->region || ( mpr && offset->region->test( faceId ) ) ) :
                !mp.region || ( mpr && mp.region->test( faceId ) );
        }

        if ( isInsideSelection )
        {
            if ( moveForward || endInterval > beginVec )
                moveForward ? ++endInterval : --endInterval;
            return;
        }

        if ( startInterval != endInterval )
        {
            if ( !moveForward && startInterval == endVec )
                res.emplace_back( startInterval - 1, endInterval );
            else
                res.emplace_back( startInterval, endInterval );
        }

        if ( moveForward )
            startInterval = endInterval = it + 1;
        else
            startInterval = endInterval = it - 1;
    };

    // we might be able to pass from the start point to the end directly directly,
    //otherwise we have to reach end ( beginning ) of the entire section and continue from beginning (end)
    if ( moveForward )
    {
        if ( startIt < endIt )
        {
            for ( auto it = startIt; it < endIt; ++it )
                processPoint( it );

            if ( startInterval < endInterval )
                res.emplace_back( startInterval, endInterval );

            return res;
        }

        for ( auto it = startIt; it < endVec; ++it )
            processPoint( it );

        if ( startInterval < endInterval )
            res.emplace_back( startInterval, endInterval );

        startInterval = beginVec;
        endInterval = beginVec;

        for ( auto it = beginVec; it < endIt; ++it )
            processPoint( it );

        if ( startInterval != endInterval )
            res.emplace_back( startInterval, endInterval );
    }
    else
    {
        if ( startIt < endIt )
        {
            for ( auto it = endIt - 1; it >= startIt; --it )
                processPoint( it );

            if ( startInterval != endInterval )
                res.emplace_back( startInterval, endInterval );

            return res;
        }

        for ( auto it = endIt - 1; it > beginVec; --it )
            processPoint( it );

        processPoint( beginVec );

        if ( startInterval != endInterval )
            res.emplace_back( startInterval, endInterval );

        startInterval = endVec;
        endInterval = endVec;

        for ( auto it = endVec - 1; it >= startIt; --it )
            processPoint( it );

        if ( startInterval == endVec )
            --startInterval;

        if ( startInterval != endInterval )
            res.emplace_back( startInterval, endInterval );
    }

    return res;
}

// if distance between the last point and the given one is more than critical distance
// we should make a transit on the safe height
void transitOverSafeZ( const Vector3f& p, ToolPathResult& res, const ToolPathParams& params, float safeZ, float currentZ, float& lastFeed )
{
    // retract the tool fast if possible
    if ( safeZ - currentZ > params.retractLength )
    {
        const float zRetract = currentZ + params.retractLength;
        res.commands.push_back( { .feed = params.retractFeed, .z = zRetract } );
        res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
    }
    else if ( safeZ != currentZ )
    {
        res.commands.push_back( { .feed = params.retractFeed, .z = safeZ } );
    }

    res.commands.push_back( { .type = MoveType::FastLinear, .x = p.x, .y = p.y } );

    // plunge the tool fast if possible
    if ( safeZ - p.z > params.plungeLength )
    {
        const float zPlunge = p.z + params.plungeLength;
        res.commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
    }
    res.commands.push_back( { .feed = params.plungeFeed, .x = p.x, .y = p.y, .z = p.z } );
    lastFeed = params.plungeFeed;
}

Expected<ToolPathResult, std::string> lacingToolPath( const MeshPart& mp, const ToolPathParams& params, Axis cutDirection )
{
    if ( cutDirection != Axis::X && cutDirection != Axis::Y )
        return unexpected( "Lacing can be done along the X or Y axis" );

    const auto cutDirectionIdx = int( cutDirection );
    const auto sideDirection = ( cutDirection == Axis::X ) ? Axis::Y : Axis::X;
    const auto sideDirectionIdx = int( sideDirection );

    ToolPathResult res;

    if ( !params.offsetMesh )
    {
        auto preprocessedMesh = preprocessMesh( mp.mesh, params, false );
        if ( !preprocessedMesh )
            return unexpected( preprocessedMesh.error() );

        res.modifiedMesh = std::move( *preprocessedMesh );
    }

    const auto& mesh = params.offsetMesh ? params.offsetMesh->mesh : res.modifiedMesh;

    const auto box = mesh.computeBoundingBox();
    const float safeZ = std::max( box.max.z + 10.0f * params.millRadius, params.safeZ );

    const Vector3f normal = (cutDirection == Axis::X) ? Vector3f::plusX() : Vector3f::plusY();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min[cutDirectionIdx] ) / params.sectionStep ) );

    MeshEdgePoint lastEdgePoint = {};
    // bypass direction is not meaningful for this toolpath, so leave it as default
    const auto allSections = extractAllSections( mesh, box, cutDirection, params.sectionStep, steps, BypassDirection::Clockwise, subprogress( params.cb, 0.25f, 0.5f ) );
    if ( allSections.empty() )
        return unexpectedOperationCanceled();
    const auto sbp = subprogress( params.cb, 0.5f, 1.0f );

    float lastFeed = 0;  

    Vector3f lastPoint;
    // if the last point is equal to parameter, do nothing
    // otherwise add new move
    const auto addPoint = [&] ( const Vector3f& point )
    {
        if ( lastPoint == point )
            return;

        if ( lastFeed == params.baseFeed )
        {
            ( cutDirection == Axis::X ) ?
            res.commands.push_back( { .y = point.y, .z = point.z } ) :
            res.commands.push_back( { .x = point.x, .z = point.z } );
        }
        else
        {
            ( cutDirection == Axis::X ) ?
            res.commands.push_back( { .feed = params.baseFeed, .y = point.y, .z = point.z } ) :
            res.commands.push_back( { .feed = params.baseFeed, .x = point.x, .z = point.z } );

            lastFeed = params.baseFeed;
        }
            
        lastPoint = point;
    };

    const float critDistSq = params.critTransitionLength * params.critTransitionLength;

    for ( int step = 0; step < steps; ++step )
    {
        if ( !reportProgress( sbp, float( step ) / steps ) )
            return unexpectedOperationCanceled();

        const auto sections = allSections[step];
        if ( sections.empty() )
            continue;

        // there could be many sections in one slice
        for ( const auto& section : sections )
        {
            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            auto contours = polyline.contours();
            auto& contour = contours.front();

            if ( contour.size() < 3 )
                continue;

            if ( contour.size() > section.size() )
                contour.resize( section.size() );

            if ( params.isolines )
                params.isolines->push_back( contour );

            // we need to find the most left and the most right point on the mesh
            // and move tol from one side to another
            auto bottomLeftIt = contour.end();
            auto bottomRightIt = contour.end();

            for ( auto it = contour.begin(); it < contour.end(); ++it )
            {
                if ( bottomLeftIt == contour.end() || ( *it )[sideDirectionIdx] < ( *bottomLeftIt )[sideDirectionIdx] || ( ( *it )[sideDirectionIdx] == ( *bottomLeftIt )[sideDirectionIdx] && it->z < bottomLeftIt->z ) )
                    bottomLeftIt = it;

                if ( bottomRightIt == contour.end() || ( *it )[sideDirectionIdx] > ( *bottomRightIt )[sideDirectionIdx] || ( ( *it )[sideDirectionIdx] == ( *bottomRightIt )[sideDirectionIdx] && it->z < bottomRightIt->z ) )
                    bottomRightIt = it;
            }

            // move from left to right and then  from right to left to make the smoothest path
            const bool moveForward = step & 1;

            if ( cutDirection == Axis::Y )
            {
                std::swap( bottomLeftIt, bottomRightIt );
                if ( !moveForward && bottomLeftIt != contour.begin() )
                    --bottomLeftIt;
            }
            
            const auto intervals = getIntervals( mp, params.offsetMesh, bottomLeftIt, bottomRightIt, contour.begin(), contour.end(), moveForward, params.millRadius );
            if ( intervals.empty() )
                continue;

            // go to the first point through the safe height
            if ( res.commands.empty() )
            {
                res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                res.commands.push_back( { .type = MoveType::FastLinear, .x = intervals[0].first->x, .y = intervals[0].first->y } );
                res.commands.push_back( { .type = MoveType::FastLinear, .z = intervals[0].first->z } );
            }
            else
            {
                // otherwise compute distance from the last point to a new one and decide how to get to it
                const auto nextEdgePoint = section[intervals[0].first - contour.begin()];
                const auto distSq = ( mesh.edgePoint( lastEdgePoint ) - mesh.edgePoint( nextEdgePoint ) ).lengthSq();

                if ( distSq > critDistSq )
                    transitOverSafeZ( *intervals[0].first, res, params, safeZ, res.commands.back().z, lastFeed );
                else
                    addSurfacePath( res.commands, mesh, lastEdgePoint, nextEdgePoint );
            }
            
            // process all the intervals except the last one and transit to the next
            for ( size_t i = 0; i < intervals.size() - 1; ++i )
            {
                const auto& interval = intervals[i];

                if ( moveForward )
                {
                    for ( auto it = interval.first; it < interval.second; ++it )
                        addPoint( *it );
                }
                else
                {
                    if ( interval.first == contour.begin() )
                        continue;

                    for ( auto it = interval.first - 1; it >= interval.second; --it )
                        addPoint( *it );
                }

                if ( *intervals[i + 1].first != lastPoint )
                    transitOverSafeZ( *intervals[i + 1].first, res, params, safeZ, res.commands.back().z, lastFeed );
            }
            // process the last interval
            if ( moveForward )
            {
                for ( auto it = intervals.back().first; it < intervals.back().second; ++it )
                    addPoint( *it );
            }
            else
            {
                for ( auto it = intervals.back().first - 1; it >= intervals.back().second; --it )
                    addPoint( *it );
            }

            const auto dist = ( intervals.back().second - contour.begin() ) % contour.size();
            lastEdgePoint = section[dist];
        }
    }

    if ( !reportProgress( params.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<ToolPathResult, std::string>  constantZToolPath( const MeshPart& mp, const ToolPathParams& params )
{
    ToolPathResult  res;

    if ( !params.offsetMesh )
    {
        auto preprocessedMesh = preprocessMesh( mp.mesh, params, false );
        if ( !preprocessedMesh )
            return unexpected( preprocessedMesh.error() );

        res.modifiedMesh = std::move( *preprocessedMesh );
    }

    const auto& mesh = params.offsetMesh ? params.offsetMesh->mesh : res.modifiedMesh;

    const auto box = mesh.computeBoundingBox();
    const float safeZ = std::max( box.max.z + 10.0f * params.millRadius, params.safeZ );
    float lastZ = 0;

    const Vector3f normal = Vector3f::plusZ();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / params.sectionStep ) );


    MeshEdgePoint prevEdgePoint;

    const float critTransitionLengthSq = params.critTransitionLength * params.critTransitionLength;

    std::vector<PlaneSections> sections = extractAllSections( mesh, box, Axis::Z, params.sectionStep, steps, params.bypassDir, subprogress( params.cb, 0.25f, 0.5f ) );
    if ( sections.empty() )
        return unexpectedOperationCanceled();

    const auto sbp = subprogress( params.cb, 0.5f, 1.0f );

    float lastFeed = 0;

    Vector3f lastPoint;
    // if the last point is equal to parameter, do nothing
    // otherwise add new move
    const auto addPoint = [&] ( const Vector3f& point )
    {
        if ( lastPoint == point )
            return;

        GCommand command { .x = point.x, .y = point.y };
        if ( lastFeed != params.baseFeed )
        {
            command.feed = params.baseFeed;
            lastFeed = params.baseFeed;
        }

        if ( lastZ != point.z )
        {
            command.z = point.z;
            lastZ = point.z;
        }

        res.commands.push_back( command );
        lastPoint = point;
    };

    for ( int step = 0; step < steps; ++step )
    {
        if ( !reportProgress( sbp, float( step ) / steps ) )
            return unexpectedOperationCanceled();

        auto& commands = res.commands;

        for ( const auto& section : sections[step] )
        {
            if ( section.size() < 2 )
                continue;

            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            
            if ( params.flatTool )
            {
                const auto currentZ = polyline.points.front().z;
                auto polyline2d = polyline.toPolyline<Vector2f>();
                const ContourToDistanceMapParams dmParams( params.voxelSize, polyline2d.contours(), params.millRadius + 3.0f * params.voxelSize, true );
                const ContoursDistanceMapOptions dmOptions{ .signMethod = ContoursDistanceMapOptions::WindingRule, .minDist = params.millRadius - 2 * params.voxelSize, .maxDist = params.millRadius + 2 * params.voxelSize };
                const auto dm = distanceMapFromContours( polyline2d, dmParams, dmOptions );
                DistanceMapToWorld dmToWorld( dmParams );
                auto offsetRes = distanceMapTo2DIsoPolyline( dm, dmToWorld, params.millRadius );
                polyline2d = offsetRes.first;
                polyline = polyline2d.toPolyline<Vector3f>();
                polyline.transform( offsetRes.second * AffineXf3f::translation( { 0, 0, currentZ } ) );
            }

            auto contours = polyline.contours();
            if ( contours.empty() )
                continue;

            auto& contour = contours.front();
            if ( !params.flatTool && contour.size() > section.size() )
                contour.resize( section.size() );

            if ( params.isolines )
                params.isolines->push_back( contour );

            if ( mp.region || params.flatTool )
            {
                const auto intervals = getIntervals( mp, params.offsetMesh, contour.begin(), contour.end(), contour.begin(), contour.end(), true, params.millRadius );
                if ( intervals.empty() )
                    continue;

                for ( const auto& interval : intervals )
                {
                    if ( !mp.region || interval.first != contour.begin() || res.commands.empty() )
                    {
                        if ( res.commands.empty() )
                            res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );

                        transitOverSafeZ( *interval.first, res, params, safeZ, lastZ, lastFeed );
                        commands.push_back( { .x = interval.first->x, .y = interval.first->y, .z = interval.first->z } );
                    }

                    for ( auto it = interval.first; it < interval.second; ++it )
                    {
                        addPoint( *it );
                    }
                }

                continue;
            }

            auto nearestPointIt = section.begin();
            auto nextEdgePointIt = section.begin();
            float minDistSq = FLT_MAX;

            if ( prevEdgePoint.e.valid() )
            {
                for ( auto it = section.begin(); it < section.end(); ++it )
                {
                    float distSq = ( mesh.edgePoint( *it ) - mesh.edgePoint( prevEdgePoint ) ).lengthSq();
                    if ( distSq < minDistSq )
                    {
                        minDistSq = distSq;
                        nearestPointIt = it;
                    }
                }

                const float sectionStepSq = params.sectionStep * params.sectionStep;
                const auto nearestPoint = mesh.edgePoint( *nearestPointIt );
                nextEdgePointIt = nearestPointIt;
                do
                {
                    std::next( nextEdgePointIt ) != section.end() ? ++nextEdgePointIt : nextEdgePointIt = section.begin();
                } while ( nextEdgePointIt != nearestPointIt && ( mesh.edgePoint( *nextEdgePointIt ) - nearestPoint ).lengthSq() < sectionStepSq );
            }

            const auto pivotIt = contour.begin() + std::distance( section.begin(), nextEdgePointIt );

            if ( !prevEdgePoint.e.valid() || minDistSq > critTransitionLengthSq )
            {
                transitOverSafeZ( *pivotIt, res, params, safeZ, lastZ, lastFeed );
            }
            else
            {
                const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt );
                if ( sp.has_value() && !sp->empty() )
                {
                    if ( sp->size() == 1 )
                    {
                        const auto p = mesh.edgePoint( sp->front() );
                        res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
                    }
                    else
                    {
                        Polyline3 transit;
                        transit.addFromSurfacePath( mesh, *sp );
                        const auto transitContours = transit.contours().front();
                        for ( const auto& p : transitContours )
                            commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
                    }
                }

                commands.push_back( { .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
            }

            auto startIt = pivotIt + 1;

            for ( auto it = startIt; it < contour.end(); ++it )
            {
                addPoint( *it );
            }

            for ( auto it = contour.begin() + 1; it < pivotIt + 1; ++it )
            {
                addPoint( *it );
            }

            prevEdgePoint = *nextEdgePointIt;
            lastZ = pivotIt->z;
        }
    }

    if ( !reportProgress( params.cb, 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}

Expected<ToolPathResult, std::string> constantCuspToolPath( const MeshPart& mp, const ConstantCuspParams& params )
{
    ToolPathResult  res;

    if ( !params.offsetMesh )
    {
        auto preprocessedMesh = preprocessMesh( mp.mesh, params, false );
        if ( !preprocessedMesh )
            return unexpected( preprocessedMesh.error() );

        res.modifiedMesh = std::move( *preprocessedMesh );
    }
    else
    {
        res.modifiedMesh = params.offsetMesh->mesh;
    }
    const auto box = res.modifiedMesh.computeBoundingBox();

    const Vector3f normal = Vector3f::plusZ();
    float minZ = box.min.z + params.millRadius;
    float safeZ = std::max( box.max.z + params.millRadius, params.safeZ );

    const auto undercutPlane = MR::Plane3f::fromDirAndPt( normal, { 0.0f, 0.0f, minZ } );
    
    //compute the lowest contour that might be processed
    const auto undercutSection = extractPlaneSections( res.modifiedMesh, undercutPlane ).front();
    Polyline3 undercutPolyline;
    undercutPolyline.addFromSurfacePath( res.modifiedMesh, undercutSection );
    const auto undercutContour = undercutPolyline.contours().front();

    // if there are multiple independent zones selected we need to process them separately
    const auto processZone = [&] ( const std::vector<EdgeLoop>& startSurfacePaths, Vector3f lastPoint, ProgressCallback cb ) -> bool
    {
        ExtractIsolinesParams extractionParams
        {
            .region = mp.region,
            .startSurfacePaths = startSurfacePaths,
            .startContours = params.startContours,
            .contoursBeforeCutMesh = params.contoursBeforeCutMesh,
            .failedContours = params.failedContours,
            .startVertices = params.startVertices,
            .sectionStep = params.sectionStep,
            .bypassDir = params.bypassDir,
            .cb = subprogress( cb, 0.0f, 0.4f )
        };
        //compute isolines based on the start point or the bounding contour
        auto extract = extractAllIsolines( res.modifiedMesh, extractionParams );
        res.modifiedMesh = extract.meshAfterCut;
        
        if ( !extract.old2NewMap.empty() )
        {
            const auto oldRegion = res.modifiedRegion;
            res.modifiedRegion = extract.region;

            BitSetParallelFor( oldRegion, [&] ( FaceId f )
            {
                for ( auto& newFace : extract.old2NewMap[f] )
                {
                    res.modifiedRegion.set( newFace, true );
                }
            } );
        }

        const auto& mesh = res.modifiedMesh;
        if ( extract.sortedIsolines.empty() )
            return reportProgress( cb, 0.4f );

        //go to the start point through safe height
        res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
        lastPoint.z = safeZ;
        float lastFeed = 0.0f;

        MeshEdgePoint prevEdgePoint;

        std::optional<Vector3f> startSkippedRegion;
        
        // skip point if it is the same as the last one
        const auto addPointToTheToolPath = [&] ( const Vector3f& p )
        {
            if ( p == lastPoint )
                return;

            res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
            lastPoint = p;
        };

        //returns the nearest point in the contour to the given point
        const auto findNearestPoint = [] ( const Contour3f& contour, const Vector3f& p )
        {
            auto res = contour.begin();
            if ( res == contour.end() )
                return res;

            float minDistSq = ( p - *res ).lengthSq();

            for ( auto it = std::next( contour.begin() ); it != contour.end(); ++it )
            {
                const float distSq = ( p - *it ).lengthSq();
                if ( distSq < minDistSq )
                {
                    minDistSq = distSq;
                    res = it;
                }
            }

            return res;
        };

        // compute bitset of the vertices that are not belonged to the undercut
        VertBitSet noUndercutVertices( mesh.points.size() );
        BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
        {
            noUndercutVertices.set( v, mesh.points[v].z >= minZ );
        } );

        ToolPathParams paramsCopy = params;
        paramsCopy.critTransitionLength = 0;

        //append a part of a contour to the result
        const auto addSliceToTheToolPath = [&] ( const Contour3f::const_iterator startIt, Contour3f::const_iterator endIt )
        {
            for ( auto it = startIt; it < endIt; ++it )
            {
                //skip point if it lies in the undercut
                bool isPointAppropriate = ( it->z >= minZ );
                if ( !isPointAppropriate )
                {
                    if ( !startSkippedRegion )
                        startSkippedRegion = *it;

                    continue;
                }
                if ( startSkippedRegion )
                {
                    transitOverSafeZ( *it, res, paramsCopy, safeZ, res.commands.back().z, lastFeed );
                    startSkippedRegion.reset();
                    continue;
                }
                
                addPointToTheToolPath( *it );
            }
        };

        if ( !reportProgress( cb, 0.5f ) )
            return false;

        const auto sbp = subprogress( cb, 0.5f, 1.0f );

       // if ( params.fromCenterToBoundary )
         //   std::reverse( extract.isolines.begin(), extract.isolines.end() );
        
        // go on in the inverse order (from the highest isoline to the lowest )
        auto isolineIt = extract.sortedIsolines.begin();
        while ( isolineIt != extract.sortedIsolines.end() )
        {
            auto& surfacePath = *isolineIt++;

            if ( surfacePath.empty() || !res.modifiedRegion.test( mesh.topology.left( surfacePath[0].e ) ) )
                continue;

            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, surfacePath );
            const auto contours = polyline.contours();
            const auto& contour = contours.front();

            if ( params.isolines )
                params.isolines->push_back( contour );

            auto nearestPointIt = surfacePath.begin();
            float minDistSq = FLT_MAX;

            // find the nearest point to the last processed one
            if ( prevEdgePoint.e.valid() )
            {
                for ( auto it = surfacePath.begin(); it < surfacePath.end(); ++it )
                {
                    float distSq = ( mesh.edgePoint( *it ) - mesh.edgePoint( prevEdgePoint ) ).lengthSq();
                    if ( distSq < minDistSq )
                    {
                        minDistSq = distSq;
                        nearestPointIt = it;
                    }
                }
            }

            auto nextEdgePointIt = nearestPointIt;
            const float sectionStepSq = params.sectionStep * params.sectionStep;
            const auto nearestPoint = mesh.edgePoint( *nearestPointIt );

            // make severals steps ahead for smoother transit
            Vector3f tmp;
            bool skipPoint = false;
            do
            {
                std::next( nextEdgePointIt ) != surfacePath.end() ? ++nextEdgePointIt : nextEdgePointIt = surfacePath.begin();
                tmp = mesh.edgePoint( *nextEdgePointIt );
                skipPoint = ( tmp.z < minZ );
            } while ( nextEdgePointIt != nearestPointIt && ( ( ( tmp - nearestPoint ).lengthSq() < sectionStepSq ) || skipPoint ) );

            const auto pivotIt = contour.begin() + std::distance( surfacePath.begin(), nextEdgePointIt );
            if ( skipPoint )
                continue;

            if ( !prevEdgePoint.e.valid() )
            {
                // go through the safe height to the first point
                res.commands.push_back( { .type = MoveType::FastLinear, .x = pivotIt->x, .y = pivotIt->y } );

                if ( safeZ - pivotIt->z > params.plungeLength )
                {
                    const float zPlunge = pivotIt->z + params.plungeLength;
                    res.commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
                }

                res.commands.push_back( { .feed = params.plungeFeed, .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
            }
            else
            {
                // go along the undercut if the part of section is below
                const auto p1 = mesh.edgePoint( prevEdgePoint );
                const auto p2 = mesh.edgePoint( *nextEdgePointIt );
                if ( p1.z == minZ )
                {
                    const auto sectionStartIt = findNearestPoint( undercutContour, p1 );
                    const auto sectionEndIt = findNearestPoint( undercutContour, p2 );

                    if ( sectionStartIt < sectionEndIt )
                    {
                        for ( auto sectionIt = sectionStartIt; sectionIt <= sectionEndIt; ++sectionIt )
                            addPointToTheToolPath( *sectionIt );
                    }
                    else
                    {
                        for ( auto sectionIt = sectionStartIt; sectionIt < undercutContour.end(); ++sectionIt )
                            addPointToTheToolPath( *sectionIt );

                        for ( auto sectionIt = std::next( undercutContour.begin() ); sectionIt <= sectionEndIt; ++sectionIt )
                            addPointToTheToolPath( *sectionIt );
                    }

                    addPointToTheToolPath( p2 );
                }
                else
                {
                    // go along the mesh in other cases
                    const auto sp = computeSurfacePath( mesh, prevEdgePoint, *nextEdgePointIt, 5, &noUndercutVertices );
                    if ( sp.has_value() && !sp->empty() )
                    {
                        if ( sp->size() == 1 )
                        {
                            addPointToTheToolPath( mesh.edgePoint( sp->front() ) );
                        }
                        else
                        {
                            Polyline3 transit;
                            transit.addFromSurfacePath( mesh, *sp );
                            const auto transitContours = transit.contours().front();
                            for ( const auto& p : transitContours )
                                addPointToTheToolPath( p );
                        }
                    }
                }

                res.commands.push_back( { .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
            }

            addSliceToTheToolPath( pivotIt + 1, contour.end() );
            addSliceToTheToolPath( contour.begin() + 1, pivotIt + 1 );
            prevEdgePoint = *nextEdgePointIt;
        }   

        return true;
    };
    
    //if selection is not specified then process all the vertices above the undercut
    if ( !mp.region && ( !params.offsetMesh || !params.offsetMesh->region ) )
    {
        std::vector<EdgeLoop> edgeLoops( 1 );
        edgeLoops[0].resize( undercutSection.size() );
        ParallelFor( size_t( 0 ), undercutSection.size(), [&] ( size_t i )
        {
            edgeLoops[0][i] = undercutSection[i].e;
        } );

        if ( !processZone( edgeLoops, {}, subprogress( params.cb, 0.25f, 1.0f ) ) || !reportProgress( params.cb, 1.0f ) )
            return unexpectedOperationCanceled();

        return res;
    }

    const auto vertBitSet = findInnerShellVerts( mp, res.modifiedMesh, Side::Positive, params.millRadius * 1.5f );
    res.modifiedRegion.resize( res.modifiedMesh.topology.lastValidFace() + 1 );
    BitSetParallelFor( vertBitSet, [&] ( VertId v )
    {
        for ( auto e : orgRing( res.modifiedMesh.topology, res.modifiedMesh.topology.edgePerVertex()[v] ) )
        {
            res.modifiedRegion.set( res.modifiedMesh.topology.left( e ) );
        }
    } );

    const auto components = MeshComponents::getAllComponents( MeshPart{ res.modifiedMesh, &res.modifiedRegion } );
    const size_t componentCount = components.size();
    std::vector<EdgeLoop> startSurfacePaths;

    for ( size_t i = 0; i < componentCount; ++i )
    {        
        const auto edgeLoops = findLeftBoundary( res.modifiedMesh.topology, components[i] );        
        startSurfacePaths.insert( startSurfacePaths.end(), edgeLoops.begin(), edgeLoops.end() );
    }

    if ( !processZone( startSurfacePaths,
        res.commands.empty() ? Vector3f{} : Vector3f{ res.commands.back().x, res.commands.back().y, res.commands.back().z },
        params.cb ) )
        //subprogress( params.cb, 0.25f + 0.75f * float( i ) / componentCount, 0.25f + 0.75f * float( i + 1 ) / componentCount ) ) )
    {
        return unexpectedOperationCanceled();
    }
    /*std::vector<EdgeLoop> edgeLoops;
    if ( params.offsetMesh && params.offsetMesh->region )
    {
        edgeLoops = findLeftBoundary( params.offsetMesh->mesh.topology, params.offsetMesh->region );
    }
    else
    {
        const float upDistLimitSq = 2.0f * params.millRadius * params.millRadius;

        for ( const auto& component : MeshComponents::getAllComponents( mp ) )
        {
            FaceBitSet filteredSelection{ component };
            
            BitSetParallelFor( component, [&] ( FaceId f )
            {
                for ( const auto& p : mp.mesh.getTriPoints( f ) )
                {
                    const auto mpr = findProjection( p, res.modifiedMesh, upDistLimitSq );
                    if ( !mpr.mtp.e.valid() )
                    {
                        filteredSelection.set( f, false );
                        break;
                    }
                }
            } );

            auto loops = findLeftBoundary( mp.mesh.topology, filteredSelection );
            if ( !loops.empty() )
                edgeLoops.push_back( loops.front() );
        }
    }

    //otherwise process each selected zone separately
    const size_t loopCount = edgeLoops.size();
    for ( size_t i = 0; i < loopCount; ++i )
    {
        const auto& edgeLoop = edgeLoops[i];        

        Contour3f* contourBeforeProjecton = nullptr;
        Contour3f* contourBeforeCutMesh = nullptr;

        if ( ( !params.offsetMesh || !params.offsetMesh->region ) && params.contoursBeforeProjection )
        {
            params.contoursBeforeProjection->emplace_back();
            contourBeforeProjecton = &params.contoursBeforeProjection->back();
        }

        if ( params.contoursBeforeCutMesh )
        {
            params.contoursBeforeCutMesh->emplace_back();
            contourBeforeCutMesh = &params.contoursBeforeCutMesh->back();
        }

        std::vector<MeshTriPoint> meshTriPoints;
        for ( auto edgeId : edgeLoop )
        {
            if ( params.offsetMesh && params.offsetMesh->region )
            {
                MeshTriPoint mtp( MeshEdgePoint( edgeId, 0.0f ) );
                meshTriPoints.push_back( mtp );
                if ( contourBeforeCutMesh )
                    contourBeforeCutMesh->push_back( res.modifiedMesh.triPoint( mtp ) );

                continue;
            }

            const auto originalPoint = mp.mesh.edgePoint( MeshEdgePoint( edgeId, 0.0f ) );
            if ( contourBeforeProjecton )
                contourBeforeProjecton->push_back( originalPoint );

            // project all the vertices in the selected contour to the modified mesh
            auto mpr = res.modifiedMesh.projectPoint( originalPoint );
            if ( mpr )
            {
                meshTriPoints.push_back( mpr->mtp );
                if ( contourBeforeCutMesh )
                    contourBeforeCutMesh->push_back( res.modifiedMesh.triPoint( mpr->mtp ) );
            }

            mpr = res.modifiedMesh.projectPoint(mp.mesh.edgePoint(MeshEdgePoint(edgeId, 0.5f)));
            if ( mpr )
                meshTriPoints.push_back( mpr->mtp );
        }
        
        if ( contourBeforeProjecton )
            contourBeforeProjecton->push_back( contourBeforeProjecton->front() );
        
        if ( !processZone( nullptr, &meshTriPoints,
            res.commands.empty() ? Vector3f{} : Vector3f{ res.commands.back().x, res.commands.back().y, res.commands.back().z },
            subprogress( params.cb, 0.25f + 0.75f * float( i ) / loopCount, 0.25f + 0.75f * float( i + 1 ) / loopCount ) ) )
        {
            return unexpectedOperationCanceled();
        }
    }*/

    return res;
}

std::vector<GCommand> replaceLineSegmentsWithCircularArcs( const std::span<GCommand>& path, float eps, float maxRadius, Axis axis )
{
    if ( path.size() < 5 )
        return {};

    std::vector<GCommand> res;

    int startIdx = 0, endIdx = 0;
    Vector2f bestArcCenter, bestArcStart, bestArcEnd;
    bool CCWrotation = false;
    double bestArcR = 0;
    for ( int i = startIdx + 2; i < path.size(); ++i )
    {
        const GCommand& d2 = path[i];
        const int middleI = ( i + startIdx ) / 2;
        const GCommand& d1 = path[middleI];

        const Vector2f p0 = project( path[startIdx], axis );
        const Vector2f p1 = project( d1, axis );
        const Vector2f p2 = project( d2, axis );

        const Vector2f dif1 = p1 - p0;
        const Vector2f dif2 = p2 - p1;

        Vector2f pCenter;
        if ( dot( dif1, dif2 ) > 0
            && calcCircleCenter( p0, p1, p2, pCenter ) )
        {
            const double rArc = ( pCenter - p0 ).length();            
            const double r2Max = sqr( rArc + eps );
            const double r2Min = sqr( rArc - eps );

            const bool ccwRotation = cross( dif1, dif2 ) > 0;

            Vector2f dirStart = rotate90( p0 - pCenter );
            Vector2f dirEnd = rotateMinus90( p2 - pCenter );
            if ( ccwRotation )
            {
                dirStart = -dirStart;
                dirEnd = -dirEnd;
            }

            bool allInTolerance = true;
            Vector2f pPrev = p0;
            for ( int k = startIdx + 1; k <= i; ++k )
            {
                const Vector2f pk = project( path[k], axis );
                double r2k = ( pCenter - pk ).lengthSq();
                const Vector2f pkMiddle = ( pk + pPrev ) * 0.5f; 
                double r2kMiddle = ( pCenter - pkMiddle ).lengthSq();
                if ( r2k < r2Min || r2k > r2Max
                    || r2kMiddle < r2Min || r2kMiddle > r2Max )
                {
                    allInTolerance = false;
                    break;
                }
                bool insideArc = dot( dirStart, pk - p0 ) >= 0 && dot( dirEnd, pk - p2 ) >= 0;
                if ( !insideArc )
                {
                    allInTolerance = false;
                    break;
                }
                pPrev = pk;
            }
            if ( allInTolerance )
            {
                endIdx = i;
                bestArcCenter = pCenter;
                bestArcStart = p0;
                bestArcEnd = p2;
                CCWrotation = ccwRotation;
                if ( axis == Axis::Y )
                    CCWrotation = !CCWrotation;
                bestArcR = rArc;

                if ( i < path.size() - 1 )
                    continue;
            }
        }

        if ( endIdx - startIdx >= 3 && bestArcR < maxRadius )
        {
            const auto& d0a = path[startIdx];
            res.push_back( {} );
            auto& d1a = res.back();

            switch ( axis )
            {
            case MR::Axis::X:
                d1a.y = path[endIdx].y;
                d1a.z = path[endIdx].z;
                d1a.arcCenter.y = bestArcCenter.x - d0a.y;
                d1a.arcCenter.z = bestArcCenter.y - d0a.z;
                break;
            case MR::Axis::Y:
                d1a.x = path[endIdx].x;
                d1a.z = path[endIdx].z;
                d1a.arcCenter.x = bestArcCenter.x - d0a.x;
                d1a.arcCenter.z = bestArcCenter.y - d0a.z;
                break;
            case MR::Axis::Z:
                d1a.x = path[endIdx].x;
                d1a.y = path[endIdx].y;
                d1a.arcCenter.x = bestArcCenter.x - d0a.x;
                d1a.arcCenter.y = bestArcCenter.y - d0a.y;
                break;
            default:
                assert( false );
                break;
            }
            
            d1a.type = CCWrotation ? MoveType::ArcCCW : MoveType::ArcCW;

            startIdx = endIdx;
            i = startIdx + 1;
        }
        else
        {
            ++startIdx;
            endIdx = startIdx;
            res.push_back( path[startIdx] );
            i = startIdx + 1;
        }
    }

    for ( size_t i = endIdx; i < path.size(); ++i )
        res.push_back( path[i] );

    return res;
}

void interpolateArcs( std::vector<GCommand>& commands, const ArcInterpolationParams& params, Axis axis )
{
    const ArcPlane arcPlane = ( axis == Axis::X ) ? ArcPlane::YZ :
        ( axis == Axis::Y ) ? ArcPlane::XZ :
        ArcPlane::XY;

    commands.insert( commands.begin(), { .arcPlane = arcPlane } );
    size_t startIndex = 1u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( coord( commands[startIndex], axis ) ) ) )
            ++startIndex;

        if ( ++startIndex >= commands.size() )
            return;

        auto endIndex = startIndex + 1;
        while ( endIndex != commands.size() && std::isnan( coord( commands[endIndex], axis ) ) )
            ++endIndex;

        const size_t segmentSize = endIndex - startIndex;
        const auto interpolatedSegment = replaceLineSegmentsWithCircularArcs( std::span<GCommand>( &commands[startIndex], segmentSize ), params.eps, params.maxRadius, axis );
        if ( interpolatedSegment.empty() )
        {
            startIndex = endIndex;
            continue;
        }

        if ( interpolatedSegment.size() != segmentSize )
        {
            commands.erase( commands.begin() + startIndex + 1, commands.begin() + endIndex );
            commands.insert( commands.begin() + startIndex + 1, interpolatedSegment.begin(), interpolatedSegment.end() );
        }

        startIndex = startIndex + interpolatedSegment.size() + 1;
    }
}

std::shared_ptr<ObjectGcode> exportToolPathToGCode( const std::vector<GCommand>& commands )
{
    auto gcodeSource = std::make_shared<std::vector<std::string>>();

    for ( const auto& command : commands )
    {
        std::ostringstream gcode;
        gcode << "G";
        gcode << ( ( command.arcPlane != ArcPlane::None ) ? int( command.arcPlane ) : int( command.type ) );

        if ( !std::isnan( command.x ) )
            gcode << " X" << command.x;

        if ( !std::isnan( command.y ) )
            gcode << " Y" << command.y;

        if ( !std::isnan( command.z ) )
            gcode << " Z" << command.z;

        if ( !std::isnan( command.arcCenter.x ) )
            gcode << " I" << command.arcCenter.x;

        if ( !std::isnan( command.arcCenter.y ) )
            gcode << " J" << command.arcCenter.y;

        if ( !std::isnan( command.arcCenter.z ) )
            gcode << " K" << command.arcCenter.z;

        if ( !std::isnan( command.feed ) )
            gcode << " F" << command.feed;

        gcode << std::endl;
        gcodeSource->push_back( gcode.str() );
    }

    auto res = std::make_shared<ObjectGcode>();
    res->setGcodeSource( gcodeSource );
    res->setName( "Tool Path" );
    res->setLineWidth( 1.0f );
    return res;
}

float distSqrToLineSegment( const MR::Vector2f p, const MR::Vector2f& seg0, const MR::Vector2f& seg1 )
{
    const auto segDir = seg1 - seg0;
    const auto len2 = segDir.lengthSq();
    constexpr float epsSq = std::numeric_limits<float>::epsilon() * std::numeric_limits<float>::epsilon();
    if ( len2 <  epsSq )
    {
        return ( seg0 - p ).lengthSq();
    }

    const auto tmp = cross( p - seg0, segDir );
    return ( tmp * tmp ) / len2;
}


std::vector<GCommand> replaceStraightSegmentsWithOneLine( const std::span<GCommand>& path, float eps, float maxLength, Axis axis )
{
    if ( path.size() < 3 )
        return {};

    std::vector<GCommand> res;

    const float epsSq = eps * eps;
    const float maxLengthSq = maxLength * maxLength;
    int startIdx = 0, endIdx = 0;
    for ( int i = startIdx + 2; i < path.size(); ++i )
    {
        const auto& d0 = path[startIdx];
        const auto& d2 = path[i];

        const Vector2f p0 = project( d0, axis );
        const Vector2f p2 = project( d2, axis );

        if ( ( p0 - p2 ).lengthSq() < maxLengthSq ) // don't merge too long lines
        {
            bool allInTolerance = true;
            for ( int k = startIdx + 1; k < i; ++k )
            {
                const auto& dk = path[k];

                const Vector2f pk = project( dk, axis );
                const float dist2 = distSqrToLineSegment( pk, p0, p2 );
                if ( dist2 > epsSq )
                {
                    allInTolerance = false;
                    break;
                }
            }
            if ( allInTolerance )
            {
                endIdx = i;
                if ( i < path.size() - 1 ) // don't continue on last point, do interpolation
                    continue;
            }
        }

        res.push_back( path[endIdx] );
        startIdx = ( startIdx <= endIdx ) ? endIdx + 1 : endIdx;
        endIdx = startIdx;
        i = startIdx + 1;
    }

    for ( int i = startIdx; i < path.size(); ++i )
    {
        res.push_back( path[i] );
    }

    return res;
}

void interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis )
{
    size_t startIndex = 0u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( coord( commands[startIndex], axis ) ) ) )
            ++startIndex;

        if ( ++startIndex >= commands.size() )
            return;

        auto endIndex = startIndex + 1;
        while ( endIndex != commands.size() && std::isnan( coord( commands[endIndex], axis ) ) &&  commands[endIndex].type == MoveType::Linear )
            ++endIndex;

        const size_t segmentSize = endIndex - startIndex;
        const auto interpolatedSegment = replaceStraightSegmentsWithOneLine( std::span<GCommand>( &commands[startIndex], segmentSize ), params.eps, params.maxLength, axis );
        if ( interpolatedSegment.empty() )
        {
            startIndex = endIndex;
            continue;
        }

        if ( interpolatedSegment.size() != segmentSize )
        {
            commands.erase( commands.begin() + startIndex + 1, commands.begin() + endIndex );
            commands.insert( commands.begin() + startIndex + 1, interpolatedSegment.begin(), interpolatedSegment.end() );
        }

        startIndex = startIndex + interpolatedSegment.size() + 1;
    }
}

FaceBitSet smoothSelection( Mesh& mesh, const FaceBitSet& region, float expandOffset, float shrinkOffset )
{
    auto innerVerts = getIncidentVerts( mesh.topology, region );
    auto dists = computeSurfaceDistances( mesh, innerVerts );
    BitSetParallelFor( getInnerVerts( mesh.topology, region ), [&] ( VertId v )
    {
        dists[v] = -dists[v];
    } );

    auto isolines = extractIsolines( mesh.topology, dists, expandOffset );

    HashMap<VertId, float> startVerticesWithDists;
    const auto components = MeshComponents::getAllComponentsVertsSeparatedByPaths( mesh, isolines );
    VertBitSet extendedVerts;
    for ( const auto& component : components )
    {
        if ( ( component & innerVerts ).count() != 0 )
            extendedVerts |= component;
    }

    for ( const auto& isoline : isolines )
    {
        Polyline3 polyline;
        polyline.addFromSurfacePath( mesh, isoline );

        for ( const auto& ep : isoline )
        {
            const VertId org = mesh.topology.org( ep.e );
            const VertId dest = mesh.topology.dest( ep.e );

            if ( ep.inVertex() )
            {
                startVerticesWithDists.insert_or_assign( org, 0.0f );
                continue;
            }

            auto proj = findProjectionOnPolyline( mesh.points[org], polyline );
            startVerticesWithDists.insert_or_assign( org, sqrt( proj.distSq ) );

            proj = findProjectionOnPolyline( mesh.points[dest], polyline );
            startVerticesWithDists.insert_or_assign( dest, sqrt( proj.distSq ) );
        }
    }

    dists = computeSurfaceDistances( mesh, startVerticesWithDists );

    BitSetParallelFor( extendedVerts, [&] ( VertId v )
    {
        dists[v] = -dists[v];
    } );

    isolines = extractIsolines( mesh.topology, dists, shrinkOffset );
       
    FaceBitSet res;

    const auto origMesh = mesh;

    for ( const auto& isoline : isolines )
    {
        std::vector<MeshTriPoint> meshTriPoints;
        meshTriPoints.reserve( isoline.size() );
        for ( const auto& ep : isoline )
        {
            const auto mpr = mesh.projectPoint( origMesh.edgePoint( ep ) );
            meshTriPoints.emplace_back( mpr->mtp );
        }

        const auto meshContour = convertMeshTriPointsToClosedContour( mesh, meshTriPoints );

        CutMeshParameters cutMeshParams;
        cutMeshParams.forceFillMode = CutMeshParameters::ForceFill::All;
        FaceMap new2OldMap;
        cutMeshParams.new2OldMap = &new2OldMap;
           
        Vector<std::vector<FaceId>, FaceId> old2NewMap( mesh.topology.faceSize() );
        const auto cutRes = cutMesh( mesh, { meshContour }, cutMeshParams );

        for ( FaceId f = FaceId( 0 ); f < new2OldMap.size(); ++f )
        {
            if ( new2OldMap[f].valid() )
                old2NewMap[new2OldMap[f]].push_back( f );
        }

        const auto oldRegion = res;
        res = fillContourLeftByGraphCut( mesh.topology, cutRes.resultCut, edgeCurvMetric( mesh ) );

        BitSetParallelFor( oldRegion, [&] ( FaceId f )
        {
            for ( auto& newFace : old2NewMap[f] )
            {
                res.set( newFace, true );
            }
        } );
    }

    return res;
}

}
#endif
