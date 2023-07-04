#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )

#include "MRToolPath.h"
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

Mesh preprocessMesh( const Mesh& inputMesh, const ToolPathParams& params, const AffineXf3f* xf )
{
    OffsetParameters offsetParams;
    offsetParams.voxelSize = params.voxelSize;    
    const Vector3f normal = Vector3f::plusZ();

    Mesh meshCopy = *offsetMesh( inputMesh, params.millRadius, offsetParams );
    if ( xf )
        meshCopy.transform( *xf );
    
    FixUndercuts::fixUndercuts( meshCopy, normal, params.voxelSize );
    return meshCopy;
}

void addSurfacePath( std::vector<GCommand>& gcode, const Mesh& mesh, const MeshEdgePoint& start, const MeshEdgePoint& end )
{
    const auto sp = computeSurfacePath( mesh, start, end );
    if ( !sp.has_value() || sp->empty() )
        return;

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

std::vector<PlaneSections> extractAllSections( const Mesh& mesh, const MeshPart& origMeshPart, Axis axis, float sectionStep, int steps, ProgressCallback cb )
{
    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };

    std::vector<PlaneSections> sections( steps );

    const int axisIndex = int( axis );
    constexpr Vector3f normals[3] = { {1, 0, 0}, {0, 1, 0}, {0, 0 ,1} };
    const Vector3f normal = normals[axisIndex];
    const auto box = mesh.computeBoundingBox();
    const auto origBox = origMeshPart.mesh.computeBoundingBox();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );    

    tbb::parallel_for( tbb::blocked_range<int>( 0, steps ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int step = range.begin(); step < range.end(); ++step )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            const float currentCoord = plane.d - sectionStep * step;

            auto stepSections = extractPlaneSections( mesh, Plane3f{ plane.n, currentCoord } );
            if ( !origMeshPart.region )
            {
                sections[step] = std::move( stepSections );
                continue;
            }

            for ( const auto section : stepSections )
            {
                auto startIt = section.begin();
                auto endIt = startIt;

                for ( auto it = section.begin(); it < section.end(); ++it )
                {
                    Vector3f rayStart = mesh.edgePoint( *it );
                    rayStart[axisIndex] = box.max[axisIndex];

                    auto intersection = rayMeshIntersect( origMeshPart.mesh, Line3f{ rayStart, -normal } );

                    if ( intersection )
                    {
                        const auto faceId = origMeshPart.mesh.topology.left( intersection->mtp.e );
                        if ( origMeshPart.region->test( faceId ) )
                        {
                            ++endIt;
                            continue;
                        }
                    }

                    if ( startIt < endIt )
                    {
                        sections[step].push_back( SurfacePath{ startIt, endIt } );
                    }

                    startIt = it + 1;
                    endIt = startIt;
                }

                if ( startIt < section.end() )
                {
                    sections[step].push_back( SurfacePath{ startIt, section.end() } );
                }
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

    if ( !keepGoing.load( std::memory_order_relaxed ) || ( cb && !cb( 1.0f ) ) )
        return {};

    return sections;
}

std::vector<IsoLines> extractAllIsolines( const Mesh& mesh, VertId startPoint, float sectionStep, ProgressCallback cb )
{
    const MeshTriPoint mtp( mesh.topology, startPoint );

    const auto distances = computeSurfaceDistances( mesh, mtp );
    const auto [min, max] = parallelMinMax( distances.vec_ );

    const size_t numIsolines = size_t( ( max - min ) / sectionStep ) - 1;

    const auto& topology = mesh.topology;
    std::vector<IsoLines> isoLines( numIsolines );

    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    std::atomic<size_t> numDone{ 0 };

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, isoLines.size() ),
                       [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                break;

            isoLines[i] = extractIsolines( topology, distances, sectionStep * ( i + 1 ) );
        }

        if ( cb )
            numDone += range.size();

        if ( cb && std::this_thread::get_id() == mainThreadId )
        {
            if ( !cb( float( numDone ) / float( numIsolines ) ) )
                keepGoing.store( false, std::memory_order_relaxed );
        }
    } );

    if ( !keepGoing.load( std::memory_order_relaxed ) || ( cb && !cb( 1.0f ) ) )
        return {};

    return isoLines;
}

Expected<ToolPathResult, std::string> lacingToolPath( const MeshPart& mp, const ToolPathParams& params, Axis cutDirection, const AffineXf3f* xf, ProgressCallback cb )
{
    if ( cutDirection != Axis::X && cutDirection != Axis::Y )
        return unexpected( "Lacing can be done along the X or Y axis" );

    const auto cutDirectionIdx = int( cutDirection );
    const auto sideDirection = ( cutDirection == Axis::X ) ? Axis::Y : Axis::X;
    const auto sideDirectionIdx = int( sideDirection );

    ToolPathResult  res{ .modifiedMesh = preprocessMesh( mp.mesh, params, xf ) };
    const auto& mesh = res.modifiedMesh;

    const auto box = mesh.getBoundingBox();
    const float safeZ = std::max( box.max.z + params.millRadius, params.safeZ );

    const Vector3f normal = (cutDirection == Axis::X) ? Vector3f::plusX() : Vector3f::plusY();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min[cutDirectionIdx] ) / params.sectionStep ) );

    MeshEdgePoint lastEdgePoint = {};

    const auto allSections = extractAllSections( mesh, mp.mesh, cutDirection, params.sectionStep, steps, subprogress( cb, 0, 0.5f ) );
    if ( allSections.empty() )
        return unexpectedOperationCanceled();
    const auto sbp = subprogress( cb, 0.5f, 1.0f );

    using Intervals = std::vector< std::pair< std::vector<Vector3f>::const_iterator, std::vector<Vector3f>::const_iterator > >;

    const auto getIntervals = [&] (const std::vector<Vector3f>::const_iterator startIt, const std::vector<Vector3f>::const_iterator endIt, const std::vector<Vector3f>::const_iterator beginVec, const std::vector<Vector3f>::const_iterator endVec, bool moveForward )
    {
        Intervals res;

        auto startInterval = moveForward ? startIt : endIt;
        auto endInterval = startInterval;

        const auto processPoint = [&] ( std::vector<Vector3f>::const_iterator it )
        {
            const auto mpr = mp.mesh.projectPoint( *it );
            const auto faceId = mp.mesh.topology.left( mpr->mtp.e );

            if ( !mp.region || ( mpr && mp.region->test( faceId ) ) )
            {
                moveForward ? ++endInterval : --endInterval;
                return;
            }

            if ( startInterval != endInterval )
                res.emplace_back( startInterval, endInterval );

            if ( moveForward )
                startInterval = endInterval = it + 1;
            else
                startInterval = endInterval = it - 1;
        };

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
    };

    float lastFeed = 0;

    const auto transitOverSafeZ = [&] ( const std::vector<Vector3f>::const_iterator it )
    {
        float currentZ = res.commands.back().z;

        if ( safeZ - currentZ > params.retractLength )
        {
            const float zRetract = currentZ + params.retractLength;
            res.commands.push_back( { .feed = params.retractFeed, .z = zRetract } );
            res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
        }
        else
        {
            res.commands.push_back( { .feed = params.retractFeed, .z = safeZ } );
        }

        res.commands.push_back( { .type = MoveType::FastLinear, .x = it->x, .y = it->y } );

        if ( safeZ - it->z > params.plungeLength )
        {
            const float zPlunge = it->z + params.plungeLength;
            res.commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
        }
        res.commands.push_back( { .feed = params.plungeFeed, .x = it->x, .y = it->y, .z = it->z } );
        lastFeed = params.plungeFeed;
    };

    Vector3f lastPoint;    
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
        if ( sbp && !sbp( float( step ) / steps ) )
            return unexpectedOperationCanceled();

        const auto sections = allSections[step];
        if ( sections.empty() )
            continue;

        for ( const auto& section : sections )
        {
            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            const auto contours = polyline.contours();            
            const auto contour = contours.front();

            if ( contour.size() < 3 )
                continue;

            auto bottomLeftIt = contour.end();
            auto bottomRightIt = contour.end();

            for ( auto it = contour.begin(); it < contour.end(); ++it )
            {
                if ( bottomLeftIt == contour.end() || ( *it )[sideDirectionIdx] < ( *bottomLeftIt )[sideDirectionIdx] || ( ( *it )[sideDirectionIdx] == ( *bottomLeftIt )[sideDirectionIdx] && it->z < bottomLeftIt->z ) )
                    bottomLeftIt = it;

                if ( bottomRightIt == contour.end() || ( *it )[sideDirectionIdx] > ( *bottomRightIt )[sideDirectionIdx] || ( ( *it )[sideDirectionIdx] == ( *bottomRightIt )[sideDirectionIdx] && it->z < bottomRightIt->z ) )
                    bottomRightIt = it;
            }

            if ( cutDirection == Axis::Y )
                std::swap( bottomLeftIt, bottomRightIt );

            const bool moveForward = step & 1;
            const auto bottomLeftIdx = bottomLeftIt - contour.begin();
            const auto bottomRightIdx = bottomRightIt - contour.begin();
            const auto intervals = getIntervals( bottomLeftIt, bottomRightIt, contour.begin(), contour.end(), moveForward );
            if ( intervals.empty() )
                continue;

            if ( res.commands.empty() )
            {
                res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                res.commands.push_back( { .type = MoveType::FastLinear, .x = intervals[0].first->x, .y = intervals[0].first->y } );
            }
            else
            {
                const auto nextEdgePoint = section[intervals[0].first - contour.begin()];
                const auto distSq = ( mesh.edgePoint( lastEdgePoint ) - mesh.edgePoint( nextEdgePoint ) ).lengthSq();

                if ( distSq > critDistSq )
                    transitOverSafeZ( intervals[0].first );
                else
                    addSurfacePath( res.commands, mesh, lastEdgePoint, nextEdgePoint );
            }

            for ( size_t i = 0; i < intervals.size() - 1; ++i )
            {
                const auto& interval = intervals[i];

                for ( auto it = interval.first; it < interval.second; ++it )
                    addPoint( *it );

                if ( *intervals[i + 1].first != lastPoint )
                    transitOverSafeZ( intervals[i + 1].first );
            }

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

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}


Expected<ToolPathResult, std::string>  constantZToolPath( const MeshPart& mp, const ToolPathParams& params, const AffineXf3f* xf, ProgressCallback cb )
{
    ToolPathResult  res{ .modifiedMesh = preprocessMesh( mp.mesh, params, xf ) };
    const auto& mesh = res.modifiedMesh;

    const auto box = mp.mesh.getBoundingBox();
    const float safeZ = std::max( box.max.z + params.millRadius, params.safeZ );

    const Vector3f normal = Vector3f::plusZ();
    const auto plane = MR::Plane3f::fromDirAndPt( normal, box.max );
    const int steps = int( std::floor( ( plane.d - box.min.z ) / params.sectionStep ) );

    float currentZ = safeZ;
    res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );

    MeshEdgePoint prevEdgePoint;

    const float critTransitionLengthSq = params.critTransitionLength * params.critTransitionLength;
    bool needToRestoreBaseFeed = true;

    std::vector<PlaneSections> sections = extractAllSections( mesh, mp, Axis::Z, params.sectionStep, steps, subprogress( cb, 0, 0.5f ) );
    if ( sections.empty() )
        return unexpectedOperationCanceled();

    const auto sbp = subprogress( cb, 0.5f, 1.0f );

    for ( int step = 0; step < steps; ++step )
    {
        if ( sbp && !sbp( float( step ) / steps ) )
            return unexpectedOperationCanceled();

        auto& commands = res.commands;

        for ( const auto& section : sections[step] )
        {   
            if ( section.size() < 2 )
                continue;

            Polyline3 polyline;
            polyline.addFromSurfacePath( mesh, section );
            const auto contours = polyline.contours().front();

            auto nearestPointIt = section.begin();
            auto nextEdgePointIt = nearestPointIt;
            float minDistSq = FLT_MAX;            

            if ( prevEdgePoint.e.valid() && !mp.region )
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
                do
                {
                    std::next( nextEdgePointIt ) != section.end() ? ++nextEdgePointIt : nextEdgePointIt = section.begin();
                } while ( nextEdgePointIt != nearestPointIt && ( mesh.edgePoint( *nextEdgePointIt ) - nearestPoint ).lengthSq() < sectionStepSq );
            }

            const auto pivotIt = contours.begin() + std::distance( section.begin(), nextEdgePointIt );

            if ( !prevEdgePoint.e.valid() || minDistSq > critTransitionLengthSq )
            {
                if ( currentZ < safeZ )
                {
                    if ( safeZ - currentZ > params.retractLength )
                    {
                        const float zRetract = currentZ + params.retractLength;
                        commands.push_back( { .feed = params.retractFeed, .z = zRetract } );
                        commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
                    }
                    else
                    {
                        commands.push_back( { .feed = params.retractFeed, .z = safeZ } );
                    }
                }

                commands.push_back( { .type = MoveType::FastLinear, .x = pivotIt->x, .y = pivotIt->y } );

                if ( safeZ - pivotIt->z > params.plungeLength )
                {
                    const float zPlunge = pivotIt->z + params.plungeLength;
                    commands.push_back( { .type = MoveType::FastLinear, .z = zPlunge } );
                }
                commands.push_back( { .feed = params.plungeFeed, .x = pivotIt->x, .y = pivotIt->y, .z = pivotIt->z } );
                needToRestoreBaseFeed = true;
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

            currentZ = pivotIt->z;
            auto startIt = pivotIt + 1;
            if ( needToRestoreBaseFeed )
            {
                commands.push_back( { .feed = params.baseFeed, .x = startIt->x, .y = startIt->y } );
                ++startIt;
            }

            for ( auto it = startIt; it < contours.end(); ++it )
            {
                commands.push_back( { .x = it->x, .y = it->y } );
            }

            for ( auto it = contours.begin() + 1; it < pivotIt + 1; ++it )
            {
                commands.push_back( { .x = it->x, .y = it->y } );
            }

            needToRestoreBaseFeed = false;
            prevEdgePoint = *nextEdgePointIt;
        }
    }

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();

    return res;
}


Expected<ToolPathResult, std::string> constantCuspToolPath( const Mesh& inputMesh, const ToolPathParams& params, VertId startPoint, const AffineXf3f* xf, ProgressCallback cb )
{
    ToolPathResult  res{ .modifiedMesh = preprocessMesh( inputMesh, params, xf ) };
    
    const auto& mesh = res.modifiedMesh;
    const auto box = mesh.getBoundingBox();
    const float safeZ = box.max.z + params.millRadius;
    
    if ( !startPoint.valid() )
        startPoint = findDirMax( Vector3f::plusZ(), mesh );

    std::vector<IsoLines> isoLines = extractAllIsolines( mesh, startPoint, params.sectionStep, subprogress(cb, 0, 0.4f ) );

    if ( isoLines.empty() )
        return unexpectedOperationCanceled();

    res.commands.push_back( { .type = MoveType::FastLinear, .z = safeZ } );
    Vector3f lastPoint{ 0.0f, 0.0f, safeZ };

    MeshEdgePoint prevEdgePoint;

    std::optional<Vector3f> startUndercut;

    const auto addPointToTheToolPath = [&] ( const Vector3f& p )
    {
        if ( p == lastPoint )
            return;
      
        res.commands.push_back( { .x = p.x, .y = p.y, .z = p.z } );
        lastPoint = p;
    };

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

    const float minZ = box.min.z + params.sectionStep;

    VertBitSet noUndercutVertices( mesh.points.size() );
    tbb::parallel_for( tbb::blocked_range<VertId>( VertId{ 0 }, VertId{ noUndercutVertices.size() } ),
                      [&] ( const tbb::blocked_range<VertId>& range )
    {
        for ( VertId i = range.begin(); i < range.end(); ++i )
        {
            noUndercutVertices.set( i, mesh.points[i].z >= minZ );
        }
    } );

    const Vector3f normal = Vector3f::plusZ();
    const auto undercutPlane = MR::Plane3f::fromDirAndPt( normal, { 0.0f, 0.0f, minZ } );
    const auto undercutSections = extractPlaneSections( mesh, undercutPlane );
    Polyline3 undercutPolyline;
    undercutPolyline.addFromSurfacePath( mesh, undercutSections[0] );
    const auto undercutContour = undercutPolyline.contours().front();

    const auto addSliceToTheToolPath = [&] ( const Contour3f::const_iterator startIt, Contour3f::const_iterator endIt )
    {
        auto it = startIt;
        while ( it < endIt )
        {
            if ( it->z >= minZ )
            {
                addPointToTheToolPath( *it++ );
                continue;
            }

            if ( !startUndercut )
                startUndercut = *it;

            while ( it < endIt && it->z < minZ )
                ++it;

            if ( it < endIt )
            {
                Vector3f endUndercut = *it;
                const auto sectionStartIt = findNearestPoint( undercutContour, *startUndercut );
                const auto sectionEndIt = findNearestPoint( undercutContour, endUndercut );
                startUndercut.reset();

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

                    addPointToTheToolPath( *it++ );
                }
            }
        }
    };

    if ( cb && !cb( 0.5f ) )
        return unexpectedOperationCanceled();

    const auto sbp = subprogress( cb, 0.5f, 1.0f );
    const size_t numIsolines = isoLines.size();

    for ( size_t i = 0; i < numIsolines; ++i )
    {
        if ( sbp && !sbp( float( i ) / numIsolines ) )
            return unexpectedOperationCanceled();

        if ( isoLines[i].empty() )
            continue;

        Polyline3 polyline;
        const auto& surfacePath = isoLines[i][0];
        polyline.addFromSurfacePath( mesh, surfacePath );
        const auto contour = polyline.contours().front();

        auto nearestPointIt = surfacePath.begin();
        float minDistSq = FLT_MAX;

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

        Vector3f tmp;

        do
        {
            std::next( nextEdgePointIt ) != surfacePath.end() ? ++nextEdgePointIt : nextEdgePointIt = surfacePath.begin();
            tmp = mesh.edgePoint( *nextEdgePointIt );
        } while ( nextEdgePointIt != nearestPointIt && ( ( ( tmp - nearestPoint ).lengthSq() < sectionStepSq ) || ( tmp.z < minZ ) ) );

        const auto pivotIt = contour.begin() + std::distance( surfacePath.begin(), nextEdgePointIt );
        if ( pivotIt->z < minZ )
            continue;

        if ( i == 0 )
        {
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

    if ( cb && !cb( 1.0f ) )
        return unexpectedOperationCanceled();

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
            res.push_back( path[startIdx] );
            i = startIdx + 1;
        }
    }

    for ( size_t i = endIdx + 1; i < path.size(); ++i )
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

        if ( startIndex == commands.size() )
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

    return res;
}

void interpolateLines( std::vector<GCommand>& commands, const LineInterpolationParams& params, Axis axis )
{
    size_t startIndex = 0u;

    while ( startIndex < commands.size() )
    {
        while ( startIndex != commands.size() && ( commands[startIndex].type != MoveType::Linear || std::isnan( coord( commands[startIndex], axis ) ) ) )
            ++startIndex;

        if ( startIndex == commands.size() )
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

}
#endif
