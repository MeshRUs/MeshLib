#include "MRSurfacePath.h"
#include "MRBitSetParallelFor.h"
#include "MREdgePaths.h"
#include "MRExtractIsolines.h"
#include "MRGTest.h"
#include "MRLaplacian.h"
#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRMeshComponents.h"
#include "MRMeshLoad.h"
#include "MRMeshPart.h"
#include "MRPlanarPath.h"
#include "MRRegionBoundary.h"
#include "MRRingIterator.h"
#include "MRSurfaceDistance.h"
#include "MRTimer.h"

namespace MR
{

// consider triangle 0bc, where a linear scalar field is defined in all vertices: v(0) = 0, v(b) = vb, v(c) = vc;
// computes field gradient in the triangle
static Vector3d computeGradient( const Vector3d & b, const Vector3d & c, double vb, double vc )
{
    const auto bb = dot( b, b );
    const auto bc = dot( b, c );
    const auto cc = dot( c, c );
    const auto det = bb * cc - bc * bc;
    if ( det <= 0 )
    {
        // degenerate triangle
        return {};
    }
    const auto kb = ( 1 / det ) * ( cc * vb - bc * vc );
    const auto kc = ( 1 / det ) * (-bc * vb + bb * vc );
    return kb * b + kc * c;
}

static Vector3f computeGradient( const Vector3f & b, const Vector3f & c, float vb, float vc )
{
    return Vector3f{ computeGradient( Vector3d( b ), Vector3d( c ), double( vb ), double( vc ) ) };
}

/// given triangle with scalar field increasing in the direction \param dir;
/// returns true if the field increases inside the triangle from the edge 01
static bool dirEnters01( const Triangle3f & t, const Vector3f & dir )
{
    auto u01 = ( t[1] - t[0] ).normalized();
    // dir part orthogonal to the edge 01
    auto ortDir = dir - dot( dir, u01 ) * u01;
    return dot( ortDir, t[2] - t[0] ) > 0;
}

/// computes the intersection between
/// 1) the infinite line passing through the origin with direction +-unitDir;
/// 2) the infinite line containing the segment bc, returning in \param a the intersection position on that line.
/// \return false if the segment bc is parallel to unitDir
static bool computeLineLineCross( const Vector3f & b, const Vector3f & c, const Vector3f & unitDir, float & a )
{
    const auto d = c - b;
    // gort is a vector in the triangle plane orthogonal to grad
    const auto gort = d - dot( d, unitDir ) * unitDir;
    //const auto god = dot( d, d ) - sqr( dot( d, unitDir ) );
    const auto god = dot( gort, d );
    if ( god <= 0 )
        return false; // segment bc is parallel to unitDir
    const auto gob = -dot( gort, b );
    a = gob / god;
    return true;
}

/// given triangle with scalar field increasing in the direction \param unitDir;
/// returns true if the field increases inside the triangle from the edge 01
/// computes the position on this edge crossed by the line passing via point \param p and directed along \param unitDir
static bool computeEnter01Cross( const Triangle3f & t, const Vector3f & unitDir, const Vector3f & p, float & a )
{
    if ( !dirEnters01( t, unitDir ) )
        return false;
    return computeLineLineCross( t[0] - p, t[1] - p, unitDir, a );
}

// consider triangle 0bc, where gradient is given;
// computes the intersection of the ray (org=0, dir=-grad) with the open segment (b,c)
static std::optional<float> computeExitPos( const Vector3f & b, const Vector3f & c, const Vector3f & grad )
{
    const auto gradSq = grad.lengthSq();
    if ( gradSq <= 0 )
        return {};
    const auto d = c - b;
    // gort is a vector in the triangle plane orthogonal to grad
    const auto gort = d - ( dot( d, grad ) / gradSq ) * grad;
    const auto god = dot( gort, d );
    if ( god <= 0 )
        return {};
    const auto gob = -dot( gort, b );
    if ( gob <= 0 || gob >= god )
        return {};
    const auto a = gob / god;
    assert( a < FLT_MAX );
    const auto ip = a * c + ( 1 - a ) * b;
    if ( dot( grad, ip ) >= 0 )
        return {}; // (b,c) is intersected in the direction +grad
    return a;
}

MeshEdgePoint findSteepestDescentPoint( const MeshPart & mp, const VertScalars & field, VertId v )
{
    assert( mp.mesh.topology.isInnerOrBdVertex( v, mp.region ) );

    MeshEdgePoint res;
    float maxGradSq = 0;
    const auto vv = field[v];
    const auto pv = mp.mesh.points[v];
    for ( EdgeId e : orgRing( mp.mesh.topology, v ) )
    {
        if ( mp.region && !mp.mesh.topology.isInnerOrBdEdge( e, mp.region ) )
            continue;
        const auto d = mp.mesh.topology.dest( e );
        const auto pd = mp.mesh.points[d] - pv;
        const auto pdSq = pd.lengthSq();
        if ( field[d] == FLT_MAX )
            continue;
        const auto vd = field[d] - vv;
        if ( vd < 0 )
        {
            if ( pdSq == 0 && maxGradSq == 0 && !res ) // degenerate edge
                res = MeshEdgePoint{ e.sym(), 0 };
            else if ( pdSq > 0 )
            {
                auto edgeGradSq = sqr( vd ) / pdSq;
                if ( edgeGradSq > maxGradSq )
                {
                    maxGradSq = edgeGradSq;
                    res = MeshEdgePoint{ e.sym(), 0 };
                }
            }
        }
        if ( auto f = mp.mesh.topology.left( e ); contains( mp.region, f ) )
        {
            const auto eBd = mp.mesh.topology.prev( e.sym() );
            const auto x = mp.mesh.topology.dest( eBd );
            const auto px = mp.mesh.points[x] - pv;
            if ( auto fx = field[x]; fx < FLT_MAX )
            {
                const auto vx = fx - vv;
                const auto triGrad = computeGradient( pd, px, vd, vx );
                const auto triGradSq = triGrad.lengthSq();
                if ( triGradSq > maxGradSq )
                {
                    if ( auto a = computeExitPos( pd, px, triGrad ) )
                    {
                        maxGradSq = triGradSq;
                        res = MeshEdgePoint{ eBd, *a };
                    }
                }
            }
        }
    }
    return res;
}

MeshEdgePoint findSteepestDescentPoint( const MeshPart & mp, const VertScalars & field, const MeshEdgePoint & ep )
{
    if ( auto v = ep.inVertex( mp.mesh.topology ) )
        return findSteepestDescentPoint( mp, field, v );
    assert( mp.mesh.topology.isInnerOrBdEdge( ep.e, mp.region ) );

    // point is not in vertex
    const auto p = mp.mesh.edgePoint( ep );

    const auto o = mp.mesh.topology.org( ep.e );
    const auto d = mp.mesh.topology.dest( ep.e );
    const auto fo = field[o];
    const auto fd = field[d];
    const auto v = ( 1 - ep.a ) * fo + ep.a * fd;
    const auto po = mp.mesh.points[o];
    const auto pd = mp.mesh.points[d];

    MeshEdgePoint result;
    float maxGradSq = -FLT_MAX;
    if ( fo != fd )
    {
        // jump by default to edge's end with smaller value
        const auto odSq = ( po - pd ).lengthSq();
        maxGradSq = odSq > 0 ? sqr( fo - fd ) / odSq : FLT_MAX;
        result = fo < fd ? MeshEdgePoint{ ep.e, 0 } : MeshEdgePoint{ ep.e.sym(), 0 };
    }

    if ( auto f = mp.mesh.topology.left( ep.e ); contains( mp.region, f ) )
    {
        const auto el = mp.mesh.topology.next( ep.e );
        const auto l = mp.mesh.topology.dest( el );
        const auto fl = field[l];
        const auto pl = mp.mesh.points[l];
        const auto triGrad = computeGradient( pd - po, pl - po, fd - fo, fl - fo );
        const auto triGradSq = triGrad.lengthSq();
        bool moveL = true;
        if ( triGradSq > maxGradSq )
        {
            const auto unitDir = triGrad / std::sqrt( triGradSq );
            moveL = false;
            if ( !dirEnters01( { po, pd, pl }, unitDir ) ) //if the gradient exits start edge then lowest point must be on the edge
            {
                float a = -1;
                if ( computeEnter01Cross( { pd, pl, po }, unitDir, p, a ) && a >= 0 )
                {
                    if ( a <= 1 )
                    {
                        moveL = false;
                        result = MeshEdgePoint{ mp.mesh.topology.prev( ep.e.sym() ), a };
                        maxGradSq = triGradSq;
                    }
                    else
                        moveL = true;
                }
                if ( computeEnter01Cross( { pl, po, pd }, unitDir, p, a ) && a <= 1 )
                {
                    if ( a >= 0 )
                    {
                        moveL = false;
                        result = MeshEdgePoint{ el.sym(), a };
                        maxGradSq = triGradSq;
                    }
                    else
                        moveL = true;
                }
            }
        }
        if ( moveL && fl <= v )
        {
            const auto plSq = ( pl - p ).lengthSq();
            auto vertGradSq = plSq > 0 ? sqr( fl - v ) / plSq : FLT_MAX;
            if ( vertGradSq >= maxGradSq )
            {
                result = MeshEdgePoint{ el.sym(), 0 };
                maxGradSq = vertGradSq;
            }
        }
    }

    if ( auto f = mp.mesh.topology.right( ep.e ); contains( mp.region, f ) )
    {
        const auto er = mp.mesh.topology.prev( ep.e );
        const auto r = mp.mesh.topology.dest( er );
        const auto fr = field[r];
        const auto pr = mp.mesh.points[r];
        const auto triGrad = computeGradient( pr - po, pd - po, fr - fo, fd - fo );
        const auto triGradSq = triGrad.lengthSq();
        bool moveR = true;
        if ( triGradSq > maxGradSq )
        {
            const auto unitDir = triGrad / std::sqrt( triGradSq );
            moveR = false;
            if ( !dirEnters01( { pd, po, pr }, unitDir ) ) //if the gradient exits start edge then lowest point must be on the edge
            {
                float a = -1;
                if ( computeEnter01Cross( { pr, pd, po }, unitDir, p, a ) && a <= 1 )
                {
                    if ( a >= 0 )
                    {
                        moveR = false;
                        result = MeshEdgePoint{ mp.mesh.topology.next( ep.e.sym() ).sym(), a };
                        maxGradSq = triGradSq;
                    }
                    else
                        moveR = true;
                }
                if ( computeEnter01Cross( { po, pr, pd }, unitDir, p, a ) && a >= 0 )
                {
                    if ( a <= 1 )
                    {
                        moveR = false;
                        result = MeshEdgePoint{ er, a };
                        maxGradSq = triGradSq;
                    }
                    else
                        moveR = true;
                }
            }
        }
        if ( moveR && fr <= v )
        {
            const auto prSq = ( pr - p ).lengthSq();
            auto vertGradSq = prSq > 0 ? sqr( fr - v ) / prSq : FLT_MAX;
            if ( vertGradSq >= maxGradSq )
            {
                result = MeshEdgePoint{ er.sym(), 0 };
                maxGradSq = vertGradSq;
            }
        }
    }

    if ( !result )
    {
        // otherwise jump in the closest edge's end
        assert( maxGradSq == -FLT_MAX );
        assert( fo == fd );
        result = ep.a <= 0.5f ? MeshEdgePoint{ ep.e, 0 } : MeshEdgePoint{ ep.e.sym(), 0 };
    }

    return result;
}

MeshEdgePoint findSteepestDescentPoint( const MeshPart & mp, const VertScalars & field, const MeshTriPoint & tp )
{
    if ( auto ep = tp.onEdge( mp.mesh.topology ) )
        return findSteepestDescentPoint( mp, field, ep );
    assert( contains( mp.region, mp.mesh.topology.left( tp.e ) ) );

    // point is not on edge
    MeshEdgePoint res;
    const auto p = mp.mesh.triPoint( tp );

    VertId v[3];
    mp.mesh.topology.getLeftTriVerts( tp.e, v );

    Vector3f pv[3];
    float vv[3];
    EdgeId e[3];
    auto ei = tp.e;
    for ( int i = 0; i < 3; ++i )
    {
        pv[i] = mp.mesh.points[v[i]];
        vv[i] = field[v[i]];
        e[i] = ei;
        ei = mp.mesh.topology.prev( ei.sym() );
    }
    if ( vv[0] == vv[1] && vv[1] == vv[2] )
        return res; // the triangle is completely "flat"
    const auto f = tp.bary.interpolate( vv[0], vv[1], vv[2] );

    const auto triGrad = computeGradient( pv[1] - pv[0], pv[2] - pv[0], vv[1] - vv[0], vv[2] - vv[0] );
    const auto triGradSq = triGrad.lengthSq();
    if ( triGradSq > 0 )
    {
        // search for line path inside the triangle in minus gradient direction
        auto unitDir = triGrad / std::sqrt( triGradSq );
        float miss = FLT_MAX;
        for ( int i = 0; i < 3; ++i )
        {
            const Triangle3f t = { pv[i], pv[( i + 1 ) % 3], pv[( i + 2 ) % 3] };
            if ( !dirEnters01( t, unitDir ) )
                continue;
            float a = 0;
            if ( !computeLineLineCross( t[0] - p, t[1] - p, unitDir, a ) )
            {
                // we know that unitDir enters via the edge 01 of the triangle,
                // and unitDir is almost parallel to this edge, so select appropriate edge's end (which leads inside the triangle)
                if ( !res )
                    res = MeshEdgePoint{ e[i], dot( t[1] - t[0], unitDir ) >= 0 ? 0.0f : 1.0f };
                continue;
            }
            // how much do we miss the boundaries of the segment [rv0,rv1]
            const auto ca = std::clamp( a, 0.0f, 1.0f );
            const auto m = std::abs( a - ca ) * ( t[1] - t[0] ).length();
            if ( m < miss ) // minor misses due to rounding errors shall be tolerated
            {
                miss = m;
                res = MeshEdgePoint{ e[i], ca };
            }
        }
        if ( res )
            return res;
        // if triangle has not-zero gradient then res must be found above
        assert( false );
    }

    // no line path inside the triangle was found, try to jump in a vertex with smaller field value
    float maxGradSq = -FLT_MAX;
    for ( int i = 0; i < 3; ++i )
    {
        if ( vv[i] <= f )
        {
            const auto pvSq = ( pv[i] - p ).lengthSq();
            // if input point is close to a triangle vertex then pvSq can be zero;
            // in that case give that vertex a priority (FLT_MAX) over others
            auto vertGradSq = pvSq > 0 ? sqr( vv[i] - f ) / pvSq : FLT_MAX;
            if ( vertGradSq > maxGradSq )
            {
                maxGradSq = vertGradSq;
                res = MeshEdgePoint{ e[i], 0 };
            }
        }
    }

    return res;
}

Expected<SurfacePath, PathError> computeGeodesicPathApprox( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype )
{
    MR_TIMER;
    if ( atype == GeodesicPathApprox::FastMarching )
        return computeFastMarchingPath( mesh, start, end );

    SurfacePath res;
    if ( !fromSameTriangle( mesh.topology, MeshTriPoint{ start }, MeshTriPoint{ end } ) )
    {
        VertId v1, v2;
        EdgePath edgePath = ( atype == GeodesicPathApprox::DijkstraBiDir ) ?
            buildShortestPathBiDir( mesh, start, end, &v1, &v2 ) :
            buildShortestPathAStar( mesh, start, end, &v1, &v2 );
        if ( !v1 || !v2 )
            return unexpected( PathError::StartEndNotConnected );

        // remove last segment from the path if end-point and the origin of last segment belong to one triangle
        while( !edgePath.empty()
            && fromSameTriangle( mesh.topology, MeshTriPoint{ end }, MeshTriPoint{ MeshEdgePoint{ edgePath.back(), 0 } } ) )
        {
            v2 = mesh.topology.org( edgePath.back() );
            edgePath.pop_back();
        }

        // remove first segment from the path if start-point and the destination of first segment belong to one triangle
        while( !edgePath.empty()
            && fromSameTriangle( mesh.topology, MeshTriPoint{ start }, MeshTriPoint{ MeshEdgePoint{ edgePath.front(), 1 } } ) )
        {
            v1 = mesh.topology.dest( edgePath.front() );
            edgePath.erase( edgePath.begin() );
        }

        if ( edgePath.empty() )
        {
            assert ( v1 == v2 );
            res = { MeshEdgePoint( mesh.topology.edgeWithOrg( v1 ), 0.0f ) };
        }
        else
        {
            res.reserve( edgePath.size() + 1 );
            for ( EdgeId e : edgePath )
                res.emplace_back( e, 0.0f );
            res.emplace_back( edgePath.back(), 1.0f );
        }
    }
    return res;
}

SurfacePath computeSteepestDescentPath( const MeshPart & mp, const VertScalars & field,
    const MeshTriPoint & start, const ComputeSteepestDescentPathSettings & settings )
{
    SurfacePath res;
    computeSteepestDescentPath( mp, field, start, &res, settings );
    return res;
}

void computeSteepestDescentPath( const MeshPart & mp, const VertScalars & field,
    const MeshTriPoint & start, SurfacePath * outPath, const ComputeSteepestDescentPathSettings & settings )
{
    assert( start );
    assert( settings.outVertexReached || settings.outBdReached || outPath );
    size_t iniPathSize = outPath ? outPath->size() : 0;
    size_t edgesPassed = 0;
    auto curr = findSteepestDescentPoint( mp, field, start );
    while ( curr )
    {
        if ( settings.outVertexReached )
        {
            if ( auto v = curr.inVertex( mp.mesh.topology ) )
            {
                *settings.outVertexReached = v;
                return;
            }
        }
        if ( settings.outBdReached && curr.isBd( mp.mesh.topology ) )
        {
            *settings.outBdReached = curr;
            return;
        }
        ++edgesPassed;
        if ( outPath )
            outPath->push_back( curr );
        if ( settings.end && fromSameTriangle( mp.mesh.topology, MeshTriPoint( settings.end ), MeshTriPoint( curr ) ) )
            break; // reached triangle with end point
        if ( edgesPassed > mp.mesh.topology.numValidFaces() )
        {
            // normal path cannot visit any triangle more than once
            assert( false );
            if ( outPath )
                outPath->resize( iniPathSize );
            return;
        }
        curr = findSteepestDescentPoint( mp, field, curr );
    }
}

UndirectedEdgeBitSet findExtremeEdges( const Mesh & mesh, const VertScalars & field, ExtremeEdgeType type )
{
    MR_TIMER
    UndirectedEdgeBitSet res( mesh.topology.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        EdgeId e = ue;
        if ( !mesh.topology.left( e ) || !mesh.topology.right( e ) )
            return;

        const auto vo = mesh.topology.org( e );
        const auto vd = mesh.topology.dest( e );
        const auto vl = mesh.topology.dest( mesh.topology.next( e ) );

        const auto po = mesh.points[vo];
        const auto pd = mesh.points[vd];
        const auto pl = mesh.points[vl];

        const auto fo = field[vo];
        const auto fd = field[vd];
        const auto fl = field[vl];

        auto gradL = computeGradient( pd - po, pl - po, fd - fo, fl - fo );
        if ( type == ExtremeEdgeType::Gorge )
            gradL = -gradL;
        if ( dirEnters01( { po, pd, pl }, gradL ) )
            return;

        const auto vr = mesh.topology.dest( mesh.topology.prev( e ) );
        const auto pr = mesh.points[vr];
        const auto fr = field[vr];

        auto gradR = computeGradient( pr - po, pd - po, fr - fo, fd - fo );
        if ( type == ExtremeEdgeType::Gorge )
            gradR = -gradR;
        if ( dirEnters01( { pd, po, pr }, gradR ) )
            return;

        res.set( ue );
    } );
    return res;
}

Expected<SurfacePath, PathError> computeFastMarchingPath( const MeshPart & mp,
    const MeshTriPoint & start, const MeshTriPoint & end,
    const VertBitSet* vertRegion, VertScalars * outSurfaceDistances )
{
    MR_TIMER;
    SurfacePath res;
    auto s = start;
    auto e = end;
    if ( fromSameTriangle( mp.mesh.topology, s, e ) )
        return res; // path does not cross any edges

    // the region can be specified by faces or by vertices, but not in both ways at the same time
    assert( !mp.region || !vertRegion );

    VertBitSet myVertRegion;
    if ( mp.region )
    {
        myVertRegion = getIncidentVerts( mp.mesh.topology, *mp.region );
        vertRegion = &myVertRegion;
    }

    // build distances from end to start, so to get correct path in reverse order
    bool connected = false;
    auto distances = computeSurfaceDistances( mp.mesh, end, start, vertRegion, &connected );
    if ( !connected )
        return unexpected( PathError::StartEndNotConnected );

    res = computeSteepestDescentPath( mp.mesh, distances, start, { .end = end } );
    if ( res.empty() ) // no edge is crossed only if start and end are from the same triangle
        return unexpected( PathError::InternalError );

    if ( outSurfaceDistances )
        *outSurfaceDistances = std::move( distances );
    return res;
}

Expected<SurfacePath, PathError> computeSurfacePath( const MeshPart & mp,
    const MeshTriPoint & start, const MeshTriPoint & end, int maxGeodesicIters,
    const VertBitSet* vertRegion, VertScalars * outSurfaceDistances )
{
    MR_TIMER;
    auto res = computeFastMarchingPath( mp, start, end, vertRegion, outSurfaceDistances );
    if ( res.has_value() && !res.value().empty() )
        reducePath( mp.mesh, start, res.value(), end, maxGeodesicIters );
    return res;
}

Expected<SurfacePath, PathError> computeGeodesicPath( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & end, GeodesicPathApprox atype,
    int maxGeodesicIters )
{
    MR_TIMER;
    auto res = computeGeodesicPathApprox( mesh, start, end, atype );
    if ( res.has_value() && !res.value().empty() )
        reducePath( mesh, start, res.value(), end, maxGeodesicIters );
    return res;
}

HashMap<VertId, VertId> computeClosestSurfacePathTargets( const Mesh & mesh,
    const VertBitSet & starts, const VertBitSet & ends, 
    const VertBitSet * vertRegion, VertScalars * outSurfaceDistances )
{
    MR_TIMER;
    auto distances = computeSurfaceDistances( mesh, ends, starts, FLT_MAX, vertRegion );

    HashMap<VertId, VertId> res;
    res.reserve( starts.count() );
    // create all keys in res before parallel region
    for ( auto v : starts )
        res[v];

    BitSetParallelFor( starts, [&]( VertId v )
    {
        auto last = findSteepestDescentPoint( mesh, distances, v );
        // if ( !last ) then v is not reachable from (ends) or it is contained in (ends)
        int steps = 0;
        while ( last )
        {
            if ( ++steps > mesh.topology.numValidFaces() )
            {
                // internal error
                assert( false );
                last = {};
                break;
            }
            if ( auto next = findSteepestDescentPoint( mesh, distances, last ) )
                last = next;
            else
                break;
        }
        if ( last )
            res[v] = last.getClosestVertex( mesh.topology );
    } );

    if ( outSurfaceDistances )
        *outSurfaceDistances = std::move( distances );
    return res;
}

SurfacePaths getSurfacePathsViaVertices( const Mesh & mesh, const VertBitSet & vs )
{
    MR_TIMER
    SurfacePaths res;
    if ( vs.empty() )
        return res;

    VertScalars scalarField( mesh.topology.vertSize(), 0 );
    VertBitSet freeVerts;
    for ( const auto & cc : MeshComponents::getAllComponentsVerts( mesh ) )
    {
        auto freeCC = cc - vs;
        auto numfree = freeCC.count();
        if ( numfree <= 0 )
            continue; // too small connected component
        if ( numfree == cc.count() )
            continue; // no single fixed vertex in the component

        // fix one additional vertex in each connected component with the value 1
        // (to avoid constant 0 solution)
        VertId fixedV = *begin( freeCC );
        scalarField[fixedV] = 1;
        freeCC.reset( fixedV );
        freeVerts |= freeCC;
    }

    Laplacian lap( const_cast<Mesh&>( mesh ) ); //mesh will not be changed
    lap.init( freeVerts, EdgeWeights::Unit, Laplacian::RememberShape::No );
    lap.applyToScalar( scalarField );
    res = extractIsolines( mesh.topology, scalarField, 0 );

    return res;
}

float surfacePathLength( const Mesh& mesh, const SurfacePath& surfacePath )
{
    if ( surfacePath.empty() )
        return 0.0f;
    float sum = 0.0f;
    auto prevPoint = mesh.edgePoint( surfacePath[0] );
    for ( int i = 1; i < surfacePath.size(); ++i )
    {
        auto curPoint = mesh.edgePoint( surfacePath[i] );
        sum += ( curPoint - prevPoint ).length();
        prevPoint = curPoint;
    }
    return sum;
}

Contour3f surfacePathToContour3f( const Mesh & mesh, const SurfacePath & line )
{
    MR_TIMER;
    Contour3f res;
    res.reserve( line.size() );
    for ( const auto& s : line )
        res.push_back( mesh.edgePoint( s ) );

    return res;
}

Contours3f surfacePathsToContours3f( const Mesh & mesh, const SurfacePaths & lines )
{
    MR_TIMER;
    Contours3f res;
    res.reserve( lines.size() );
    for ( const auto& l : lines )
        res.push_back( surfacePathToContour3f( mesh, l ) );
    return res;
}

TEST(MRMesh, SurfacePath) 
{
    Vector3f g;

    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 0, 1, 0 } ).length(), 0, 1e-5f );

    std::optional<float> e;
    g = computeGradient( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1, 1 );
    EXPECT_NEAR( ( g - Vector3f{ 1, 1, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, g );
    EXPECT_FALSE( e.has_value() );
    e = computeExitPos ( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, -g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );

    g = computeGradient( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, g );
    EXPECT_NEAR( *e, 0.5f, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -1, 0 }, Vector3f{ 1, 1, 0 }, -g );
    EXPECT_FALSE( e.has_value() );

    g = computeGradient( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, g );
    EXPECT_NEAR( *e, 0.1f, 1e-5f );

    g = computeGradient( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, -0.9f, 0 }, Vector3f{ 1, 0.1f, 0 }, g );
    EXPECT_NEAR( *e, 0.9f, 1e-5f );

    g = computeGradient( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -1, -1 );
    EXPECT_NEAR( ( g - Vector3f{ -1, 0, 0 } ).length(), 0, 1e-5f );
    e = computeExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, g );
    EXPECT_FALSE( e.has_value() );
    e = computeExitPos ( Vector3f{ 1, 0.1f, 0 }, Vector3f{ 1, 0.9f, 0 }, -g );
    EXPECT_FALSE( e.has_value() );
}

TEST( MRMesh, SurfacePathTargets )
{
    Triangulation t{
        { 0_v, 1_v, 2_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );

    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // 0_v
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // 1_v
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // 2_v

    VertBitSet starts(3);
    starts.set( 1_v );
    starts.set( 2_v );

    VertBitSet ends(3);
    ends.set( 0_v );

    const auto map = computeClosestSurfacePathTargets( mesh, starts, ends );
    EXPECT_EQ( map.size(), starts.count() );
    for ( const auto & [start, end] : map )
    {
        EXPECT_TRUE( starts.test( start ) );
        EXPECT_TRUE( ends.test( end ) );
    }
}

} //namespace MR
