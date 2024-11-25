#include "MREdgePaths.h"
#include "MREdgePathsBuilder.h"
#include "MREdgeIterator.h"
#include "MRRegionBoundary.h"
#include "MRPlane3.h"
#include "MRTimer.h"
#include "MRCube.h"
#include "MRUnionFind.h"
#include "MRGTest.h"

namespace MR
{

bool isEdgePath( const MeshTopology & topology, const std::vector<EdgeId> & edges )
{
    for ( int i = 0; i + 1 < edges.size(); ++i )
        if ( topology.org( edges[i + 1] ) != topology.dest( edges[i] ) )
            return false;
    return true;
}

bool isEdgeLoop( const MeshTopology & topology, const std::vector<EdgeId> & edges )
{
    return isEdgePath( topology, edges ) && !edges.empty()
        && topology.org( edges.front() ) == topology.dest( edges.back() );
}

std::vector<EdgeLoop> splitOnSimpleLoops( const MeshTopology & topology, std::vector<EdgeLoop> && loops )
{
    MR_TIMER
    std::vector<EdgeLoop> res;
    res.reserve( loops.size() );
    HashMap<VertId, int> vmap; // vertex -> edge# in loop with origin in vertex
    for ( auto & loop : loops )
    {
        assert( isEdgeLoop( topology, loop ) );
        for (;;)
        {
            bool split = false;
            for ( int i = 0; i < loop.size(); ++i )
            {
                auto [it, inserted] = vmap.insert( { topology.org( loop[i] ), i } );
                if ( !inserted )
                {
                    const auto beg = loop.begin() + it->second;
                    const auto end = loop.begin() + i;
                    res.push_back( EdgeLoop{ beg, end } );
                    assert( isEdgeLoop( topology, res.back() ) );
                    loop.erase( beg, end );
                    assert( isEdgeLoop( topology, loop ) );
                    split = true;
                    break;
                }
            }
            vmap.clear();
            if ( split )
                continue;
            res.push_back( std::move( loop ) );
            break;
        }
    }

    return res;
}

void reverse( EdgePath & path )
{
    std::reverse( path.begin(), path.end() );
    for ( auto & e : path )
        e = e.sym();
}

void reverse( std::vector<EdgePath> & paths )
{
    for ( auto & path : paths )
        reverse( path );
}

double calcPathMetric( const EdgePath & path, EdgeMetric metric )
{
    double res = 0;
    for ( auto & e : path )
        res += metric( e );
    return res;
}

Vector3d calcOrientedArea( const EdgeLoop & loop, const Mesh & mesh )
{
    assert( isEdgeLoop( mesh.topology, loop ) );
    Vector3d dblArea;
    for ( auto e : loop )
        dblArea += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    return 0.5 * dblArea;
}

void sortPathsByMetric( std::vector<EdgePath> & paths, EdgeMetric metric )
{
    MR_TIMER
    const auto sz = paths.size();
    std::vector<int> sortedIds( sz );
    std::vector<double> lens( sz );
    for ( int i = 0; i < sz; ++i )
    {
        sortedIds[i] = i;
        lens[i] = calcPathMetric( paths[i], metric );
    }
    std::sort( sortedIds.begin(), sortedIds.end(), [&]( int a, int b )
        { return lens[a] < lens[b]; } );

    std::vector<EdgePath> sorted( sz );
    for ( int i = 0; i < sz; ++i )
    {
        sorted[i] = std::move( paths[sortedIds[i]] );
    }
    paths = std::move( sorted );
}

void addLeftBand( const MeshTopology & topology, const EdgeLoop & loop, FaceBitSet & addHere )
{
    if ( loop.empty() )
        return;
    assert( isEdgeLoop( topology, loop ) );

    EdgeId stop = loop.back().sym();
    for ( EdgeId e : loop )
    {
        for ( EdgeId ei : orgRing( topology, e ) )
        {
            if ( ei == stop )
                break;
            if ( auto l = topology.left( ei ) )
                addHere.autoResizeSet( l );
        }
        stop = e.sym();
    }
}

// internal version with pre-initialized builder
static EdgePath buildSmallestMetricPath( VertId start, EdgePathsBuilder& b, float maxPathMetric )
{
    for (;;)
    {
        auto vinfo = b.growOneEdge();
        if ( !vinfo.v )
        {
            // unable to find the path
            return {};
        }
        if ( vinfo.metric > maxPathMetric )
        {
            // unable to find the path within given metric limitation
            return {};
        }
        if ( vinfo.v == start )
            break;
    }
    return b.getPathBack( start );
}

EdgePath buildSmallestMetricPath( const MeshTopology& topology, const EdgeMetric& metric, VertId start, const VertBitSet& finish, float maxPathMetric /*= FLT_MAX */ )
{
    MR_TIMER

    EdgePathsBuilder b( topology, metric );
    for ( VertId v : finish )
        b.addStart( v, 0 );
    return buildSmallestMetricPath( start, b, maxPathMetric );
}

EdgePath buildSmallestMetricPath(
    const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric )
{
    MR_TIMER

    EdgePathsBuilder b( topology, metric );
    b.addStart( finish, 0 );
    return buildSmallestMetricPath( start, b, maxPathMetric );
}

EdgePath buildSmallestMetricPathBiDir(
    const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric )
{
    TerminalVertex s{ start, 0 };
    TerminalVertex f{ finish, 0 };
    return buildSmallestMetricPathBiDir( topology, metric, &s, 1, &f, 1, nullptr, nullptr, maxPathMetric );
}

EdgePath buildSmallestMetricPathBiDir( const MeshTopology & topology, const EdgeMetric & metric,
    const TerminalVertex * starts, int numStarts,
    const TerminalVertex * finishes, int numFinishes,
    VertId * outPathStart, VertId * outPathFinish, float maxPathMetric )
{
    MR_TIMER
    assert( numStarts > 0 && numFinishes > 0 );

    VertId join;
    float joinPathMetric = maxPathMetric;

    EdgePathsBuilder bs( topology, metric );
    for ( int si = 0; si < numStarts; ++si )
        bs.addStart( starts[si].v, starts[si].metric );

    EdgePathsBuilder bf( topology, metric );
    for ( int fi = 0; fi < numFinishes; ++fi )
        bf.addStart( finishes[fi].v, finishes[fi].metric );

    bool keepGrowing = true;
    for (;;)
    {
        auto ds = bs.doneDistance();
        auto df = bf.doneDistance();
        if ( keepGrowing && join && joinPathMetric <= ds + df )
        {
            keepGrowing = false;
        }
        if ( ds <= df )
        {
            if ( ds >= FLT_MAX )
                break;
            auto c = bs.reachNext();
            if ( !c.v )
                continue;
            if ( keepGrowing )
                bs.addOrgRingSteps( c );
            if ( auto info = bf.getVertInfo( c.v ) )
            {
                auto newMetric = c.metric + info->metric;
                if ( newMetric < joinPathMetric )
                {
                    join = c.v;
                    joinPathMetric = newMetric;
                }
            }
        }
        else
        {
            auto c = bf.reachNext();
            if ( !c.v )
                continue;
            if ( keepGrowing )
                bf.addOrgRingSteps( c );
            if ( auto info = bs.getVertInfo( c.v ) )
            {
                auto newMetric = c.metric + info->metric;
                if ( newMetric < joinPathMetric )
                {
                    join = c.v;
                    joinPathMetric = newMetric;
                }
            }
        }
    }

    EdgePath res;
    if ( join )
    {
        res = bs.getPathBack( join );
        reverse( res );
        auto tail = bf.getPathBack( join );
        res.insert( res.end(), tail.begin(), tail.end() );
        assert( isEdgePath( topology, res ) );

        if ( res.empty() )
        {
            if ( outPathStart )
                *outPathStart = join;
            if ( outPathFinish )
                *outPathFinish = join;
        }
        else
        {
            assert( numStarts > 1 || topology.org( res.front() ) == starts[0].v );
            assert( numFinishes > 1 || topology.dest( res.back() ) == finishes[0].v );

            if ( outPathStart )
                *outPathStart = topology.org( res.front() );
            if ( outPathFinish )
                *outPathFinish = topology.dest( res.back() );
        }
    }
    return res;
}

EdgePath buildShortestPath( const Mesh & mesh, VertId start, VertId finish, float maxPathLen )
{
    return buildSmallestMetricPath( mesh.topology, edgeLengthMetric( mesh ), start, finish, maxPathLen );
}

EdgePath buildShortestPathBiDir( const Mesh & mesh, VertId start, VertId finish, float maxPathLen )
{
    return buildSmallestMetricPathBiDir( mesh.topology, edgeLengthMetric( mesh ), start, finish, maxPathLen );
}

EdgePath buildShortestPathBiDir( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & finish,
    VertId * outPathStart, VertId * outPathFinish, float maxPathLen )
{
    const auto startPt = mesh.triPoint( start );
    TerminalVertex starts[3];
    int numStarts = 0;
    mesh.topology.forEachVertex( start, [&]( VertId v )
    {
        starts[ numStarts++ ] = { v, ( mesh.points[v] - startPt ).length() };
    } );

    const auto finishPt = mesh.triPoint( finish );
    TerminalVertex finishes[3];
    int numFinishes = 0;
    mesh.topology.forEachVertex( finish, [&]( VertId v )
    {
        finishes[ numFinishes++ ] = { v, ( mesh.points[v] - finishPt ).length() };
    } );

    return buildSmallestMetricPathBiDir( mesh.topology, edgeLengthMetric( mesh ),
        starts, numStarts,
        finishes, numFinishes,
        outPathStart, outPathFinish, maxPathLen );
}

EdgePath buildShortestPath( const Mesh& mesh, VertId start, const VertBitSet& finish, float maxPathLen /*= FLT_MAX */ )
{
    return buildSmallestMetricPath( mesh.topology, edgeLengthMetric( mesh ), start, finish, maxPathLen );
}

EdgePath buildShortestPathAStar( const Mesh & mesh, VertId start, VertId finish, float maxPathLen )
{
    return buildShortestPathAStar( mesh,
        MeshTriPoint( mesh.topology, start ),
        MeshTriPoint( mesh.topology, finish ),
        nullptr, nullptr, maxPathLen );
}

EdgePath buildShortestPathAStar( const Mesh & mesh, const MeshTriPoint & start, const MeshTriPoint & finish,
    VertId * outPathStart, VertId * outPathFinish, float maxPathLen )
{
    MR_TIMER
    EdgePathsAStarBuilder b( mesh, start, finish );

    VertId starts[3];
    int numStarts = 0;
    mesh.topology.forEachVertex( start, [&]( VertId v )
    {
        starts[ numStarts++ ] = v;
    } );

    for (;;)
    {
        auto vinfo = b.growOneEdge();
        if ( !vinfo.v )
        {
            // unable to find the path
            return {};
        }
        if ( vinfo.metric > maxPathLen )
        {
            // unable to find the path within given metric limitation
            return {};
        }
        auto it = std::find( starts, starts + numStarts, vinfo.v );
        if ( it != starts + numStarts )
        {
            if ( outPathStart )
                *outPathStart = vinfo.v;
            EdgePath res = b.getPathBack( vinfo.v );
            if ( outPathFinish )
                *outPathFinish = res.empty() ? vinfo.v : mesh.topology.dest( res.back() );
            return res;
        }
    }
}

std::vector<VertId> getVertexOrdering( const MeshTopology & topology, VertBitSet region )
{
    MR_TIMER

    auto metric = [&]( EdgeId e ) 
    { 
        return region.test( topology.dest( e ) ) ? 1.0f : FLT_MAX; 
    }; 
    EdgePathsBuilder b( topology, metric );

    std::vector<VertId> res;
    res.reserve( region.count() );
    auto addToRes = [&]( VertId v )
    {
        region.reset( v );
        res.push_back( v );
    };

    while ( auto v0 = region.find_first() )
    {
        addToRes( v0 );
        b.addStart( v0, 0 );
        for(;;)
        {
            auto vinfo = b.growOneEdge();
            if ( !vinfo.v )
                break;
            addToRes( vinfo.v );
        }
    }
    return res;
}

std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, EdgeBitSet & edges )
{
    MR_TIMER
    std::vector<EdgeLoop> res;
    for ( ;; )
    {
        UnionFind<VertId> vertComponents( topology.vertSize() );
        EdgeId loopEdge;
        for ( EdgeId e : edges )
        {
            const auto o = topology.org( e );
            const auto d = topology.dest( e );
            assert( o != d );
            if ( vertComponents.united( o, d ) )
            {
                loopEdge = e;
                break;
            }
            vertComponents.unite( o, d );
        }
        if ( !loopEdge.valid() )
            break;

        edges.reset( loopEdge );
        auto path = buildSmallestMetricPath( topology, [&]( EdgeId e )
        {
            return edges.test( e.sym() ) ? 1.0f : 1e5f; // sym() because inside buildSmallestMetricPath we search from finish to start
        }, topology.dest( loopEdge ), topology.org( loopEdge ) );
        assert ( !path.empty() );
        for ( EdgeId e : path )
        {
            assert( edges.test( e ) );
            edges.reset( e );
        }

        path.push_back( loopEdge );
        assert( isEdgeLoop( topology, path ) );

        res.push_back( std::move( path ) );
    }

    return res;
}

std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, const std::vector<EdgeId> & inEdges, EdgeBitSet * outNotLoopEdges )
{
    MR_TIMER
    EdgeBitSet edges;
    for ( auto e : inEdges )
    {
        if ( !edges.autoResizeTestSet( e.sym(), false ) )
            edges.autoResizeSet( e );
    }
    auto res = extractClosedLoops( topology, edges );
    if ( outNotLoopEdges )
        *outNotLoopEdges = std::move( edges );
    return res;
}

EdgeLoop extractLongestClosedLoop( const Mesh & mesh, const std::vector<EdgeId> & inEdges )
{
    MR_TIMER
    auto loops = extractClosedLoops( mesh.topology, inEdges );
    if ( loops.empty() )
        return {};
    sortPathsByLength( loops, mesh );
    return std::move( loops.back() );
}

bool dilateRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, FaceBitSet & region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getRegionBoundaryVerts( topology, region );
    if ( !dilateRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region |= getInnerFaces( topology, vertRegion );
    return true;
}

bool dilateRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, VertBitSet & region, float dilation, ProgressCallback callback )
{
    MR_TIMER

    EdgePathsBuilder builder( topology, metric );
    for( VertId v : region )
        builder.addStart( v, 0 );

    for ( int i = 0;; ++i )
    {
        const auto vinfo = builder.reachNext();
        if ( !vinfo.v || vinfo.penalty > dilation )
            break;

        region.autoResizeSet( vinfo.v );
        builder.addOrgRingSteps( vinfo );

        if ( !reportProgress( callback, [&]{ return vinfo.penalty / dilation; }, i, 1024 ) )
            return false;
    }

    if ( callback && !callback( 1.0f ) )
        return false;

    return true;
}

bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getIncidentVerts( topology, region );
    if ( !dilateRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region = getInnerEdges( topology, vertRegion );
    return true;
}

bool erodeRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, FaceBitSet & region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getRegionBoundaryVerts( topology, region );
    if ( !dilateRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region -= getInnerFaces( topology, vertRegion );
    return true;
}

bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, VertBitSet& region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto faceRegion = getInnerFaces( topology, region );
    if ( !erodeRegionByMetric( topology, metric, faceRegion, dilation, callback ) )
        return false;

    region = getIncidentVerts( topology, faceRegion );
    return true;
}

bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getIncidentVerts( topology, region );
    if ( !erodeRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region = getInnerEdges( topology, vertRegion );
    return true;
}

bool dilateRegion( const Mesh& mesh, FaceBitSet& region, float dilation, ProgressCallback callback )
{
    return dilateRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool dilateRegion( const Mesh& mesh, VertBitSet& region, float dilation, ProgressCallback callback )
{
    return dilateRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool dilateRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    return dilateRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool erodeRegion( const Mesh& mesh, FaceBitSet & region, float dilation, ProgressCallback callback )
{
    return erodeRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool erodeRegion( const Mesh& mesh, VertBitSet & region, float dilation, ProgressCallback callback )
{
    return erodeRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool erodeRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    return erodeRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

int getPathPlaneIntersections( const Mesh & mesh, const EdgePath & path, const Plane3f & plane,
    std::vector<MeshEdgePoint> * outIntersections )
{
    MR_TIMER;
    int found = 0;
    for ( auto e : path )
    {
        auto o = plane.distance( mesh.orgPnt( e ) );
        auto d = plane.distance( mesh.destPnt( e ) );
        if ( ( o <= 0 && d > 0 ) || ( o >= 0 && d < 0 ) )
        {
            float a = -o / ( d - o );
            if ( outIntersections )
                outIntersections->emplace_back( e, a );
            ++found;
        }
    }
    return found;
}

int getPathEdgesInPlane( const Mesh & mesh, const EdgePath & path, const Plane3f & plane, float tolerance,
    std::vector<EdgeId> * outInPlaneEdges )
{
    MR_TIMER;
    int found = 0;
    for ( auto e : path )
    {
        auto o = plane.distance( mesh.orgPnt( e ) );
        auto d = plane.distance( mesh.destPnt( e ) );
        if ( std::abs( o ) <= tolerance && std::abs( d ) <= tolerance )
        {
            if ( outInPlaneEdges )
                outInPlaneEdges->emplace_back( e );
            ++found;
        }
    }
    return found;
}

TEST(MRMesh, BuildShortestPath) 
{
    Mesh cube = makeCube();
    auto path = buildShortestPath( cube, 0_v, 6_v );
    EXPECT_EQ( path.size(), 2 );
    EXPECT_EQ( cube.topology.org( path[0] ), 0_v );
    EXPECT_EQ( cube.topology.dest( path[0] ), cube.topology.org( path[1] ) );
    EXPECT_EQ( cube.topology.dest( path[1] ), 6_v );

    auto path34 = buildShortestPath( cube, 3_v, 4_v );
    EXPECT_EQ( path34.size(), 2 );

    std::vector<EdgePath> paths{ path, path34 };
    auto euclid = edgeLengthMetric( cube );
    EXPECT_GT( calcPathMetric( paths[0], euclid ), calcPathMetric( paths[1], euclid ) );
    sortPathsByMetric( paths, euclid );
    EXPECT_LE( calcPathMetric( paths[0], euclid ), calcPathMetric( paths[1], euclid ) );
}

} //namespace MR
