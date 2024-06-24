#include "MRPolylineTopology.h"
#include "MRParallelFor.h"
#include "MRMapEdge.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

void PolylineTopology::buildOpenLines( const std::vector<VertId> & comp2firstVert )
{
    MR_TIMER
    if ( comp2firstVert.empty() )
    {
        assert( false );
        return;
    }
    assert( comp2firstVert.front() == 0_v );
    numValidVerts_ = comp2firstVert.back();
    edges_.resizeNoInit( 2 * numValidVerts_ ); // the edges in between lines will be lone
    edgePerVertex_.resizeNoInit( numValidVerts_ );
    validVerts_.clear();
    validVerts_.resize( numValidVerts_, true );
    ParallelFor( edgePerVertex_, [&]( VertId v )
    {
        EdgeId e( 2 * (int)v );
        if ( v + 1 >= numValidVerts_ )
            return;
        edgePerVertex_[v] = e;
        edges_[e].next = v > 0 ? e - 1 : e;
        edges_[e].org = v;
        edges_[e + 1].next = e + 2;
        edges_[e + 1].org = v + 1;
    } );
    for ( int j = 0; j + 1 < comp2firstVert.size(); ++j )
    {
        auto v0 = comp2firstVert[j];
        auto v1 = comp2firstVert[j + 1];
        if ( v0 == v1 )
            continue;

        EdgeId e0( 2 * (int)v0 );
        assert( org( e0 ) == v0 );
        edges_[e0].next = e0;

        EdgeId e1( 2 * (int)(v1 - 1) );
        assert( org( e1 - 1 ) == v1 - 1 );
        if ( v1 < numValidVerts_ )
        {
            assert( org( e1 ) == v1 - 1 );
            assert( edgePerVertex_[v1 - 1] == e1 );
        }
        edgePerVertex_[v1 - 1] = e1 - 1;
        edges_[e1 - 1].next = e1 - 1;
        edges_[e1].next = e1;
        edges_[e1 + 1].next = e1 + 1;
        edges_[e1].org = {};
        edges_[e1 + 1].org = {};
        assert( isLoneEdge( e1 ) );
    }
    assert( checkValidity() );
}

void PolylineTopology::vertResize( size_t newSize )
{
    if ( edgePerVertex_.size() >= newSize )
        return;
    edgePerVertex_.resize( newSize );
    validVerts_.resize( newSize );
}

void PolylineTopology::vertResizeWithReserve( size_t newSize )
{ 
    if ( edgePerVertex_.size() >= newSize )
        return;
    edgePerVertex_.resizeWithReserve( newSize );
    validVerts_.resizeWithReserve( newSize );
}

EdgeId PolylineTopology::makeEdge()
{
    assert( edges_.size() % 2 == 0 );
    EdgeId he0( int( edges_.size() ) );
    EdgeId he1( int( edges_.size() + 1 ) );

    HalfEdgeRecord d0;
    d0.next = he0;
    edges_.push_back( d0 );

    HalfEdgeRecord d1;
    d1.next = he1;
    edges_.push_back( d1 );

    return he0;
}

EdgeId PolylineTopology::makeEdge( VertId a, VertId b )
{
    const auto ea = edgeWithOrg( a );
    if ( ea && next( ea ) != ea )
        return {};
    const auto eb = edgeWithOrg( b );
    if ( eb && next( eb ) != eb )
        return {};

    const auto newe = makeEdge();

    if ( ea )
        splice( ea, newe );
    else
        setOrg( newe, a );

    if ( eb )
        splice( eb, newe.sym() );
    else
        setOrg( newe.sym(), b );

    return newe;
}

bool PolylineTopology::isLoneEdge( EdgeId a ) const
{
    assert( a.valid() );
    auto & adata = edges_[a];
    if ( adata.org.valid() || adata.next != a )
        return false;

    auto b = a.sym();
    auto & bdata = edges_[b];
    if ( bdata.org.valid() || bdata.next != b )
        return false;

    return true;
}

EdgeId PolylineTopology::lastNotLoneEdge() const
{
    assert( edges_.size() % 2 == 0 );
    for ( EdgeId i{ (int)edges_.size() - 1 }; i.valid(); ----i ) // one decrement returns sym-edge
    {
        if ( !isLoneEdge( i ) )
            return i;
    }
    return {};
}

size_t PolylineTopology::computeNotLoneUndirectedEdges() const
{
    MR_TIMER;
    size_t res = 0;
    for ( EdgeId i{ 0 }; i < (int)edges_.size(); ++++i ) // one increment returns sym-edge
        if ( !isLoneEdge( i ) )
            ++res;
    return res;
}

void PolylineTopology::deleteEdge( UndirectedEdgeId ue )
{
    assert( ue.valid() );
    const EdgeId e = ue;

    if ( next( e ) != e )
        splice( next( e ), e );
    else
        setOrg( e, {} );

    if ( next( e.sym() ) != e.sym() )
        splice( next( e.sym() ), e.sym() );
    else
        setOrg( e.sym(), {} );

    assert( isLoneEdge( e ) );
}

void PolylineTopology::deleteEdges( const UndirectedEdgeBitSet & es )
{
    MR_TIMER
    for ( auto ue : es )
        deleteEdge( ue );
}

size_t PolylineTopology::heapBytes() const
{
    return
        edges_.heapBytes() +
        edgePerVertex_.heapBytes() +
        validVerts_.heapBytes();
}

void PolylineTopology::splice( EdgeId a, EdgeId b )
{
    assert( a.valid() && b.valid() );
    if ( a == b )
        return;

    [[maybe_unused]] bool wasSameORing = next( a ) == b;
    assert( wasSameORing == ( next( b ) == a ) );

    auto & aData = edges_[a];
    [[maybe_unused]] auto & aNextData = edges_[next( a )];
    assert( aNextData.next == a );
    auto & bData = edges_[b];
    [[maybe_unused]] auto & bNextData = edges_[next( b )];
    assert( bNextData.next == b );

    bool wasSameOriginId = aData.org == bData.org;
    assert( wasSameOriginId || !aData.org.valid() || !bData.org.valid() );

    if ( !wasSameOriginId )
    {
        if ( aData.org.valid() )
            setOrg_( b, aData.org );
        else if ( bData.org.valid() )
            setOrg_( a, bData.org );
    }

    std::swap( aData.next, bData.next );
    assert( ( &aData == &aNextData ) == ( &bData == &bNextData ) );

    if ( wasSameOriginId && bData.org.valid() )
    {
        setOrg_( b, VertId() );
        if ( aData.org )
            edgePerVertex_[aData.org] = a;
    }

    assert( ( wasSameORing && next( a ) != b && next( b ) != a ) || ( !wasSameORing && next( a ) == b && next( b ) == a ) );
}

void PolylineTopology::setOrg_( EdgeId a, VertId v )
{
    assert( a.valid() );
    for ( EdgeId i = a; ; )
    {
        edges_[i].org = v;
        i = edges_[i].next;
        if ( i == a )
            break;
    }
}

void PolylineTopology::setOrg( EdgeId a, VertId v )
{
    auto oldV = org( a );
    if ( v == oldV )
        return;
    setOrg_( a, v );
    if ( oldV.valid() )
    {
        assert( edgePerVertex_[oldV].valid() );
        edgePerVertex_[oldV] = EdgeId();
        validVerts_.reset( oldV );
        --numValidVerts_;
    }
    if ( v.valid() )
    {
        assert( !edgePerVertex_[v].valid() );
        edgePerVertex_[v] = a;
        validVerts_.set( v );
        ++numValidVerts_;
    }
}

EdgeId PolylineTopology::findEdge( VertId o, VertId d ) const
{
    assert( o.valid() && d.valid() );
    EdgeId e0 = edgeWithOrg( o );
    if ( !e0.valid() )
        return {};

    for ( EdgeId e = e0;; )
    {
        if ( dest( e ) == d )
            return e;
        e = next( e );
        if ( e == e0 )
            return {};
    }
}

int PolylineTopology::getVertDegree( VertId a ) const
{
    const auto e = edgeWithOrg( a );
    if ( !e )
        return 0;
    const auto e1 = next( e );
    if ( e == e1 )
        return 1;
    assert( e == next( e1 ) );
    return 2;
}

VertId PolylineTopology::lastValidVert() const
{
    if ( numValidVerts_ <= 0 )
        return {};
    return validVerts_.find_last();
}

VertBitSet PolylineTopology::getPathVertices( const EdgePath & path ) const
{
    VertBitSet res;
    for ( auto e : path )
    {
        res.autoResizeSet( org( e ) );
        res.autoResizeSet( dest( e ) );
    }
    return res;
}

EdgeId PolylineTopology::splitEdge( EdgeId e )
{
    // disconnect edge e from its origin
    EdgeId eNext = next( e );
    VertId v0;
    if ( eNext != e )
    {
        splice( eNext, e );
    }
    else
    {
        v0 = org( e );
        setOrg( e, {} );
    }

    // e now becomes the second part of split edge, add first part to it
    EdgeId e0 = makeEdge();
    assert( !org(e) );
    splice( e, e0.sym() );
    if ( eNext != e )
        splice( eNext, e0 );
    else
        setOrg( e0, v0 );

    // allocate id from new vertex
    VertId newv = addVertId();
    setOrg( e, newv );
    return e0;
}

EdgeId PolylineTopology::makePolyline( const VertId * vs, size_t num )
{
    if ( !vs || num < 2 )
    {
        assert( false );
        return {};
    }

    VertId maxVertId;
    for ( size_t i = 0; i < num; ++i )
        maxVertId = std::max( maxVertId, vs[i] );
    if ( maxVertId >= (int)vertSize() )
        vertResizeWithReserve( maxVertId + 1 );

    PolylineMaker maker{ *this };
    auto e0 = maker.start( vs[0] );
    for ( int j = 1; j + 1 < num; ++j )
        maker.proceed( vs[j] );

    if ( vs[0] == vs[num-1] )
        maker.close();
    else
        maker.finishOpen( vs[num-1] );
    return e0;
}

void PolylineTopology::addPart( const PolylineTopology & from, VertMap * outVmap, WholeEdgeMap * outEmap )
{
    MR_TIMER

    // in all maps: from index -> to index
    WholeEdgeMap emap;
    emap.resize( from.undirectedEdgeSize() );
    EdgeId firstNewEdge = edges_.endId();
    for ( UndirectedEdgeId i{ 0 }; i < emap.size(); ++i )
    {
        if ( from.isLoneEdge( i ) )
            continue;
        emap[i] = edges_.endId();
        edges_.push_back( from.edges_[ EdgeId( i ) ] );
        edges_.push_back( from.edges_[ EdgeId( i ).sym() ] );
    }

    VertMap vmap;
    VertId lastFromValidVertId = from.lastValidVert();
    vmap.resize( lastFromValidVertId + 1 );
    for ( VertId i{ 0 }; i <= lastFromValidVertId; ++i )
    {
        auto efrom = from.edgePerVertex_[i];
        if ( !efrom.valid() )
            continue;
        auto nv = addVertId();
        vmap[i] = nv;
        edgePerVertex_[nv] = mapEdge( emap, efrom );
        validVerts_.set( nv );
        ++numValidVerts_;
    }

    // translate edge records
    tbb::parallel_for( tbb::blocked_range( firstNewEdge.undirected(), edges_.endId().undirected() ),
        [&]( const tbb::blocked_range<UndirectedEdgeId> & range )
    {
        for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
        {
            const EdgeId e{ ue };
            edges_[e].next = mapEdge( emap, edges_[e].next );
            edges_[e.sym()].next = mapEdge( emap, edges_[e.sym()].next );
        
            edges_[e].org = vmap[edges_[e].org];
            edges_[e.sym()].org = vmap[edges_[e.sym()].org];
        }
    } );

    if ( outVmap )
        *outVmap = std::move( vmap );
    if ( outEmap )
        *outEmap = std::move( emap );
}

void PolylineTopology::addPartByMask( const PolylineTopology& from, const UndirectedEdgeBitSet& mask, 
    VertMap* outVmap /*= nullptr*/, EdgeMap* outEmap /*= nullptr */ )
{
    MR_TIMER
    // in all maps: from index -> to index
    EdgeMap emap;
    EdgeId lastFromValidEdgeId = from.lastNotLoneEdge();
    emap.resize( lastFromValidEdgeId + 1 );
    for ( auto ue : mask )
    {
        if ( from.isLoneEdge( ue ) )
            continue;
        auto e = EdgeId( ue );
        emap[e] = makeEdge();
        emap[e.sym()] = emap[e].sym();
    }

    VertMap vmap;
    VertId lastFromValidVertId = from.lastValidVert();
    vmap.resize( lastFromValidVertId + 1 );
    VertId maxValidVert_;
    for ( auto ue : mask )
    {
        if ( from.isLoneEdge( ue ) )
            continue;
        for ( EdgeId e : { EdgeId( ue ), EdgeId( ue ).sym() } )
        {
            auto v = from.org( e );
            if ( vmap[v].valid() )
                continue;
            auto nv = addVertId();
            vmap[v] = nv;
            edgePerVertex_[nv] = emap[e];
            validVerts_.set( nv );
            maxValidVert_ = std::max( maxValidVert_, v );
            ++numValidVerts_;
        }
    }
    
    const auto& fromEdges = from.edges_;
    for ( auto ue : mask )
    {
        auto e = EdgeId( ue );
        auto ne = emap[fromEdges[e].next];
        // If next edge is not presented in mask then it's value in emap is invalid. In that case we should skip it
        if ( ne.valid() )
            edges_[emap[e]].next = ne;

        ne = emap[fromEdges[e.sym()].next];
        if ( ne.valid() )
            edges_[emap[e.sym()]].next = ne;
        
        edges_[emap[e]].org = vmap[fromEdges[e].org];
        edges_[emap[e.sym()]].org = vmap[fromEdges[e.sym()].org];
    }

    vmap.resize( maxValidVert_ + 1 );
    emap.resize( EdgeId( mask.find_last() + 1 ) );

    if ( outVmap )
        *outVmap = std::move( vmap );
    if ( outEmap )
        *outEmap = std::move( emap );
}

void PolylineTopology::pack( VertMap * outVmap, WholeEdgeMap * outEmap )
{
    MR_TIMER

    PolylineTopology packed;
    packed.vertReserve( numValidVerts() );
    packed.edgeReserve( 2 * computeNotLoneUndirectedEdges() );
    packed.addPart( *this, outVmap, outEmap );
    *this = std::move( packed );
}

void PolylineTopology::write( std::ostream & s ) const
{
    // write edges
    auto numEdges = (std::uint32_t)edges_.size();
    s.write( (const char*)&numEdges, 4 );
    s.write( (const char*)edges_.data(), edges_.size() * sizeof(HalfEdgeRecord) );

    // write verts
    auto numVerts = (std::uint32_t)edgePerVertex_.size();
    s.write( (const char*)&numVerts, 4 );
    s.write( (const char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof(EdgeId) );
}

bool PolylineTopology::read( std::istream & s )
{
    // read edges
    std::uint32_t numEdges;
    s.read( (char*)&numEdges, 4 );
    if ( !s )
        return false;

    auto posCur = s.tellg();
    s.seekg( 0, std::ios_base::end );
    const auto posEnd = s.tellg();
    s.seekg( posCur );
    if ( size_t( posEnd - posCur ) < numEdges * sizeof(HalfEdgeRecord) )
        return false; // stream is too short

    edges_.resize( numEdges );
    s.read( (char*)edges_.data(), edges_.size() * sizeof(HalfEdgeRecord) );

    // read verts
    std::uint32_t numVerts;
    s.read( (char*)&numVerts, 4 );
    if ( !s )
        return false;
    edgePerVertex_.resize( numVerts );
    validVerts_.resize( numVerts );
    s.read( (char*)edgePerVertex_.data(), edgePerVertex_.size() * sizeof(EdgeId) );

    computeValidsFromEdges();

    return s.good() && checkValidity();
}

bool PolylineTopology::isConsistentlyOriented() const
{
    MR_TIMER

    for ( EdgeId e{0}; e < edges_.size(); ++e )
    {
        auto ne = next( e );
        if ( e == ne || e.odd() == ne.sym().odd() ) 
            continue;
        return false;
    }
    return true;
}

void PolylineTopology::flip()
{
    MR_TIMER

    for ( auto & e : edgePerVertex_ )
    {
        if ( e.valid() )
            e = e.sym();
    }

    for ( EdgeId i{0}; i + 1 < edges_.size(); ++++i )
    {
        auto & r0 = edges_[i];
        auto & r1 = edges_[i + 1];
        std::swap( r0, r1 );
        r0.next = r0.next.sym();
        r1.next = r1.next.sym();
    }
}

#define CHECK(x) { assert(x); if (!(x)) return false; }

bool PolylineTopology::checkValidity() const
{
    MR_TIMER

    for ( EdgeId e{0}; e < edges_.size(); ++e )
    {
        CHECK( edges_[edges_[e].next].next == e );
        if ( auto v = edges_[e].org )
            CHECK( validVerts_.test( v ) );
    }

    const auto vSize = edgePerVertex_.size();
    CHECK( vSize == validVerts_.size() )

    int realValidVerts = 0;
    for ( VertId v{0}; v < edgePerVertex_.size(); ++v )
    {
        if ( edgePerVertex_[v].valid() )
        {
            CHECK( validVerts_.test( v ) )
            const auto e0 = edgePerVertex_[v]; 
            CHECK( e0 < edges_.size() );
            CHECK( edges_[e0].org == v );
            ++realValidVerts;
            for ( EdgeId e = e0;; )
            {
                CHECK( org(e) == v );
                e = next( e );
                if ( e == e0 )
                    break;
            }
        }
        else
        {
            CHECK( !validVerts_.test( v ) )
        }
    }
    CHECK( numValidVerts_ == realValidVerts );

    return true;
}

void PolylineTopology::computeValidsFromEdges()
{
    MR_TIMER

    numValidVerts_ = 0;
    for ( VertId v{0}; v < edgePerVertex_.size(); ++v )
        if ( edgePerVertex_[v].valid() )
        {
            validVerts_.set( v );
            ++numValidVerts_;
        }
}

bool PolylineTopology::isClosed() const
{
    MR_TIMER;
    for ( EdgeId e( 0 ); e < edges_.size(); ++e )
    {
        if ( !edges_[e].org.valid() )
            continue; // skip edges without valid vertices
        if ( edges_[e].next == e )
            return false;
    }
    return true;
}

TEST( MRMesh, PolylineTopology )
{
    PolylineTopology t;
    VertId vs[4] = { 0_v, 1_v, 2_v, 0_v };
    t.makePolyline( vs, 4 );
    EXPECT_TRUE( t.checkValidity() );
    EXPECT_TRUE( t.isConsistentlyOriented() );
    EXPECT_EQ( t.org( 0_e ), 0_v );
    EXPECT_EQ( t.dest( 0_e ), 1_v );

    t.flip();
    EXPECT_TRUE( t.checkValidity() );
    EXPECT_TRUE( t.isConsistentlyOriented() );
    EXPECT_EQ( t.org( 0_e ), 1_v );
    EXPECT_EQ( t.dest( 0_e ), 0_v );

    EXPECT_EQ( t.numValidVerts(), 3 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 3 );

    t.deleteEdge( 0_ue );
    EXPECT_EQ( t.numValidVerts(), 3 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 2 );

    t.deleteEdge( 1_ue );
    EXPECT_EQ( t.numValidVerts(), 2 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 1 );

    t.deleteEdge( 2_ue );
    EXPECT_EQ( t.numValidVerts(), 0 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 0 );
}

} //namespace MR
