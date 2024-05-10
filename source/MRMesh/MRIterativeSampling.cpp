#include "MRIterativeSampling.h"
#include "MRPointCloud.h"
#include "MRHeap.h"
#include "MRBuffer.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsProject.h"
#include "MRTimer.h"
#include <cfloat>

namespace MR
{

namespace
{

// an element of heap
struct PointInfo
{
    /// negative squared distance to the closest neighbor, points with minimum distance (maximal negative distance) are merged first
    float negDistSq;
    /// the number of points having this as the closest neighbor, points with maximal number are merged first
    int numRecipClosest;

    explicit PointInfo( NoInit ) noexcept {}
    PointInfo() : negDistSq( -FLT_MAX ), numRecipClosest( 0 ) {}

    auto operator <=>( const PointInfo & ) const = default;
};

} //anonymous namespace

std::optional<VertBitSet> pointIterativeSampling( const PointCloud& cloud, int numSamples, const ProgressCallback & cb )
{
    MR_TIMER
    VertBitSet res = cloud.validPoints;
    auto toRemove = (int)res.count() - numSamples;
    if ( toRemove <= 0 )
        return res;

    const auto sz = cloud.validPoints.size();
    Buffer<VertId, VertId> closestNei( sz );
    Buffer<PointInfo, VertId> info( sz );
    cloud.getAABBTree();
    BitSetParallelFor( cloud.validPoints, [&]( VertId v )
    {
        const auto prj = findProjectionOnPoints( cloud.points[v], cloud, FLT_MAX, nullptr, 0, [v]( VertId x ) { return v == x; } );
        closestNei[v] = prj.vId;
        info[v].negDistSq = -prj.distSq;
    } );

    if ( !reportProgress( cb, 0.1f ) )
        return {};

    Vector<VertId, VertId> first( sz ); ///< first[v] contains a pointId having closest point v
    Vector<VertId, VertId> next( sz );  ///< next[v] contains pointId having the same closest point as v's closest point
    for ( auto v : cloud.validPoints )
    {
        const auto cv = closestNei[v];
        ++info[cv].numRecipClosest;
        if ( auto f = first[cv] )
            next[v] = f;
        first[cv] = v;
    }

    if ( !reportProgress( cb, 0.2f ) )
        return {};

    using HeapT = Heap<PointInfo, VertId>;
    std::vector<HeapT::Element> elms;
    elms.reserve( numSamples + toRemove );
    for ( auto vid : cloud.validPoints )
        elms.push_back( { vid, info[vid] } );
    info.clear();
    HeapT heap( std::move( elms ) );

    if ( !reportProgress( cb, 0.3f ) )
        return {};

    const auto k = 1.0f / toRemove;
    while ( toRemove > 0 )
    {
        auto [v, vinfo] = heap.top();
        assert( vinfo.negDistSq > -FLT_MAX );
        --toRemove;
        assert( res.test( v ) );
        res.reset( v );
        heap.setSmallerValue( v, PointInfo() );

        auto cv = closestNei[v];
        auto cvinfo = heap.value( cv );
        assert( cvinfo.numRecipClosest > 0 );
        --cvinfo.numRecipClosest;
        heap.setSmallerValue( cv, cvinfo );

        auto nr = first[v];
        for ( auto r = nr; nr = next[r], r; r = nr )
        {
            const auto prj = findProjectionOnPoints( cloud.points[r], cloud, FLT_MAX, nullptr, 0, [&res]( VertId x ) { return !res.test( x ); } );
            assert( closestNei[r] == v );
            const auto cr = closestNei[r] = prj.vId;
            auto rinfo = heap.value( r );
            assert( rinfo.negDistSq > -FLT_MAX );
            assert( rinfo.negDistSq >= -prj.distSq );
            if ( rinfo.negDistSq > prj.distSq )
            {
                rinfo.negDistSq = -prj.distSq;
                heap.setSmallerValue( r, rinfo );
            }

            auto crinfo = heap.value( cr );
            ++crinfo.numRecipClosest;
            heap.setLargerValue( cr, crinfo );
            if ( auto f = first[cr] )
                next[r] = f;
            first[cr] = r;
        }
        if ( !reportProgress( cb, [&] { return 0.3f + 0.7f * ( 1 - k * toRemove ); }, toRemove, 0x10000 ) )
            return {};
    }

    if ( !reportProgress( cb, 1.0f ) )
        return {};
    return res;
}

} //namespace MR
