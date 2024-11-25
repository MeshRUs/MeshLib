#include "MRIntersectionContour.h"
#include "MRMesh.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRFillContour.h"
#include "MRAffineXf3.h"
#include "MRParallelFor.h"
#include <parallel_hashmap/phmap.h>

namespace
{

using namespace MR;
// returns 4 * (contour area)^2
float calcLoneContourAreaSq( const OneMeshContour& contour )
{
    Vector3f dblDirArea;
    for ( int i = 0; i + 1 < contour.intersections.size(); ++i )
        dblDirArea += cross( contour.intersections[i].coordinate, contour.intersections[i + 1].coordinate );
    return dblDirArea.lengthSq();
}

bool isClosedContourTrivial( const MeshTopology& topology, const OneMeshContour& contour )
{
    assert( contour.closed );
    FaceBitSet fbs( topology.faceSize() );
    for ( const auto& inter : contour.intersections )
    {
        assert( inter.primitiveId.index() == OneMeshIntersection::Edge );
        auto eid = std::get<EdgeId>( inter.primitiveId );
        if ( auto l = topology.left( eid ) )
            fbs.set( l );
    }
    auto boundary = findLeftBoundary( topology, fbs );
    if ( boundary.empty() )
        return false;
    auto fillRes = fillContourLeft( topology, boundary.front() );
    return !fillRes.test( topology.right( boundary.front().front() ) );
}

}

namespace MR
{

struct EdgeTriHash
{
    size_t operator()( const EdgeTri& et ) const
    {
        return 17 * et.edge.undirected() + 23 * et.tri;
    }
};

using EdgeTriSet = HashSet<EdgeTri, EdgeTriHash>;

struct AccumulativeSet
{
    const MeshTopology& topologyA;
    const MeshTopology& topologyB;

    EdgeTriSet eAtB;
    EdgeTriSet eBtA;

    const MeshTopology& topologyByEdge( bool edgesATriB )
    {
        return edgesATriB ? topologyA : topologyB;
    }

    const MeshTopology& topologyByTri( bool edgesATriB )
    {
        return topologyByEdge( !edgesATriB );
    }

    EdgeTriSet& set( bool edgesATriB )
    {
        return edgesATriB ? eAtB : eBtA;
    }
    const EdgeTriSet& set( bool edgesATriB ) const
    {
        return edgesATriB ? eAtB : eBtA;
    }
    bool empty() const
    {
        return eAtB.empty() && eBtA.empty();
    }
    VariableEdgeTri getFirst() const
    {
        if ( !eAtB.empty() )
            return {*eAtB.begin(),true};
        else if ( !eBtA.empty() )
            return {*eBtA.begin(),false};
        return {};
    }
};

EdgeTriSet createSet( const std::vector<EdgeTri>& edgeTris )
{
    EdgeTriSet set;
    set.reserve( edgeTris.size() * 2 ); // 2 here is for mental peace
    for ( const auto& edgeTri : edgeTris )
        set.insert( edgeTri );
    return set;
}

bool erase( AccumulativeSet& accumulativeSet, VariableEdgeTri& item )
{
    auto& itemSet = accumulativeSet.set( item.isEdgeATriB );
    auto it = itemSet.find( item );
    if ( it == itemSet.end() )
        return false;
    item = {*it,item.isEdgeATriB};
    itemSet.erase( it );
    return true;
}

VariableEdgeTri orientBtoA( const VariableEdgeTri& curr )
{
    VariableEdgeTri res = curr;
    if ( !curr.isEdgeATriB )
        res.edge = res.edge.sym();
    return res;
}

VariableEdgeTri sym( const VariableEdgeTri& curr )
{
    VariableEdgeTri res = curr;
    res.edge = res.edge.sym();
    return res;
}

bool getNext( AccumulativeSet& accumulativeSet, const VariableEdgeTri& curr, VariableEdgeTri& next )
{
    const auto& edgeTopology = accumulativeSet.topologyByEdge( curr.isEdgeATriB );
    const auto& triTopology = accumulativeSet.topologyByTri( curr.isEdgeATriB );
    auto leftTri = edgeTopology.left( curr.edge );
    auto leftEdge = triTopology.edgePerFace()[curr.tri];

    assert( curr.tri );

    if ( leftTri.valid() )
    {
        VariableEdgeTri variants[5] =
        {
            {{edgeTopology.next( curr.edge ),curr.tri},curr.isEdgeATriB},
            {{edgeTopology.prev( curr.edge.sym() ) ,curr.tri},curr.isEdgeATriB},

            {{leftEdge,leftTri},!curr.isEdgeATriB},
            {{triTopology.next( leftEdge ),leftTri},!curr.isEdgeATriB},
            {{triTopology.prev( leftEdge.sym() ),leftTri},!curr.isEdgeATriB}
        };

        for ( const auto& v : variants )
        {
            if ( !v.edge.valid() )
                continue;
            next = v;
            if ( erase( accumulativeSet, next ) )
                return true;
        }
    }
    return false;
}

ContinuousContour orderFirstIntersectionContour( AccumulativeSet& accumulativeSet )
{
    ContinuousContour forwardRes;
    auto first = accumulativeSet.getFirst();
    forwardRes.push_back( orientBtoA( first ) );
    VariableEdgeTri next;
    while ( getNext( accumulativeSet, forwardRes.back(), next ) )
    {
        forwardRes.push_back( orientBtoA( next ) );
    }

    // if not closed
    if ( erase( accumulativeSet, first ) )
    {
        ContinuousContour backwardRes;
        backwardRes.push_back( orientBtoA( first ) );
        while ( getNext( accumulativeSet, sym( backwardRes.back() ), next ) )
        {
            backwardRes.push_back( orientBtoA( next ) );
        }
        forwardRes.insert( forwardRes.begin(), backwardRes.rbegin(), backwardRes.rend() - 1 );
    }
    return forwardRes;
}

ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections )
{
    MR_TIMER;
    AccumulativeSet accumulativeSet{topologyA,topologyB, createSet( intersections.edgesAtrisB ),createSet( intersections.edgesBtrisA ),};
    ContinuousContours res;
    while ( !accumulativeSet.empty() )
    {
        res.push_back( orderFirstIntersectionContour( accumulativeSet ) );
    }
    return res;
}

Contours3f extractIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& orientedContours, 
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    std::function<Vector3f( const Vector3f& coord, bool meshA )> getCoord;

    if ( !rigidB2A )
    {
        getCoord = [] ( const Vector3f& coord, bool )
        {
            return coord;
        };
    }
    else
    {
        getCoord = [xf = *rigidB2A] ( const Vector3f& coord, bool meshA )
        {
            return meshA ? coord : xf( coord );
        };
    }
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();

    Contours3f res( orientedContours.size() );
    for ( int i = 0; i < res.size(); ++i )
    {
        const auto& inCont = orientedContours[i];
        auto& resI = res[i];
        resI.resize( inCont.size() );
        ParallelFor( inCont, [&] ( size_t j )
        {
            Vector3f a, b, c, d, e;
            const auto& vet = inCont[j];
            if ( vet.isEdgeATriB )
            {
                meshB.getTriPoints( vet.tri, a, b, c );
                d = meshA.orgPnt( vet.edge );
                e = meshA.destPnt( vet.edge );
            }
            else
            {
                meshA.getTriPoints( vet.tri, a, b, c );
                d = meshB.orgPnt( vet.edge );
                e = meshB.destPnt( vet.edge );
            }
            // always calculate in mesh A space
            resI[j] = findTriangleSegmentIntersectionPrecise(
                getCoord( a, !vet.isEdgeATriB ),
                getCoord( b, !vet.isEdgeATriB ),
                getCoord( c, !vet.isEdgeATriB ),
                getCoord( d, vet.isEdgeATriB ),
                getCoord( e, vet.isEdgeATriB ), converters );
        } );
    }
    return res;
}

std::vector<int> detectLoneContours( const ContinuousContours& contours )
{
    std::vector<int> res;
    for ( int i = 0; i < contours.size(); ++i )
    {
        auto& contour = contours[i];
        if ( contour.empty() )
            continue;
        bool first = contour[0].isEdgeATriB;
        bool isLone = true;
        for ( const auto& vet : contour )
        {
            if ( vet.isEdgeATriB != first )
            {
                isLone = false;
                break;
            }
        }
        if ( isLone )
            res.push_back( i );
    }
    return res;
}

void removeLoneDegeneratedContours( const MeshTopology& edgesTopology, OneMeshContours& faceContours, OneMeshContours& edgeContours )
{
    for ( int i = int( faceContours.size() ) - 1; i >= 0; --i )
    {
        if ( faceContours[i].closed && calcLoneContourAreaSq( faceContours[i] ) == 0.0f && isClosedContourTrivial( edgesTopology, edgeContours[i] ) )
        {
            faceContours.erase( faceContours.begin() + i );
            edgeContours.erase( edgeContours.begin() + i );
        }
    }
}

void removeLoneContours( ContinuousContours& contours )
{
    auto loneContours = detectLoneContours( contours );
    for ( int i = int( loneContours.size() ) - 1; i >= 0; --i )
    {
        contours.erase( contours.begin() + loneContours[i] );
    }
}

} //namespace MR
