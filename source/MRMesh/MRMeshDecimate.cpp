#include "MRMeshDecimate.h"
#include "MRMesh.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRCylinder.h"
#include "MRGTest.h"
#include "MRMeshDelone.h"
#include "MRMeshSubdivide.h"
#include "MRMeshRelax.h"
#include "MRLineSegm.h"
#include <queue>

namespace MR
{

/// collapses given edge and deletes
/// 1) faces: left( e ) and right( e );
/// 2) vertex org( e )/dest( e ) if given edge was their only edge, otherwise only dest( e );
/// 3) edges: e, next( e.sym() ), prev( e.sym() ), and optionally next( e ), prev( e ) if their left and right triangles are deleted;
/// updates notFlippable removing deleted edges from there, and adding the edges that shall replace them;
/// calls onEdgeDel for every deleted edge;
/// returns prev( e ) if it is valid;
EdgeId collapseEdge( MeshTopology & topology, const EdgeId e, UndirectedEdgeBitSet* notFlippable, const std::function<void(EdgeId e, EdgeId e1)> & onEdgeDel )
{
    topology.setLeft( e, FaceId() );
    topology.setLeft( e.sym(), FaceId() );

    if ( notFlippable )
        notFlippable->reset( e.undirected() );
    if ( onEdgeDel )
        onEdgeDel( e, EdgeId{} );

    if ( topology.next( e ) == e )
    {
        topology.setOrg( e, VertId() );
        const EdgeId b = topology.prev( e.sym() );
        if ( b == e.sym() )
            topology.setOrg( e.sym(), VertId() );
        else
            topology.splice( b, e.sym() );

        assert( topology.isLoneEdge( e ) );
        return EdgeId();
    }

    topology.setOrg( e.sym(), VertId() );

    const EdgeId ePrev = topology.prev( e );
    const EdgeId eNext = topology.next( e );
    if ( ePrev != e )
        topology.splice( ePrev, e );

    const EdgeId a = topology.next( e.sym() );
    if ( a == e.sym() )
    {
        assert( topology.isLoneEdge( e ) );
        return ePrev != e ? ePrev : EdgeId();
    }
    const EdgeId b = topology.prev( e.sym() );

    topology.splice( b, e.sym() );
    assert( topology.isLoneEdge( e ) );

    assert( topology.next( b ) == a );
    assert( topology.next( ePrev ) == eNext );
    topology.splice( b, ePrev );
    assert( topology.next( b ) == eNext );
    assert( topology.next( ePrev ) == a );

    if ( topology.next( a.sym() ) == ePrev.sym() )
    {
        topology.splice( ePrev, a );
        topology.splice( topology.prev( a.sym() ), a.sym() );
        assert( topology.isLoneEdge( a ) );
        if ( !topology.left( ePrev ) && !topology.right( ePrev ) )
        {
            topology.splice( topology.prev( ePrev ), ePrev );
            topology.splice( topology.prev( ePrev.sym() ), ePrev.sym() );
            topology.setOrg( ePrev, {} );
            topology.setOrg( ePrev.sym(), {} );
            assert( topology.isLoneEdge( ePrev ) );
            if ( notFlippable )
            {
                notFlippable->reset( a.undirected() );
                notFlippable->reset( ePrev.undirected() );
            }
            if ( onEdgeDel )
            {
                onEdgeDel( a, EdgeId{} );
                onEdgeDel( ePrev, EdgeId{} );
            }
        }
        else 
        {
            if ( notFlippable && notFlippable->test_set( a.undirected(), false ) )
                notFlippable->autoResizeSet( ePrev.undirected() );
            if ( onEdgeDel )
                onEdgeDel( a, ePrev );
        }
    }

    if ( topology.next( eNext.sym() ) == b.sym() )
    {
        topology.splice( eNext.sym(), b.sym() );
        topology.splice( topology.prev( b ), b );
        assert( topology.isLoneEdge( b ) );
        if ( !topology.left( eNext ) && !topology.right( eNext ) )
        {
            topology.splice( topology.prev( eNext ), eNext );
            topology.splice( topology.prev( eNext.sym() ), eNext.sym() );
            topology.setOrg( eNext, {} );
            topology.setOrg( eNext.sym(), {} );
            assert( topology.isLoneEdge( eNext ) );
            if ( notFlippable )
            {
                notFlippable->reset( b.undirected() );
                notFlippable->reset( eNext.undirected() );
            }
            if ( onEdgeDel )
            {
                onEdgeDel( b, EdgeId{} );
                onEdgeDel( eNext, EdgeId{} );
            }
        }
        else
        {
            if ( notFlippable && notFlippable->test_set( b.undirected(), false ) )
                notFlippable->autoResizeSet( eNext.undirected() );
            if ( onEdgeDel )
                onEdgeDel( b, eNext );
        }
    }

    return ePrev != e ? ePrev : EdgeId();
}

class MeshDecimator
{
public:
    MeshDecimator( Mesh & mesh, const DecimateSettings & settings );
    DecimateResult run();

private: 
    Mesh & mesh_;
    const DecimateSettings & settings_;
    const DeloneSettings deloneSettings_;
    const float maxErrorSq_;
    Vector<QuadraticForm3f, VertId> myVertForms_;
    Vector<QuadraticForm3f, VertId> * pVertForms_ = nullptr;
    UndirectedEdgeBitSet regionEdges_;
    VertBitSet myBdVerts_;
    const VertBitSet * pBdVerts_ = nullptr;

    enum class EdgeOp : unsigned int
    {
        CollapseOptPos, ///< collapse the edge with target position optimization
        CollapseEnd,    ///< collapse the edge in one of its current vertices
        Flip            ///< flip the edge inside quadrangle
        // one more option is available to fit in 2 bits
    };

    struct QueueElement
    {
        float c = 0;
        struct X {
            EdgeOp edgeOp : 2 = EdgeOp::CollapseOptPos;
            unsigned int uedgeId : 30 = 0;
        } x;
        UndirectedEdgeId uedgeId() const { return UndirectedEdgeId{ (int)x.uedgeId }; }
        std::pair<float, int> asPair() const { return { -c, x.uedgeId }; }
        bool operator < ( const QueueElement & r ) const { return asPair() < r.asPair(); }
    };
    static_assert( sizeof( QueueElement ) == 8 );
    std::priority_queue<QueueElement> queue_;
    UndirectedEdgeBitSet presentInQueue_;
    DecimateResult res_;
    std::vector<VertId> originNeis_;
    std::vector<Vector3f> triDblAreas_; // directed double areas of newly formed triangles to check that they are consistently oriented
    class EdgeMetricCalc;

    bool initializeQueue_();
    std::optional<QueueElement> computeQueueElement_( UndirectedEdgeId ue, bool optimizeVertexPos,
        QuadraticForm3f * outCollapseForm = nullptr, Vector3f * outCollapsePos = nullptr ) const;
    void addInQueueIfMissing_( UndirectedEdgeId ue );

    enum class CollapseStatus
    {
        Done,        ///< collapse was successful
        SharedEdge,  ///< cannot collapse internal edge if its left and right faces share another edge
        PosFarBd,    ///< new vertex position is too far from collapsing boundary edge
        MultipleEdge,///< cannot collapse an edge because there is another edge in mesh with same ends
        Flippable,   ///< cannot collapse a flippable edge incident to a not-flippable edge
        TouchBd,     ///< prohibit collapse of an inner edge if it brings two boundaries in touch
        TriAspect,   ///< new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings
        LongEdge,    ///< new edge would be longer than all of old edges and longer than allowed in settings
        NormalFlip,  ///< if at least one triangle normal flips, checks that all new normals are consistent (do not check for degenerate edges)
        User         ///< user callback prohibits the collapse
    };

    struct CollapseRes
    {
        /// if collapse failed (or it was the last edge) then it is invalid, otherwise it is the remaining vertex
        VertId v;
        CollapseStatus status = CollapseStatus::Done;
    };

    CollapseRes collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos );
};

MeshDecimator::MeshDecimator( Mesh & mesh, const DecimateSettings & settings )
    : mesh_( mesh )
    , settings_( settings )
    , deloneSettings_{
        .maxAngleChange = settings.maxAngleChange,
        .criticalTriAspectRatio = settings.criticalTriAspectRatio,
        .region = settings.region }
    , maxErrorSq_( sqr( settings.maxError ) )
{
}

class MeshDecimator::EdgeMetricCalc 
{
public:
    EdgeMetricCalc( const MeshDecimator & decimator ) : decimator_( decimator ) { }
    EdgeMetricCalc( EdgeMetricCalc & x, tbb::split ) : decimator_( x.decimator_ ) { }
    void join( EdgeMetricCalc & y ) { auto yes = y.takeElements(); elems_.insert( elems_.end(), yes.begin(), yes.end() ); }

    const std::vector<QueueElement> & elements() const { return elems_; }
    std::vector<QueueElement> takeElements() { return std::move( elems_ ); }

    void operator()( const tbb::blocked_range<UndirectedEdgeId> & r ) 
    {
        const bool optimizeVertexPos = decimator_.settings_.optimizeVertexPos;
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue ) 
        {
            EdgeId e{ ue };
            if ( decimator_.regionEdges_.empty() )
            {
                if ( decimator_.mesh_.topology.isLoneEdge( e ) )
                    continue;
            }
            else
            {
                if ( !decimator_.regionEdges_.test( ue ) )
                    continue;
            }
            if ( auto qe = decimator_.computeQueueElement_( ue, optimizeVertexPos ) )
                elems_.push_back( *qe );
        }
    }

public:
    const MeshDecimator & decimator_;
    std::vector<QueueElement> elems_;
};

QuadraticForm3f computeFormAtVertex( const MR::MeshPart & mp, MR::VertId v, float stabilizer )
{
    QuadraticForm3f qf = mp.mesh.quadraticForm( v, mp.region );
    qf.addDistToOrigin( stabilizer );
    return qf;
}

Vector<QuadraticForm3f, VertId> computeFormsAtVertices( const MeshPart & mp, float stabilizer )
{
    MR_TIMER;

    VertBitSet store;
    const VertBitSet & regionVertices = getIncidentVerts( mp.mesh.topology, mp.region, store );

    Vector<QuadraticForm3f, VertId> res( regionVertices.find_last() + 1 );
    BitSetParallelFor( regionVertices, [&]( VertId v )
    {
        res[v] = computeFormAtVertex( mp, v, stabilizer );
    } );

    return res;
}

bool resolveMeshDegenerations( Mesh& mesh, const ResolveMeshDegenSettings & settings )
{
    MR_TIMER;

    DecimateSettings dsettings
    {
        .maxError = settings.maxDeviation,
        .criticalTriAspectRatio = settings.criticalAspectRatio,
        .tinyEdgeLength = settings.tinyEdgeLength,
        .stabilizer = settings.stabilizer,
        .optimizeVertexPos = false, // this decreases probability of normal inversion near mesh degenerations
        .region = settings.region,
        .maxAngleChange = settings.maxAngleChange
    };
    return decimateMesh( mesh, dsettings ).vertsDeleted > 0;
}

bool resolveMeshDegenerations( MR::Mesh& mesh, int, float maxDeviation, float maxAngleChange, float criticalAspectRatio )
{
    ResolveMeshDegenSettings settings
    {
        .maxDeviation = maxDeviation,
        .maxAngleChange = maxAngleChange,
        .criticalAspectRatio = criticalAspectRatio
    };
    return resolveMeshDegenerations( mesh, settings );
}

bool MeshDecimator::initializeQueue_()
{
    MR_TIMER;

    if ( settings_.vertForms )
        pVertForms_ = settings_.vertForms;
    else
        pVertForms_ = &myVertForms_;

    if ( pVertForms_->empty() )
        *pVertForms_ = computeFormsAtVertices( MeshPart{ mesh_, settings_.region }, settings_.stabilizer );

    if ( settings_.progressCallback && !settings_.progressCallback( 0.1f ) )
        return false;

    // initialize regionEdges_ if some edges (out-of-region or touching boundary) cannot be collapsed
    if ( settings_.region )
    {
        // all region edges
        regionEdges_ = getIncidentEdges( mesh_.topology, *settings_.region );
        if ( settings_.edgesToCollapse )
            regionEdges_ &= *settings_.edgesToCollapse;
        if ( !settings_.touchBdVertices )
        {
            // exclude edges touching boundary
            BitSetParallelFor( regionEdges_, [&]( UndirectedEdgeId ue )
            {
                if ( pBdVerts_->test( mesh_.topology.org( ue ) ) ||
                     pBdVerts_->test( mesh_.topology.dest( ue ) ) )
                    regionEdges_.reset( ue );
            } );
        }
    }
    else if ( !settings_.touchBdVertices )
    {
        assert( !settings_.region );
        regionEdges_.clear();
        regionEdges_.resize( mesh_.topology.undirectedEdgeSize(), true );
        // exclude lone edges and edges touching boundary
        BitSetParallelForAll( regionEdges_, [&]( UndirectedEdgeId ue )
        {
            if ( ( settings_.edgesToCollapse && !settings_.edgesToCollapse->test( ue ) ) ||
                 mesh_.topology.isLoneEdge( ue ) ||
                 pBdVerts_->test( mesh_.topology.org( ue ) ) ||
                 pBdVerts_->test( mesh_.topology.dest( ue ) ) )
                regionEdges_.reset( ue );
        } );
    }
    else if ( settings_.edgesToCollapse )
        regionEdges_ = *settings_.edgesToCollapse;

    EdgeMetricCalc calc( *this );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( UndirectedEdgeId{0}, UndirectedEdgeId{mesh_.topology.undirectedEdgeSize()} ), calc );

    if ( settings_.progressCallback && !settings_.progressCallback( 0.2f ) )
        return false;

    presentInQueue_.resize( mesh_.topology.undirectedEdgeSize() );
    for ( const auto & qe : calc.elements() )
        presentInQueue_.set( qe.uedgeId() );
    queue_ = std::priority_queue<QueueElement>{ std::less<QueueElement>(), calc.takeElements() };

    if ( settings_.progressCallback && !settings_.progressCallback( 0.25f ) )
        return false;
    return true;
}

auto MeshDecimator::computeQueueElement_( UndirectedEdgeId ue, bool optimizeVertexPos,
    QuadraticForm3f * outCollapseForm, Vector3f * outCollapsePos ) const -> std::optional<QueueElement>
{
    EdgeId e{ ue };
    const auto o = mesh_.topology.org( e );
    const auto d = mesh_.topology.org( e.sym() );
    const auto po = mesh_.points[o];
    const auto pd = mesh_.points[d];
    const auto vo = (*pVertForms_)[o];
    const auto vd = (*pVertForms_)[d];

    std::optional<QueueElement> res;
    // prepares res; checks flip metric; returns true if the edge does not collpase and function can return
    auto earlyReturn = [&]( float errSq )
    {
        EdgeOp edgeOp = optimizeVertexPos ? EdgeOp::CollapseOptPos : EdgeOp::CollapseEnd;
        if ( settings_.maxAngleChange >= 0 && ( !settings_.notFlippable || !settings_.notFlippable->test( ue ) ) )
        {
            float deviationSqAfterFlip = FLT_MAX;
            if ( !checkDeloneQuadrangleInMesh( mesh_, ue, deloneSettings_, &deviationSqAfterFlip )
                && deviationSqAfterFlip < errSq )
            {
                edgeOp = EdgeOp::Flip;
                errSq = deviationSqAfterFlip;
            }
        }
        if ( ( edgeOp == EdgeOp::Flip || !settings_.adjustCollapse ) && errSq > maxErrorSq_ )
            return true;
        res.emplace();
        res->x.uedgeId = (int)ue;
        res->x.edgeOp = edgeOp;
        res->c = errSq;
        return edgeOp == EdgeOp::Flip;
    };

    if ( settings_.strategy == DecimateStrategy::ShortestEdgeFirst )
    {
        if ( earlyReturn( mesh_.edgeLengthSq( e ) ) )
            return res;
    }

    QuadraticForm3f qf;
    Vector3f pos;
    std::tie( qf, pos ) = sum( vo, po, vd, pd, !optimizeVertexPos );

    if ( settings_.strategy == DecimateStrategy::MinimizeError )
    {
        if ( earlyReturn( qf.c ) )
            return res;
    }

    assert( res && res->x.edgeOp != EdgeOp::Flip );
    if ( settings_.adjustCollapse )
    {
        const auto pos0 = pos;
        settings_.adjustCollapse( ue, res->c, pos );
        if ( res->c > maxErrorSq_ )
            return {};
        if ( outCollapseForm && pos != pos0 )
            qf.c = vo.eval( po - pos ) + vd.eval( pd - pos );
    }

    if ( outCollapseForm )
        *outCollapseForm = qf;
    if ( outCollapsePos )
        *outCollapsePos = pos;

    return res;
}

void MeshDecimator::addInQueueIfMissing_( UndirectedEdgeId ue )
{
    if ( !regionEdges_.empty() && !regionEdges_.test( ue ) )
        return;
    if ( presentInQueue_.test( ue ) )
        return;
    if ( auto qe = computeQueueElement_( ue, settings_.optimizeVertexPos ) )
    {
        queue_.push( *qe );
        presentInQueue_.set( ue );
    }
}

auto MeshDecimator::collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos ) -> CollapseRes
{
    auto & topology = mesh_.topology;
    auto vl = topology.left( edgeToCollapse ).valid()  ? topology.dest( topology.next( edgeToCollapse ) ) : VertId{};
    auto vr = topology.right( edgeToCollapse ).valid() ? topology.dest( topology.prev( edgeToCollapse ) ) : VertId{};

    // cannot collapse internal edge if its left and right faces share another edge
    if ( vl && vr )
    {
        if ( auto pe = topology.prev( edgeToCollapse ); pe != edgeToCollapse && pe == topology.next( edgeToCollapse ) )
            return { .status =  CollapseStatus::SharedEdge };
        if ( auto pe = topology.prev( edgeToCollapse.sym() ); pe != edgeToCollapse.sym() && pe == topology.next( edgeToCollapse.sym() ) )
            return { .status =  CollapseStatus::SharedEdge };
    }
    const bool collapsingFlippable = !settings_.notFlippable || !settings_.notFlippable->test( edgeToCollapse );

    auto vo = topology.org( edgeToCollapse );
    auto vd = topology.dest( edgeToCollapse );
    auto po = mesh_.points[vo];
    auto pd = mesh_.points[vd];
    if ( collapsePos == pd )
    {
        // reverse the edge to have its origin in remaining fixed vertex
        edgeToCollapse = edgeToCollapse.sym();
        std::swap( vo, vd );
        std::swap( vl, vr );
        std::swap( po, pd );
    }

    auto smallShift = [maxBdShiftSq = sqr( settings_.maxBdShift )]( const LineSegm3f & segm, const Vector3f & p )
    {
        return ( closestPointOnLineSegm( p, segm ) - p ).lengthSq() <= maxBdShiftSq;
    };
    if ( ( !vl || !vr ) && settings_.maxBdShift < FLT_MAX )
    {
        if ( !smallShift( mesh_.edgeSegment( edgeToCollapse ), collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from collapsing boundary edge
        if ( !vr )
        {
            if ( !smallShift( LineSegm3f{ mesh_.orgPnt( mesh_.topology.prevLeftBd( edgeToCollapse ) ), collapsePos }, po ) )
                return { .status =  CollapseStatus::PosFarBd }; // origin of collapsing boundary edge is too far from new boundary segment
            if ( !smallShift( LineSegm3f{ mesh_.destPnt( mesh_.topology.nextLeftBd( edgeToCollapse ) ), collapsePos }, pd ) )
                return { .status =  CollapseStatus::PosFarBd }; // destination of collapsing boundary edge is too far from new boundary segment
        }
        if ( !vl )
        {
            if ( !smallShift( LineSegm3f{ mesh_.orgPnt( mesh_.topology.prevLeftBd( edgeToCollapse.sym() ) ), collapsePos }, pd ) )
                return { .status =  CollapseStatus::PosFarBd }; // destination of collapsing boundary edge is too far from new boundary segment
            if ( !smallShift( LineSegm3f{ mesh_.destPnt( mesh_.topology.nextLeftBd( edgeToCollapse.sym() ) ), collapsePos }, po ) )
                return { .status =  CollapseStatus::PosFarBd }; // origin of collapsing boundary edge is too far from new boundary segment
        }
    }

    float maxOldAspectRatio = settings_.maxTriangleAspectRatio;
    float maxNewAspectRatio = 0;
    const float edgeLenSq = ( po - pd ).lengthSq();
    float maxOldEdgeLenSq = std::max( sqr( settings_.maxEdgeLen ), edgeLenSq );
    float maxNewEdgeLenSq = 0;

    bool normalFlip = false; // at least one triangle flips its normal or a degenerate triangle becomes not-degenerate
    originNeis_.clear();
    triDblAreas_.clear();
    Vector3d sumDblArea_;
    EdgeId oBdEdge; // a boundary edge !right(e) incident to org( edgeToCollapse )
    for ( EdgeId e : orgRing0( topology, edgeToCollapse ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == vd )
            return { .status =  CollapseStatus::MultipleEdge }; // multiple edge found
        if ( eDest != vl && eDest != vr )
            originNeis_.push_back( eDest );

        const auto pDest = mesh_.points[eDest];
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( po - pDest ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - pDest ).lengthSq() );
        if ( !topology.left( e ) )
        {
            oBdEdge = topology.next( e );
            assert( topology.isLeftBdEdge( oBdEdge ) );
            continue;
        }
        if ( collapsingFlippable && settings_.notFlippable && settings_.notFlippable->test( e ) )
            return { .status =  CollapseStatus::Flippable }; // cannot collapse a flippable edge incident to a not-flippable edge

        const auto pDest2 = mesh_.destPnt( topology.next( e ) );
        if ( eDest != vr )
        {
            auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
            if ( !normalFlip )
            {
                const auto oldA = cross( pDest - po, pDest2 - po );
                if ( dot( da, oldA ) <= 0 )
                    normalFlip = true;
            }
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            if ( triAspect >= settings_.criticalTriAspectRatio )
                triDblAreas_.back() = Vector3f{}; //cannot trust direction of degenerate triangles
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( po, pDest, pDest2 ) );
    }
    if ( oBdEdge
        && !smallShift( LineSegm3f{ po, mesh_.destPnt( oBdEdge ) }, collapsePos )
        && !smallShift( LineSegm3f{ po, mesh_.orgPnt( topology.prevLeftBd( oBdEdge ) ) }, collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from both existed boundary edges
    std::sort( originNeis_.begin(), originNeis_.end() );

    EdgeId dBdEdge; // a boundary edge !right(e) incident to dest( edgeToCollapse )
    for ( EdgeId e : orgRing0( topology, edgeToCollapse.sym() ) )
    {
        const auto eDest = topology.dest( e );
        assert ( eDest != vo );
        if ( std::binary_search( originNeis_.begin(), originNeis_.end(), eDest ) )
            return { .status =  CollapseStatus::MultipleEdge }; // to prevent appearance of multiple edges

        const auto pDest = mesh_.points[eDest];
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( pd - pDest ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - pDest ).lengthSq() );
        if ( !topology.left( e ) )
        {
            dBdEdge = topology.next( e );
            assert( topology.isLeftBdEdge( dBdEdge ) );
            continue;
        }
        if ( collapsingFlippable && settings_.notFlippable && settings_.notFlippable->test( e ) )
            return { .status =  CollapseStatus::Flippable }; // cannot collapse a flippable edge incident to a not-flippable edge

        const auto pDest2 = mesh_.destPnt( topology.next( e ) );
        if ( eDest != vl )
        {
            auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
            if ( !normalFlip )
            {
                const auto oldA = cross( pDest - pd, pDest2 - pd );
                if ( dot( da, oldA ) <= 0 )
                    normalFlip = true;
            }
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            if ( triAspect >= settings_.criticalTriAspectRatio )
                triDblAreas_.back() = Vector3f{}; //cannot trust direction of degenerate triangles
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( pd, pDest, pDest2 ) );
    }
    if ( dBdEdge
        && !smallShift( LineSegm3f{ pd, mesh_.destPnt( dBdEdge ) }, collapsePos )
        && !smallShift( LineSegm3f{ pd, mesh_.orgPnt( topology.prevLeftBd( dBdEdge ) ) }, collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from both existed boundary edges

    if ( vl && vr && oBdEdge && dBdEdge )
        return { .status =  CollapseStatus::TouchBd }; // prohibit collapse of an inner edge if it brings two boundaries in touch

    // special treatment for short edges on the last stage when collapse position is equal to one of edge's ends
    const bool tinyEdge = ( settings_.tinyEdgeLength >= 0 && ( po == collapsePos || pd == collapsePos ) )
        ? edgeLenSq <= sqr( settings_.tinyEdgeLength ) : false;
    if ( !tinyEdge && maxNewAspectRatio > maxOldAspectRatio && maxOldAspectRatio <= settings_.criticalTriAspectRatio )
        return { .status =  CollapseStatus::TriAspect }; // new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings

    if ( maxNewEdgeLenSq > maxOldEdgeLenSq )
        return { .status =  CollapseStatus::LongEdge }; // new edge would be longer than all of old edges and longer than allowed in settings

    // if at least one triangle normal flips, checks that all new normals are consistent
    if ( normalFlip && ( ( po != pd ) || ( po != collapsePos ) ) )
    {
        auto n = Vector3f{ sumDblArea_.normalized() };
        for ( const auto da : triDblAreas_ )
            if ( dot( da, n ) < 0 )
                return { .status =  CollapseStatus::NormalFlip };
    }

    if ( settings_.preCollapse && !settings_.preCollapse( edgeToCollapse, collapsePos ) )
        return { .status =  CollapseStatus::User }; // user prohibits the collapse

    ++res_.vertsDeleted;
    if ( vl )
        ++res_.facesDeleted;
    if ( vr )
        ++res_.facesDeleted;

    mesh_.points[vo] = collapsePos;
    if ( settings_.region )
    {
        if ( auto l = topology.left( edgeToCollapse ) )
            settings_.region->reset( l );
        if ( auto r = topology.left( edgeToCollapse.sym() ) )
            settings_.region->reset( r );
    }
    auto eo = collapseEdge( topology, edgeToCollapse, settings_.notFlippable, settings_.onEdgeDel );
    const auto remainingVertex = eo ? vo : VertId{};
    return { .v = remainingVertex };
}

static void optionalPackMesh( Mesh & mesh, const DecimateSettings & settings )
{
    if ( !settings.packMesh )
        return;

    MR_TIMER
    FaceMap fmap;
    VertMap vmap;
    WholeEdgeMap emap;
    mesh.pack(
        settings.region ? &fmap : nullptr,
        settings.vertForms ? &vmap : nullptr,
        settings.notFlippable ? &emap : nullptr );

    if ( settings.region )
        *settings.region = settings.region->getMapping( fmap, mesh.topology.faceSize() );

    if ( settings.vertForms )
    {
        auto & vertForms = *settings.vertForms;
        for ( VertId oldV{ 0 }; oldV < vmap.size(); ++oldV )
            if ( auto newV = vmap[oldV] )
                if ( newV < oldV )
                    vertForms[newV] = vertForms[oldV];
        vertForms.resize( mesh.topology.vertSize() );
    }

    if ( settings.notFlippable )
        *settings.notFlippable = settings.notFlippable->getMapping( [&emap]( UndirectedEdgeId i ) { return emap[i].undirected(); }, mesh.topology.undirectedEdgeSize() );
}

DecimateResult MeshDecimator::run()
{
    MR_TIMER

    if ( settings_.bdVerts )
        pBdVerts_ = settings_.bdVerts;
    else
    {
        pBdVerts_ = &myBdVerts_;
        if ( !settings_.touchBdVertices )
            myBdVerts_ = getBoundaryVerts( mesh_.topology, settings_.region );
    }

    if ( !initializeQueue_() )
        return res_;

    res_.errorIntroduced = settings_.maxError;
    int lastProgressFacesDeleted = 0;
    const int maxFacesDeleted = std::min(
        settings_.region ? (int)settings_.region->count() : mesh_.topology.numValidFaces(), settings_.maxDeletedFaces );
    while ( !queue_.empty() )
    {
        const auto topQE = queue_.top();
        const auto ue = topQE.uedgeId();
        assert( presentInQueue_.test( ue ) );
        queue_.pop();
        if ( res_.facesDeleted >= settings_.maxDeletedFaces || res_.vertsDeleted >= settings_.maxDeletedVertices )
        {
            res_.errorIntroduced = std::sqrt( topQE.c );
            break;
        }

        if ( settings_.progressCallback && res_.facesDeleted >= 1000 + lastProgressFacesDeleted ) 
        {
            if ( !settings_.progressCallback( 0.25f + 0.75f * res_.facesDeleted / maxFacesDeleted ) )
                return res_;
            lastProgressFacesDeleted = res_.facesDeleted;
        }

        if ( mesh_.topology.isLoneEdge( ue ) )
        {
            // edge has been deleted by this moment
            presentInQueue_.reset( ue );
            continue;
        }

        QuadraticForm3f collapseForm;
        Vector3f collapsePos;
        auto qe = computeQueueElement_( ue, topQE.x.edgeOp == EdgeOp::CollapseOptPos, &collapseForm, &collapsePos );
        if ( !qe )
        {
            presentInQueue_.reset( ue );
            continue;
        }

        if ( qe->c > topQE.c )
        {
            queue_.push( *qe );
            continue;
        }

        presentInQueue_.reset( ue );
        if ( qe->x.edgeOp == EdgeOp::Flip )
        {
            EdgeId e = ue;
            mesh_.topology.flipEdge( e );
            assert( mesh_.topology.left( e ) );
            assert( mesh_.topology.right( e ) );
            addInQueueIfMissing_( e.undirected() );
            addInQueueIfMissing_( mesh_.topology.prev( e ).undirected() );
            addInQueueIfMissing_( mesh_.topology.next( e ).undirected() );
            addInQueueIfMissing_( mesh_.topology.prev( e.sym() ).undirected() );
            addInQueueIfMissing_( mesh_.topology.next( e.sym() ).undirected() );
        }
        else
        {
            // edge collapse
            auto collapseRes = collapse_( ue, collapsePos );
            if ( !collapseRes.v )
            {
                if ( topQE.x.edgeOp == EdgeOp::CollapseOptPos &&
                    // collapse failed due to a geometric criterion (e.g. bad collapse position)
                    ( collapseRes.status == CollapseStatus::TriAspect || collapseRes.status == CollapseStatus::NormalFlip ||
                      collapseRes.status == CollapseStatus::PosFarBd || collapseRes.status == CollapseStatus::LongEdge ) )
                {
                    qe = computeQueueElement_( ue, false );
                    if ( qe )
                    {
                        queue_.push( *qe );
                        presentInQueue_.set( ue );
                    }
                }
                continue;
            }
            assert( collapseRes.status == CollapseStatus::Done );

            (*pVertForms_)[collapseRes.v] = collapseForm;

            for ( EdgeId e : orgRing( mesh_.topology, collapseRes.v ) )
            {
                addInQueueIfMissing_( e.undirected() );
                if ( mesh_.topology.left( e ) )
                    addInQueueIfMissing_( mesh_.topology.prev( e.sym() ).undirected() );
            }
        }
    }

    if ( settings_.progressCallback && !settings_.progressCallback( 1.0f ) )
        return res_;

    optionalPackMesh( mesh_, settings_ );
    res_.cancelled = false;
    return res_;
}

static DecimateResult decimateMeshSerial( Mesh & mesh, const DecimateSettings & settings )
{
    MR_TIMER
    if ( settings.maxDeletedFaces <= 0 || settings.maxDeletedVertices <= 0 )
    {
        DecimateResult res;
        res.cancelled = false;
        return res;
    }
    MR_WRITER( mesh );
    MeshDecimator md( mesh, settings );
    return md.run();
}

FaceBitSet getSubdividePart( const FaceBitSet & valids, size_t subdivideParts, size_t myPart )
{
    assert( subdivideParts > 1 );
    const auto sz = subdivideParts;
    assert( myPart < sz );
    const auto facesPerPart = ( valids.size() / ( sz * FaceBitSet::bits_per_block ) ) * FaceBitSet::bits_per_block;

    const auto fromFace = myPart * facesPerPart;
    const auto toFace = ( myPart + 1 ) < sz ? ( myPart + 1 ) * facesPerPart : valids.size();
    FaceBitSet res( toFace );
    res.set( FaceId{ fromFace }, toFace - fromFace, true );
    res &= valids;
    return res;
}

static DecimateResult decimateMeshParallelInplace( MR::Mesh & mesh, const DecimateSettings & settings )
{
    MR_TIMER
    assert( settings.subdivideParts > 1 );
    const auto sz = settings.subdivideParts;
    assert( !settings.partFaces || settings.partFaces->size() == sz );

    DecimateResult res;

    MR_WRITER( mesh );
    if ( settings.progressCallback && !settings.progressCallback( 0 ) )
        return res;

    struct alignas(64) Parts
    {
        /// region faces of the subdivision part
        FaceBitSet region;

        /// vertices to be fixed during subdivision of individual parts
        VertBitSet bdVerts;

        DecimateResult decimRes;
    };
    std::vector<Parts> parts( sz );

    // determine faces for each part
    ParallelFor( parts, [&]( size_t i )
    {
        if ( settings.partFaces )
            parts[i].region = std::move( (*settings.partFaces)[i] );
        else
            parts[i].region = getSubdividePart( mesh.topology.getValidFaces(), settings.subdivideParts, i );
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.03f ) )
        return res;

    // determine edges in between the parts
    UndirectedEdgeBitSet stableEdges( mesh.topology.undirectedEdgeSize() );
    BitSetParallelForAll( stableEdges, [&]( UndirectedEdgeId ue )
    {
        FaceId l = mesh.topology.left( ue );
        FaceId r = mesh.topology.right( ue );
        if ( !l || !r )
            return;
        for ( size_t i = 0; i < sz; ++i )
        {
            if ( parts[i].region.test( l ) != parts[i].region.test( r ) )
            {
                stableEdges.set( ue );
                break;
            }
        }
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.07f ) )
        return res;

    // limit each part to region, find boundary vertices
    ParallelFor( parts, [&]( size_t i )
    {
        auto & faces = parts[i].region;
        if ( settings.touchBdVertices )
        {
            /// vertices on the boundary of subdivision,
            /// if a vertex is only on hole's boundary or region's boundary but not on subdivision boundary, it is not here
            parts[i].bdVerts = getRegionBoundaryVerts( mesh.topology, faces );
            if ( settings.region )
                faces &= *settings.region;
        }
        else
        {
            if ( settings.region )
                faces &= *settings.region;
            /// all boundary vertices of subdivision faces, including hole boundaries
            parts[i].bdVerts = getBoundaryVerts( mesh.topology, &faces );
        }
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.14f ) )
        return res;

    mesh.topology.preferEdges( stableEdges );
    if ( settings.progressCallback && !settings.progressCallback( 0.16f ) )
        return res;

    // compute quadratic form in each vertex
    Vector<QuadraticForm3f, VertId> mVertForms;
    if ( settings.vertForms )
        mVertForms = std::move( *settings.vertForms );
    if ( mVertForms.empty() )
        mVertForms = computeFormsAtVertices( MeshPart{ mesh, settings.region }, settings.stabilizer );
    if ( settings.progressCallback && !settings.progressCallback( 0.2f ) )
        return res;

    mesh.topology.stopUpdatingValids();
    const auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> cancelled{ false };
    std::atomic<int> finishedParts{ 0 };
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ), [&]( const tbb::blocked_range<size_t>& range )
    {
        const bool reportProgressFromThisThread = settings.progressCallback && mainThreadId == std::this_thread::get_id();
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto reportThreadProgress = [&]( float p )
            {
                if ( cancelled.load( std::memory_order_relaxed ) )
                    return false;
                if ( reportProgressFromThisThread && !settings.progressCallback( 0.2f + 0.65f * ( finishedParts.load( std::memory_order_relaxed ) + p ) / sz ) )
                {
                    cancelled.store( true, std::memory_order_relaxed );
                    return false;
                }
                return true;
            };
            if ( !reportThreadProgress( 0 ) )
                break;

            DecimateSettings subSeqSettings = settings;
            subSeqSettings.maxDeletedVertices = settings.maxDeletedVertices / ( sz + 1 );
            subSeqSettings.maxDeletedFaces = settings.maxDeletedFaces / ( sz + 1 );
            if ( settings.minFacesInPart > 0 )
            {
                int startFaces = (int)parts[i].region.count();
                if ( startFaces > settings.minFacesInPart )
                    subSeqSettings.maxDeletedFaces = std::min( subSeqSettings.maxDeletedFaces, startFaces - settings.minFacesInPart );
                else
                    subSeqSettings.maxDeletedFaces = 0;
            }
            subSeqSettings.packMesh = false;
            subSeqSettings.vertForms = &mVertForms;
            subSeqSettings.touchBdVertices = false;
            // after mesh.topology.stopUpdatingValids(), seq-subdivider cannot find bdVerts by itself
            subSeqSettings.bdVerts = &parts[i].bdVerts;
            subSeqSettings.region = &parts[i].region;
            if ( reportProgressFromThisThread )
                subSeqSettings.progressCallback = [reportThreadProgress]( float p ) { return reportThreadProgress( p ); };
            else if ( settings.progressCallback )
                subSeqSettings.progressCallback = [&cancelled]( float ) { return !cancelled.load( std::memory_order_relaxed ); };
            parts[i].decimRes = decimateMeshSerial( mesh, subSeqSettings );

            if ( parts[i].decimRes.cancelled || !reportThreadProgress( 1 ) )
                break;
            finishedParts.fetch_add( 1, std::memory_order_relaxed );
        }
    } );

    if ( settings.region )
    {
        // not ParallelFor to allow for subdivisions not aligned to BitSet's block boundaries
        *settings.region = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, sz ), FaceBitSet{},
            [&] ( const tbb::blocked_range<size_t> range, FaceBitSet cur )
            {
                for ( size_t i = range.begin(); i < range.end(); i++ )
                    cur |= parts[i].region;
                return cur;
            },
            [&] ( FaceBitSet a, const FaceBitSet& b )
            {
                a |= b;
                return a;
            } );
    }

    // restore valids computation before return even if the operation was canceled
    mesh.topology.computeValidsFromEdges();

    if ( cancelled.load( std::memory_order_relaxed ) || ( settings.progressCallback && !settings.progressCallback( 0.9f ) ) )
        return res;

    if ( settings.partFaces )
    {
        assert( !settings.packMesh ); // otherwise, returned partFaces will contain wrong distribution of faces
        assert( !settings.decimateBetweenParts ); // otherwise, returned partFaces will contain some deleted faces, and some faces can actually be in-between original parts
        for ( int i = 0; i < sz; ++i )
            (*settings.partFaces)[i] = std::move( parts[i].region );
    }

    DecimateSettings seqSettings = settings;
    for ( const auto & submesh : parts )
    {
        seqSettings.maxDeletedFaces -= submesh.decimRes.facesDeleted;
        seqSettings.maxDeletedVertices -= submesh.decimRes.vertsDeleted;
    }
    seqSettings.vertForms = &mVertForms;
    seqSettings.progressCallback = subprogress( settings.progressCallback, 0.9f, 1.0f );
    if ( settings.decimateBetweenParts )
    {
        res = decimateMeshSerial( mesh, seqSettings );
    }
    else
    {
        optionalPackMesh( mesh, seqSettings );
        res.cancelled = false;
    }

    // update res from submesh decimations
    for ( const auto & submesh : parts )
    {
        res.facesDeleted += submesh.decimRes.facesDeleted;
        res.vertsDeleted += submesh.decimRes.vertsDeleted;
    }

    if ( settings.vertForms )
        *settings.vertForms = std::move( mVertForms );
    return res;
}

DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings )
{
    if ( settings.subdivideParts > 1 )
        return decimateMeshParallelInplace( mesh, settings );
    else
        return decimateMeshSerial( mesh, settings );
}

bool remesh( MR::Mesh& mesh, const RemeshSettings & settings )
{
    MR_TIMER
    if ( settings.progressCallback && !settings.progressCallback( 0.0f ) )
        return false;
    if ( settings.targetEdgeLen <= 0 )
    {
        assert( false );
        return false;
    }
    if ( settings.region && !settings.region->any() )
    {
        assert( false );
        return false;
    }

    MR_WRITER( mesh );

    SubdivideSettings subs;
    subs.maxEdgeLen = settings.targetEdgeLen;
    subs.maxEdgeSplits = settings.maxEdgeSplits;
    subs.maxAngleChangeAfterFlip = settings.maxAngleChangeAfterFlip;
    subs.smoothMode = settings.useCurvature;
    subs.region = settings.region;
    subs.notFlippable = settings.notFlippable;
    subs.projectOnOriginalMesh = settings.projectOnOriginalMesh;
    subs.onEdgeSplit = settings.onEdgeSplit;
    subs.progressCallback = subprogress( settings.progressCallback, 0.0f, 0.5f );
    subdivideMesh( mesh, subs );
    if ( !reportProgress( settings.progressCallback, 0.5f ) )
        return false;

    // compute target number of triangles to get desired average edge length
    const auto regionArea = mesh.area( settings.region );
    const auto targetTriArea = sqr( settings.targetEdgeLen ) * ( sqrt( 3.0f ) / 4 ); // for equilateral triangle
    const auto targetNumTri = int( regionArea / targetTriArea );
    const auto currNumTri = settings.region ? (int)settings.region->count() : mesh.topology.numValidFaces();

    if ( currNumTri > targetNumTri )
    {
        DecimateSettings decs;
        decs.strategy = DecimateStrategy::ShortestEdgeFirst;
        decs.maxError = FLT_MAX;
        decs.maxEdgeLen = 1.5f * settings.targetEdgeLen; // not to over-decimate when there are many notFlippable edges in the region
        decs.maxDeletedFaces = currNumTri - targetNumTri;
        decs.maxBdShift = settings.maxBdShift;
        decs.region = settings.region;
        decs.notFlippable = settings.notFlippable;
        decs.packMesh = settings.packMesh;
        decs.progressCallback = subprogress( settings.progressCallback, 0.5f, 0.95f );
        decs.preCollapse = settings.preCollapse;
        decs.onEdgeDel = settings.onEdgeDel;
        decs.stabilizer = 1e-6f;
        // it was a bad idea to make decs.stabilizer = settings.targetEdgeLen;
        // yes, it increased the uniformity of vertices, but shifted boundary vertices after edge collapse inside
        decimateMesh( mesh, decs );
        if ( !reportProgress( settings.progressCallback, 0.95f ) )
            return false;
    }

    if ( settings.finalRelaxIters > 0 )
    {
        // even if region is not given, we need to exclude mesh boundary from relaxation
        VertBitSet innerVerts = getInnerVerts( mesh.topology, settings.region );
        if ( settings.notFlippable )
            innerVerts -= getIncidentVerts( mesh.topology, *settings.notFlippable );
        MeshEqualizeTriAreasParams rp;
        rp.region = &innerVerts;
        rp.hardSmoothTetrahedrons = true;
        rp.noShrinkage = settings.finalRelaxNoShrinkage;
        DeloneSettings ds;
        ds.maxAngleChange = settings.maxAngleChangeAfterFlip;
        ds.region = settings.region;
        ds.notFlippable = settings.notFlippable;
        auto sp = subprogress( settings.progressCallback, 0.95f, 1.0f );
        for ( int i = 0; i < settings.finalRelaxIters; ++i )
        {
            if ( !reportProgress( sp, float( i ) / settings.finalRelaxIters ) )
                return false;
            equalizeTriAreas( mesh, rp );
            makeDeloneEdgeFlips( mesh, ds );
        }
    }

    return reportProgress( settings.progressCallback, 1.0f );
}

// check if Decimator updates region
TEST( MRMesh, MeshDecimate )
{
    Mesh meshCylinder = makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

    // select all faces
    MR::FaceBitSet regionForDecimation = meshCylinder.topology.getValidFaces();
    MR::FaceBitSet regionSaved(regionForDecimation);

    // setup and run decimator
    DecimateSettings decimateSettings;
    decimateSettings.region = &regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0f;

    auto decimateResults = decimateMesh(meshCylinder, decimateSettings);

    // compare regions and deleted vertices and faces
    ASSERT_NE(regionSaved, regionForDecimation);
    ASSERT_GT(decimateResults.vertsDeleted, 0);
    ASSERT_GT(decimateResults.facesDeleted, 0);
}

} //namespace MR
