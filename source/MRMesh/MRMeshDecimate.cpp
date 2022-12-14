#include "MRMeshDecimate.h"
#include "MRMesh.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRCylinder.h"
#include "MRGTest.h"
#include "MRMeshDelone.h"
#include "MRMeshSubdivide.h"
#include "MRPch/MRTBB.h"
#include <queue>

namespace MR
{

// collapses given edge and deletes
// 1) faces: left( e ) and right( e );
// 2) vertex org( e )/dest( e ) if given edge was their only edge, otherwise only dest( e );
// 3) edges: e, next( e.sym() ), prev( e.sym() );
// returns prev( e ) if it is valid
EdgeId collapseEdge( MeshTopology & topology, const EdgeId e )
{
    topology.setLeft( e, FaceId() );
    topology.setLeft( e.sym(), FaceId() );

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
        }
    }

    return ePrev != e ? ePrev : EdgeId();
}

class MeshDecimator
{
public:
    MeshDecimator( Mesh & mesh, const DecimateSettings & settings );
    DecimateResult run();

    // returns true if the collapse of given edge is permitted by the region and settings
    bool isInRegion( EdgeId e ) const;

private: 
    Mesh & mesh_;
    const DecimateSettings & settings_;
    const DeloneSettings deloneSettings_;
    const float maxErrorSq_;
    Vector<QuadraticForm3f, VertId> myVertForms_;
    Vector<QuadraticForm3f, VertId> * pVertForms_ = nullptr;
    VertBitSet bdVerts_;

    struct QueueElement
    {
        float c = 0;
        struct X {
            unsigned int flip : 1 = 0;
            unsigned int uedgeId : 31 = 0;
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
    std::optional<QueueElement> computeQueueElement_( UndirectedEdgeId ue, QuadraticForm3f * outCollapseForm = nullptr, Vector3f * outCollapsePos = nullptr ) const;
    void addInQueueIfMissing_( UndirectedEdgeId ue );
    VertId collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos );
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

bool MeshDecimator::isInRegion( EdgeId e ) const
{
    if ( !mesh_.topology.isInnerOrBdEdge( e, settings_.region ) )
        return false;
    if ( !settings_.touchBdVertices )
    {
        if ( bdVerts_.test( mesh_.topology.org( e ) ) ||
             bdVerts_.test( mesh_.topology.dest( e ) ) )
            return false;
    }
    return true;
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
        const auto & mesh = decimator_.mesh_;
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue ) 
        {
            EdgeId e{ ue };
            if ( mesh.topology.isLoneEdge( e ) )
                continue;
            if ( !decimator_.isInRegion( e ) )
                continue;
            if ( auto qe = decimator_.computeQueueElement_( ue ) )
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
        .stabilizer = settings.stabilizer,
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

auto MeshDecimator::computeQueueElement_( UndirectedEdgeId ue, QuadraticForm3f * outCollapseForm, Vector3f * outCollapsePos ) const -> std::optional<QueueElement>
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
        bool flip = false;
        if ( settings_.maxAngleChange >= 0 )
        {
            float deviationSqAfterFlip = FLT_MAX;
            if ( !checkDeloneQuadrangleInMesh( mesh_, ue, deloneSettings_, &deviationSqAfterFlip )
                && deviationSqAfterFlip < errSq )
            {
                flip = true;
                errSq = deviationSqAfterFlip;
            }
        }
        if ( ( flip || !settings_.adjustCollapse ) && errSq > maxErrorSq_ )
            return true;
        res.emplace();
        res->x.uedgeId = (int)ue;
        res->x.flip = flip;
        res->c = errSq;
        return flip;
    };

    if ( settings_.strategy == DecimateStrategy::ShortestEdgeFirst )
    {
        if ( earlyReturn( mesh_.edgeLengthSq( e ) ) )
            return res;
    }

    QuadraticForm3f qf;
    Vector3f pos;
    std::tie( qf, pos ) = sum( vo, po, vd, pd, !settings_.optimizeVertexPos );

    if ( settings_.strategy == DecimateStrategy::MinimizeError )
    {
        if ( earlyReturn( qf.c ) )
            return res;
    }

    assert( res && !res->x.flip );
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
    EdgeId e{ ue };
    if ( !isInRegion( e ) )
        return;
    if ( presentInQueue_.test_set( ue ) )
        return;
    if ( auto qe = computeQueueElement_( ue ) )
        queue_.push( *qe );
}

VertId MeshDecimator::collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos )
{
    auto & topology = mesh_.topology;
    // cannot collapse edge if its left and right faces share another edge
    if ( auto pe = topology.prev( edgeToCollapse ); pe != edgeToCollapse && pe == topology.next( edgeToCollapse ) )
        return {};
    if ( auto pe = topology.prev( edgeToCollapse.sym() ); pe != edgeToCollapse.sym() && pe == topology.next( edgeToCollapse.sym() ) )
        return {};

    auto vo = topology.org( edgeToCollapse );
    auto vd = topology.dest( edgeToCollapse );
    auto po = mesh_.points[vo];
    auto pd = mesh_.points[vd];
    if ( !settings_.optimizeVertexPos && collapsePos == pd )
    {
        // reverse the edge to have its origin in remaining fixed vertex
        edgeToCollapse = edgeToCollapse.sym();
        std::swap( vo, vd );
        std::swap( po, pd );
    }
    auto vl = topology.left( edgeToCollapse ).valid()  ? topology.dest( topology.next( edgeToCollapse ) ) : VertId{};
    auto vr = topology.right( edgeToCollapse ).valid() ? topology.dest( topology.prev( edgeToCollapse ) ) : VertId{};

    float maxOldAspectRatio = settings_.maxTriangleAspectRatio;
    float maxNewAspectRatio = 0;
    float maxOldEdgeLenSq = std::max( sqr( settings_.maxEdgeLen ), ( po - pd ).lengthSq() );
    float maxNewEdgeLenSq = 0;

    originNeis_.clear();
    triDblAreas_.clear();
    Vector3d sumDblArea_;
    for ( EdgeId e : orgRing0( topology, edgeToCollapse ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == vd )
            return {}; // multiple edge found
        if ( eDest != vl && eDest != vr )
            originNeis_.push_back( eDest );

        const auto pDest = mesh_.points[eDest];
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( po - pDest ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - pDest ).lengthSq() );

        const auto pDest2 = mesh_.destPnt( topology.next( e ) );
        if ( eDest != vr && topology.left( e ) )
        {
            auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            maxNewAspectRatio = std::max( maxNewAspectRatio, triangleAspectRatio( collapsePos, pDest, pDest2 ) );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( po, pDest, pDest2 ) );
    }
    std::sort( originNeis_.begin(), originNeis_.end() );

    for ( EdgeId e : orgRing0( topology, edgeToCollapse.sym() ) )
    {
        const auto eDest = topology.dest( e );
        assert ( eDest != vo );
        if ( std::binary_search( originNeis_.begin(), originNeis_.end(), eDest ) )
            return {}; // to prevent appearance of multiple edges

        const auto pDest = mesh_.points[eDest];
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( pd - pDest ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - pDest ).lengthSq() );

        const auto pDest2 = mesh_.destPnt( topology.next( e ) );
        if ( eDest != vl && topology.left( e ) )
        {
            auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            if ( triAspect >= settings_.criticalTriAspectRatio )
                triDblAreas_.back() = Vector3f{}; //cannot trust direction of degenerate triangles
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( pd, pDest, pDest2 ) );
    }

    if ( maxNewAspectRatio > maxOldAspectRatio )
        return {}; // new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings

    if ( maxNewEdgeLenSq > maxOldEdgeLenSq )
        return {}; // new edge would be longer than all of old edges and longer than allowed in settings

    // checks that all new normals are consistent (do not check for degenerate edges)
    if ( ( po != pd ) || ( po != collapsePos ) )
    {
        auto n = Vector3f{ sumDblArea_.normalized() };
        for ( const auto da : triDblAreas_ )
            if ( dot( da, n ) < 0 )
                return {};
    }

    if ( settings_.preCollapse && !settings_.preCollapse( edgeToCollapse, collapsePos ) )
        return {}; // user prohibits the collapse

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
    auto eo = collapseEdge( topology, edgeToCollapse );
    const auto remainingVertex = eo ? vo : VertId{};
    if ( !settings_.touchBdVertices && remainingVertex )
       bdVerts_.set( remainingVertex, mesh_.topology.isBdVertex( remainingVertex, settings_.region ) );

    return remainingVertex;
}

DecimateResult MeshDecimator::run()
{
    MR_TIMER;

    if ( !settings_.touchBdVertices )
    {
        bdVerts_.clear();
        bdVerts_.resize( mesh_.topology.vertSize() );
        BitSetParallelForAll( bdVerts_, [&]( VertId v )
        {
            if ( mesh_.topology.isBdVertex( v, settings_.region ) )
                bdVerts_.set( v );
        } );
    }

    if ( !initializeQueue_() )
        return res_;

    res_.errorIntroduced = settings_.maxError;
    int lastProgressFacesDeleted = 0;
    const int maxFacesDeleted = std::min(
        settings_.region ? (int)settings_.region->count() : mesh_.topology.numValidFaces(), settings_.maxDeletedFaces );
    while ( !queue_.empty() )
    {
        auto topQE = queue_.top();
        assert( presentInQueue_.test( topQE.uedgeId() ) );
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

        if ( mesh_.topology.isLoneEdge( topQE.uedgeId() ) )
        {
            // edge has been deleted by this moment
            presentInQueue_.reset( topQE.uedgeId() );
            continue;
        }

        QuadraticForm3f collapseForm;
        Vector3f collapsePos;
        auto qe = computeQueueElement_( topQE.uedgeId(), &collapseForm, &collapsePos );
        if ( !qe )
        {
            presentInQueue_.reset( topQE.uedgeId() );
            continue;
        }

        if ( qe->c > topQE.c )
        {
            queue_.push( *qe );
            continue;
        }

        if ( qe->x.flip )
        {
            EdgeId e = topQE.uedgeId();
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
            presentInQueue_.reset( topQE.uedgeId() );
            VertId collapseVert = collapse_( topQE.uedgeId(), collapsePos );
            if ( !collapseVert )
                continue;

            (*pVertForms_)[collapseVert] = collapseForm;

            for ( EdgeId e : orgRing( mesh_.topology, collapseVert ) )
            {
                addInQueueIfMissing_( e.undirected() );
                if ( mesh_.topology.left( e ) )
                    addInQueueIfMissing_( mesh_.topology.prev( e.sym() ).undirected() );
            }
        }
    }

    if ( settings_.packMesh )
    {
        FaceMap fmap;
        VertMap vmap;
        mesh_.pack( 
            settings_.region ? &fmap : nullptr,
            settings_.vertForms ? &vmap : nullptr );

        if ( settings_.region )
            *settings_.region = settings_.region->getMapping( fmap, mesh_.topology.faceSize() );

        if ( settings_.vertForms )
        {
            for ( VertId oldV{ 0 }; oldV < vmap.size(); ++oldV )
                if ( auto newV = vmap[oldV] )
                    if ( newV < oldV )
                        (*pVertForms_)[newV] = (*pVertForms_)[oldV];
            pVertForms_->resize( mesh_.topology.vertSize() );
        }
    }

    res_.cancelled = false;
    return res_;
}

DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings )
{
    MR_TIMER;
    MR_WRITER( mesh );
    MeshDecimator md( mesh, settings );
    return md.run();
}

bool remesh( MR::Mesh& mesh, const RemeshSettings & settings )
{
    MR_TIMER;
    MR_WRITER( mesh );

    if ( settings.progressCallback && !settings.progressCallback( 0.0f ) )
        return false;

    SubdivideSettings subs;
    subs.maxEdgeLen = 2 * settings.targetEdgeLen;
    subs.maxEdgeSplits = 10'000'000;
    subs.maxAngleChangeAfterFlip = settings.maxAngleChangeAfterFlip;
    subs.useCurvature = settings.useCurvature;
    subs.region = settings.region;
    subs.notFlippable = settings.notFlippable;
    subs.onEdgeSplit = settings.onEdgeSplit;
    if ( settings.progressCallback )
        subs.progressCallback = [settings] ( float arg ) { return settings.progressCallback( arg * 0.5f ); };
    subdivideMesh( mesh, subs );

    if ( settings.progressCallback && !settings.progressCallback( 0.5f ) )
        return false;

    DecimateSettings decs;
    decs.strategy = DecimateStrategy::ShortestEdgeFirst;
    decs.maxError = settings.targetEdgeLen / 2;
    decs.region = settings.region;
    decs.packMesh = settings.packMesh;
    if ( settings.progressCallback )
        decs.progressCallback = [settings] ( float arg ) { return settings.progressCallback( 0.5f + arg * 0.5f ); };
    decimateMesh( mesh, decs );

    if ( settings.progressCallback && !settings.progressCallback( 1.0f ) )
        return false;

    return true;
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
