#include "MRMultiwayICP.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPointToPointAligningTransform.h"
#include "MRPointToPlaneAligningTransform.h"
#include "MRMultiwayAligningTransform.h"
#include "MRBitSetParallelFor.h"
#include "MRAABBTreeObjects.h"
#include "MRAABBTreeObjects.h"
#include <algorithm>

namespace MR
{

// sequential cascade indexer
class SeqCascade : public IICPTreeIndexer
{
public:
    SeqCascade( int numObjects, int maxGroupSize ) :
        numObjects_{ numObjects },
        maxGroupSize_{ maxGroupSize }
    {}

    virtual bool fromSameNode( ICPLayer, ICPElementId eI, ICPElementId eJ ) const override
    {
        return ( eI.get() / maxGroupSize_ ) == ( eJ.get() / maxGroupSize_ );
    }

    virtual ObjBitSet getElementLeaves( ICPLayer l, ICPElementId eId ) const override
    {
        size_t numLeaves = 1;
        for ( int i = 0; i < l; ++i )
            numLeaves *= maxGroupSize_;
        ObjId first = ObjId( eId * numLeaves );
        ObjId last = std::min( ObjId( ( eId + 1 ) * numLeaves ), ObjId( numObjects_ ) );
        ObjBitSet obs( last );
        obs.set( first, last - first, true );
        return obs;
    }
    virtual ICPElementBitSet getElementNodes( ICPLayer l, ICPElementId eId ) const override
    {
        assert( l > 0 );
        size_t nodeSize = 1;
        for ( int i = 1; i < l; ++i )
            nodeSize *= maxGroupSize_;
        auto maxNode = ( numObjects_ + nodeSize - 1 ) / nodeSize;
        ICPElementId first = ICPElementId( eId * maxGroupSize_ );
        ICPElementId last = std::min( ICPElementId( ( eId + 1 ) * maxGroupSize_ ), ICPElementId( maxNode ) );
        ICPElementBitSet ebs( last );
        ebs.set( first, last - first, true );
        return ebs;
    }

    virtual size_t getNumElements( ICPLayer l ) const override
    {
        size_t numLeaves = 1;
        for ( int i = 0; i < l; ++i )
            numLeaves *= maxGroupSize_;
        return ( numObjects_ + numLeaves - 1 ) / numLeaves;
    }
    virtual size_t getNumLayers() const override
    {
        int numLayers = 1;
        int numElements = numObjects_;
        while ( numElements > 1 )
        {
            numElements = ( numElements + maxGroupSize_ - 1 ) / maxGroupSize_;
            numLayers++;
        }
        return numLayers;
    }

private:
    int numObjects_{ 0 };
    int maxGroupSize_{ 0 };
};

class AABBTreeCascade : public IICPTreeIndexer
{
    using Subtrees = std::vector<NodeId>;
    using LayerNodes = Vector<ICPElementBitSet, ICPElementId>;
    using LayerLeaves = Vector<ObjBitSet, ICPElementId>;
public:
    AABBTreeCascade( const ICPObjects& objs, int maxGroupSize ) :
        tree_( objs ),
        maxGroupSize_{ maxGroupSize },
        numObjs_{ objs.size() }
    {
        int numLeaves = int( objs.size() );
        while ( numLeaves > maxGroupSize_ )
        {
            int numSubtrees = 1;
            while ( numLeaves > maxGroupSize_ )
            {
                numLeaves = ( numLeaves + 1 ) / 2;
                numSubtrees <<= 1;
            }
            layers_.emplace_back( tree_.getSubtrees( numSubtrees ) );
            numLeaves = int( layers_.back().size() );
        }
        leavesPerLayer_.resize( layers_.size() );
        for ( ICPLayer l( 0 ); l < layers_.size(); ++l )
        {
            const auto& subtrees = layers_[l];
            auto& leavesOnLayer = leavesPerLayer_[l];
            leavesOnLayer.resize( subtrees.size() );
            ParallelFor( leavesOnLayer, [&] ( ICPElementId id )
            {
                leavesOnLayer[id] = tree_.getSubtreeLeaves( subtrees[id.get()] );
            } );
        }
        if ( layers_.size() < 2 )
            return;
        nodesPerLayer_.resize( layers_.size() - 1 );
        for ( int l = 0; l < nodesPerLayer_.size(); ++l )
        {
            auto& nodesOnLayer = nodesPerLayer_[l];
            nodesOnLayer.resize( layers_[l + 1].size() );
            for ( ICPElementId nId( 0 ); nId < nodesOnLayer.size(); ++nId )
            {
                ICPElementBitSet& elements = nodesOnLayer[nId];
                elements.resize( layers_[l].size() );
                BitSetParallelForAll( elements, [&] ( ICPElementId id )
                {
                    elements.set( id, ( leavesPerLayer_[l][id] & leavesPerLayer_[l + 1][nId] ).any() );
                } );
            }
        }
    }

    virtual bool fromSameNode( ICPLayer l, ICPElementId eI, ICPElementId eJ ) const override
    {
        if ( l == 0 )
        {
            for ( const auto& leaves : leavesPerLayer_[0] )
            {
                if ( leaves.test( ObjId( eI.get() ) ) && leaves.test( ObjId( eJ.get() ) ) )
                    return true;
            }
            return false;
        }
        if ( l - 1 < nodesPerLayer_.size() )
        {
            for ( const auto& elements : nodesPerLayer_[l - 1] )
            {
                if ( elements.test( eI ) && elements.test( eJ ) )
                    return true;
            }
            return false;
        }
        return true;
    }

    virtual ObjBitSet getElementLeaves( ICPLayer l, ICPElementId eId ) const override
    {
        if ( l == 0 )
        {
            ObjBitSet obs( ObjId( eId.get() ) + 1 );
            obs.set( ObjId( eId.get() ) );
            return obs;
        }
        assert( l - 1 < leavesPerLayer_.size() );
        return leavesPerLayer_[l - 1][eId];
    }

    virtual ICPElementBitSet getElementNodes( ICPLayer l, ICPElementId eId ) const override
    {
        assert( l > 0 );
        if ( l == 1 )
            return ICPElementBitSet( leavesPerLayer_[l - 1][eId] );

        if ( l - 2 < nodesPerLayer_.size() )
            return nodesPerLayer_[l - 2][eId];

        auto res = ICPElementBitSet( layers_.back().size() );
        res.flip();
        return res;
    }

    virtual size_t getNumElements( ICPLayer l ) const override
    {
        if ( l == 0 )
            return numObjs_;
        if ( l - 1 < layers_.size() )
            return layers_[l - 1].size();
        return 1;
    }

    virtual size_t getNumLayers() const override
    {
        return layers_.size() + 1;
    }
private:
    AABBTreeObjects tree_;
    int maxGroupSize_{ 0 };
    size_t numObjs_{ 0 };

    std::vector<Subtrees> layers_; // start from layer 1 (layers_[0] is l==1 from ICP)
    std::vector<LayerNodes> nodesPerLayer_; // start from layer 1 (nodesPerLayer_[0] is l==1 from ICP)
    std::vector<LayerLeaves> leavesPerLayer_; // start from layer 2 (leavesPerLayer_[0] is l==2 from ICP)
};

void updateGroupPairs( ICPGroupPairs& pairs, const ICPObjects& objs,
    ICPGroupProjector srcProjector, ICPGroupProjector tgtProjector,
    float cosThreshold, float distThresholdSq, bool mutualClosest )
{
    MR_TIMER;
    pairs.active.clear();
    pairs.active.resize( pairs.vec.size(), true );

    // calculate pairs
    BitSetParallelForAll( pairs.active, [&] ( size_t idx )
    {
        auto & res = pairs.vec[idx];
        const auto p0 = objs[res.srcId.objId].obj.points()[res.srcId.vId];
        const auto pt = objs[res.srcId.objId].xf( p0 );

        ObjId prjObj;
        MeshOrPoints::ProjectionResult prj;
        // do not search for target point further than distance threshold
        prj.distSq = distThresholdSq;
        if ( res.tgtClosestId.objId && res.tgtClosestId.vId )
        {
            // start with old closest point ...
            prj.point = objs[res.tgtClosestId.objId].obj.points()[res.tgtClosestId.vId];
            if ( auto tgtNormals = objs[res.tgtClosestId.objId].obj.normals() )
                prj.normal = tgtNormals( res.tgtClosestId.vId );
            prj.distSq = ( pt - objs[res.tgtClosestId.objId].xf( prj.point ) ).lengthSq();
            prj.closestVert = res.tgtClosestId.vId;
            prjObj = res.tgtClosestId.objId;
        }
        // ... and try to find only closer one
        tgtProjector( pt, prj, prjObj );
        if ( !prj.closestVert || prj.distSq > distThresholdSq || prj.isBd )
        {
            // no target point found within distance threshold
            pairs.active.reset( idx );
            return;
        }
        assert( prjObj );

        res.distSq = prj.distSq;
        res.weight = 1.0;
        if ( auto srcWeights = objs[res.srcId.objId].obj.weights() )
            res.weight = srcWeights( res.srcId.vId );

        res.tgtClosestId.objId = prjObj;
        res.tgtClosestId.vId = prj.closestVert;

        res.srcPoint = pt;
        res.tgtPoint = objs[res.tgtClosestId.objId].xf( prj.point );

        auto srcNormals = objs[res.srcId.objId].obj.normals();
        res.srcNorm = srcNormals ? ( objs[res.srcId.objId].xf.A * srcNormals( res.srcId.vId ) ).normalized() : Vector3f();
        res.tgtNorm = prj.normal ? ( objs[res.tgtClosestId.objId].xf.A * prj.normal.value() ).normalized() : Vector3f();

        auto normalsAngleCos = ( prj.normal && srcNormals ) ? dot( res.tgtNorm, res.srcNorm ) : 1.0f;

        if ( normalsAngleCos < cosThreshold )
        {
            pairs.active.reset( idx );
            return;
        }
        if ( mutualClosest )
        {
            assert( srcProjector );
            prjObj = res.srcId.objId;
            prj.closestVert = res.srcId.vId;
            // keep prj.distSq
            srcProjector( res.tgtPoint, prj, prjObj );
            if ( prj.closestVert != res.srcId.vId )
                pairs.active.reset( idx );
        }
    } );
}

MultiwayICP::MultiwayICP( const ICPObjects& objects, const MultiwayICPSamplingParameters& samplingParams ) :
    objs_{ objects }
{
    resamplePoints( samplingParams );
}

Vector<AffineXf3f, ObjId> MultiwayICP::calculateTransformations( ProgressCallback cb )
{
    MR_TIMER
    float minDist = std::numeric_limits<float>::max();
    int badIterCount = 0;
    resultType_ = ICPExitType::MaxIterations;

    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        const bool pt2pt = ( prop_.method == ICPMethod::Combined && iter_ < 3 )
            || prop_.method == ICPMethod::PointToPoint;

        bool res = doIteration_( !pt2pt );

        if ( perIterationCb_ )
            perIterationCb_( iter_ );

        if ( !res )
        {
            resultType_ = ICPExitType::NotFoundSolution;
            break;
        }

        const float curDist = pt2pt ? getMeanSqDistToPoint() : getMeanSqDistToPlane();
        if ( prop_.exitVal > curDist )
        {
            resultType_ = ICPExitType::StopMsdReached;
            break;
        }

        // exit if several(3) iterations didn't decrease minimization parameter
        if ( curDist < minDist )
        {
            minDist = curDist;
            badIterCount = 0;
        }
        else
        {
            if ( badIterCount >= prop_.badIterStopCount )
            {
                resultType_ = ICPExitType::MaxBadIterations;
                break;
            }
            badIterCount++;
        }
        if ( !reportProgress( cb, float( iter_ ) / float( prop_.iterLimit ) ) )
            return {};
    }

    Vector<AffineXf3f, ObjId> res;
    res.resize( objs_.size() );
    for ( int i = 0; i < objs_.size(); ++i )
        res[ObjId( i )] = objs_[ObjId( i )].xf;
    return res;
}

Vector<AffineXf3f, ObjId> MultiwayICP::calculateTransformationsFixFirst( ProgressCallback cb )
{
    Vector<AffineXf3f, ObjId> res;
    if ( objs_.empty() )
        return res;

    const auto xf0 = objs_[ObjId( 0 )].xf;
    res = calculateTransformations( cb );

    /// apply the same (updateXf) to all objects to restore transformation of first object,
    /// and make relative position of others the same
    assert( res[ObjId( 0 )] == objs_[ObjId( 0 )].xf );
    const auto updateXf = xf0 * res[ObjId( 0 )].inverse();
    res[ObjId( 0 )] = objs_[ObjId( 0 )].xf = xf0;
    for ( int i = 1; i < objs_.size(); ++i )
    {
        assert( res[ObjId( i )] == objs_[ObjId( i )].xf );
        res[ObjId( i )] = objs_[ObjId( i )].xf = updateXf * res[ObjId( i )];
    }

    return res;
}

bool MultiwayICP::resamplePoints( const MultiwayICPSamplingParameters& samplingParams )
{
    MR_TIMER;

    maxGroupSize_ = samplingParams.maxGroupSize;
    setupLayers_( samplingParams.cascadeMode );

    samplingSize_ = samplingParams.samplingVoxelSize;
    bool cascadeMode = pairsGridPerLayer_.size() > 1;

    Vector<VertBitSet, ObjId> samplesPerObj( objs_.size() );

    float maxProgress = cascadeMode ? 0.2f : 0.5f;
    auto keepGoing = ParallelFor( objs_, [&] ( ObjId ind )
    {
        const auto& obj = objs_[ind];
        samplesPerObj[ind] = *obj.obj.pointsGridSampling( samplingSize_ );
    }, subprogress( samplingParams.cb, 0.0f, maxProgress ) );

    if ( !keepGoing )
        return false;

    float maxProgress2 = cascadeMode ? 0.4f : 1.0f;
    reservePairsLayer0_( std::move( samplesPerObj ), subprogress( samplingParams.cb, maxProgress, maxProgress2 ) );

    if ( !cascadeMode )
        return true;

    maxProgress = maxProgress2;
    maxProgress2 = 0.7f;
    auto samplesPerUnit = resampleUpperLayers_( subprogress( samplingParams.cb, maxProgress, maxProgress2 ) );
    if ( !samplesPerUnit )
        return false;
    // only do something if cascade mode is required, really should be called on each iteration
    return reserveUpperLayerPairs_( std::move( *samplesPerUnit ), subprogress( samplingParams.cb, maxProgress2, 1.0f ) );
}

float MultiwayICP::getMeanSqDistToPoint( std::optional<double> value ) const
{
    NumSum numSum;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        numSum = numSum + tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), NumSum(),
        [&] ( const auto& range, NumSum curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr = curr + MR::getSumSqDistToPoint( pairs[ICPElementId( i )][ICPElementId( j )], value );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return numSum.rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPlane( std::optional<double> value ) const
{
    NumSum numSum;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        numSum = numSum + tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), NumSum(),
        [&] ( const auto& range, NumSum curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr = curr + MR::getSumSqDistToPlane( pairs[ICPElementId( i )][ICPElementId( j )], value );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return numSum.rootMeanSqF();
}

size_t MultiwayICP::getNumSamples() const
{
    size_t num = 0;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        // it is deterministic to reduce integers without parallel_deterministic_reduce
        num += tbb::parallel_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), size_t( 0 ),
        [&] ( const auto& range, size_t curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr += MR::getNumSamples( pairs[ICPElementId( i )][ICPElementId( j )] );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return num;
}

size_t MultiwayICP::getNumActivePairs() const
{
    size_t num = 0;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        // it is deterministic to reduce integers without parallel_deterministic_reduce
        num += tbb::parallel_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), size_t( 0 ),
        [&] ( const auto& range, size_t curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr += MR::getNumActivePairs( pairs[ICPElementId( i )][ICPElementId( j )] );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return num;
}

std::string MultiwayICP::getStatusInfo() const
{
    return getICPStatusInfo( iter_, resultType_ );
}

bool MultiwayICP::updateAllPointPairs( ProgressCallback cb )
{
    MR_TIMER;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        bool keepGoing = updateLayerPairs_( l, subprogress( cb, float( l ) / float( pairsGridPerLayer_.size() ), float( l + 1 ) / float( pairsGridPerLayer_.size() ) ) );
        if ( !keepGoing )
            return false;
    }
    return true;
}

void MultiwayICP::setupLayers_( MultiwayICPSamplingParameters::CascadeMode mode )
{
    if ( maxGroupSize_ <= 1 || objs_.size() <= maxGroupSize_ )
    {
        pairsGridPerLayer_.resize( 1 );
        return;
    }
    if ( mode == MultiwayICPSamplingParameters::CascadeMode::Sequential )
        cascadeIndexer_ = std::make_unique<SeqCascade>( int( objs_.size() ), maxGroupSize_ );
    else if ( mode == MultiwayICPSamplingParameters::CascadeMode::AABBTreeBased )
        cascadeIndexer_ = std::make_unique<AABBTreeCascade>( objs_, maxGroupSize_ );

    assert( cascadeIndexer_ );
    pairsGridPerLayer_.resize( cascadeIndexer_->getNumLayers() );
}

bool MultiwayICP::reservePairsLayer0_( Vector<VertBitSet, ObjId>&& samplesPerObj, ProgressCallback cb )
{
    bool cascadeMode = pairsGridPerLayer_.size() > 1;

    assert( !pairsGridPerLayer_.empty() );
    auto& pairsGrid = pairsGridPerLayer_[0];
    pairsGrid.clear();
    pairsGrid.resize( objs_.size() );
    for ( ICPElementId i( 0 ); i < objs_.size(); ++i )
    {
        auto& pairs = pairsGrid[i];
        pairs.resize( objs_.size() );
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( i == j )
                continue;

            if ( cascadeMode )
            {
                if ( !cascadeIndexer_->fromSameNode( 0, i, j ) )
                    continue;
            }

            auto& thisPairs = pairs[j];
            ObjId srcObj = ObjId( i.get() );
            thisPairs.vec.reserve( samplesPerObj[srcObj].count() );
            for ( auto v : samplesPerObj[srcObj] )
            {
                auto& back = thisPairs.vec.emplace_back();
                back.srcId.objId = srcObj;
                back.srcId.vId = v;
            }
            thisPairs.active.reserve( thisPairs.vec.size() );
            thisPairs.active.clear();
        }
        if ( !reportProgress( cb, float( i + 1 ) / float( objs_.size() ), i, 64 ) )
            return false;
    }
    return true;
}

std::optional<MultiwayICP::LayerSamples> MultiwayICP::resampleUpperLayers_( ProgressCallback cb )
{
    MR_TIMER;
    if ( pairsGridPerLayer_.size() < 2 )
        return {};

    auto numLayers = cascadeIndexer_->getNumLayers();
    LayerSamples samples( numLayers );
    for ( ICPLayer l = 1; l < numLayers; ++l )
    {
        auto& layerSamples = samples[l];
        layerSamples.resize( cascadeIndexer_->getNumElements( l ) );
        bool keepGoing = ParallelFor( layerSamples, [&] ( ICPElementId gId )
        {
            const auto& layerLeaves = cascadeIndexer_->getElementLeaves( l, gId );
            Vector<ModelPointsData, ObjId> groupData;
            groupData.reserve( layerLeaves.count() );
            for ( ObjId oId : layerLeaves )
            {
                const auto& obj = objs_[oId];
                groupData.emplace_back( ModelPointsData{ .points = &obj.obj.points(), .validPoints = &obj.obj.validPoints(),.xf = &obj.xf,.fakeObjId = oId } );
            }
            layerSamples[gId] = *multiModelGridSampling( groupData, samplingSize_ );
        }, subprogress( cb, float( l - 1 ) / float( numLayers - 1 ), float( l ) / float( numLayers - 1 ) ) );
        if ( !keepGoing )
            return {};
    }
    return samples;
}

bool MultiwayICP::reserveUpperLayerPairs_( LayerSamples&& samples, ProgressCallback cb )
{
    MR_TIMER;
    if ( samples.empty() )
        return true;
    assert( pairsGridPerLayer_.size() > 0 );
    pairsGridPerLayer_.resize( samples.size() );
    for ( ICPLayer l( 1 ); l < pairsGridPerLayer_.size(); ++l )
    {
        auto sb = subprogress( cb, float( l - 1 ) / float( pairsGridPerLayer_.size() - 1 ), float( l ) / float( pairsGridPerLayer_.size() - 1 ) );
        const auto& samplesOnLayer = samples[l];
        auto& pairsOnLayer = pairsGridPerLayer_[l];
        int numGroups = int( samplesOnLayer.size() );
        pairsOnLayer.resize( numGroups );
        for ( ICPElementId gI( 0 ); gI < numGroups; ++gI )
        {
            auto& pairsOnGroup = pairsOnLayer[gI];
            pairsOnGroup.resize( numGroups );
            const auto& thisSamples = samplesOnLayer[gI];
            for ( ICPElementId gJ( 0 ); gJ < numGroups; ++gJ )
            {
                if ( gI == gJ )
                    continue;

                if ( !cascadeIndexer_->fromSameNode( l, gI, gJ ) )
                    continue;

                auto& thisPairs = pairsOnGroup[gJ];

                thisPairs.vec.resize( thisSamples.size() );
                for ( int i = 0; i < thisPairs.vec.size(); ++i )
                    thisPairs.vec[i].srcId = thisSamples[i];
                thisPairs.active.reserve( thisPairs.vec.size() );
                thisPairs.active.clear();
            }
            if ( !reportProgress( sb, float( gI + 1 ) / float( numGroups ) ) )
                return false;
        }
    }
    return true;
}

bool MultiwayICP::updateLayerPairs_( ICPLayer l, ProgressCallback cb )
{
    MR_TIMER;
    auto numGroups = pairsGridPerLayer_[l].size();
    bool cascadeMode = pairsGridPerLayer_.size() > 1;

    assert( l == 0 || cascadeMode );
    assert( l < pairsGridPerLayer_.size() );

    // build trees
    using LayerTrees = Vector<AABBTreeObjects, ICPElementId>;
    using LayerMaps = Vector<ObjMap, ICPElementId>;
    LayerTrees trees;
    LayerMaps maps;
    if ( l == 0 )
    {
        for ( const auto& obj : objs_ )
            obj.obj.cacheAABBTree(); // trigger AABB Tree build if needed not to fall in deep stack while projecting
    }
    else
    {
        // only for upper layers
        trees.resize( numGroups );
        maps.resize( numGroups );
        ParallelFor( trees, [&] ( ICPElementId gI )
        {
            const auto& leaves = cascadeIndexer_->getElementLeaves( l, gI );
            maps[gI].reserve( leaves.count() );
            ICPObjects leafObjs;
            leafObjs.reserve( leaves.count() );
            for ( auto leaf : leaves )
            {
                maps[gI].emplace_back( leaf );
                leafObjs.emplace_back( objs_[leaf] );
            }
            trees[gI] = AABBTreeObjects( std::move( leafObjs ) );
        } );
    }

    auto createProjector = [&] ( ICPElementId elId )->ICPGroupProjector
    {
        if ( l == 0 )
            return [this, elId] ( const Vector3f& p, MeshOrPoints::ProjectionResult& res, ObjId& resId ) mutable
        {
            auto proj = objs_[ObjId( elId.get() )].obj.limitedProjector();
            proj( objs_[ObjId( elId.get() )].xf.inverse()( p ), res );
            if ( res.closestVert )
                resId = ObjId( elId.get() );
        };
        return [&trees, &maps, elId] ( const Vector3f& p, MeshOrPoints::ProjectionResult& res, ObjId& resId )
        {
            projectOnAll( p, trees[elId], res.distSq, [&] ( ObjId oId, MeshOrPoints::ProjectionResult prj )
            {
                if ( prj.distSq >= res.distSq )
                    return;
                res = prj;
                resId = maps[elId][oId];
            } );
        };
    };


    // update pairs
    auto gridSize = numGroups * numGroups;
    auto keepGoung = ParallelFor( 0, int( gridSize ), [&] ( int gridId )
    {
        ICPElementId gI = ICPElementId( gridId / numGroups );
        ICPElementId gJ = ICPElementId( gridId % numGroups );
        if ( gI == gJ )
            return;

        if ( cascadeMode )
        {
            if ( !cascadeIndexer_->fromSameNode( l, gI, gJ ) )
                return;
        }

        MR::updateGroupPairs( pairsGridPerLayer_[l][gI][gJ], objs_, createProjector( gI ), createProjector( gJ ), prop_.cosThreshold, prop_.distThresholdSq, prop_.mutualClosest );
    }, cb );
    if ( !keepGoung )
        return false;
    deactivateFarDistPairs_( l );
    return true;
}

// unified function for mean sq dist to point calculation
float calcMeanSqDistToPointEx( const auto& pairs, auto id )
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( decltype( id )( 0 ), decltype( id )( pairs.size() ) ), NumSum(),
        [&] ( const auto& range, NumSum curr )
    {
        for ( auto i = range.begin(); i < range.end(); ++i )
        {
            if ( i == id )
                continue;
            curr = curr + MR::getSumSqDistToPoint( pairs[i][id] ) + MR::getSumSqDistToPoint( pairs[id][i] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } ).rootMeanSqF();
}

void MultiwayICP::deactivateFarDistPairs_( ICPLayer l )
{
    MR_TIMER;
    auto& pairs = pairsGridPerLayer_[l];
    Vector<float, ICPElementId> maxDistSq( pairs.size() );
    for ( int it = 0; it < 3; ++it )
    {
        ParallelFor( maxDistSq, [&] ( auto id )
        {
            maxDistSq[id] = sqr( prop_.farDistFactor * MR::calcMeanSqDistToPointEx( pairs, id ) );
        } );

        tbb::enumerable_thread_specific<size_t> counters( 0 );
        ParallelFor( size_t( 0 ), pairs.size() * pairs.size(), [&] ( size_t r )
        {
            auto i = ICPElementId( r % pairs.size() );
            auto j = ICPElementId( r / pairs.size() );
            if ( i == j )
                return;
            if ( maxDistSq[i] >= prop_.distThresholdSq )
                return;
            counters.local() += MR::deactivateFarPairs( pairs[i][j], maxDistSq[i] );
        } );

        size_t numDeactivated = 0;
        for ( auto counter : counters )
            numDeactivated += counter;
        if ( numDeactivated == 0 )
            break;
    }
}

bool MultiwayICP::doIteration_( bool p2pl )
{
    if ( pairsGridPerLayer_.size() > 1 )
        return cascadeIter_( p2pl );

    updateAllPointPairs();

    if ( maxGroupSize_ == 1 )
        return p2pl ? p2plIter_() : p2ptIter_();

    return multiwayIter_( p2pl );
}

bool MultiwayICP::p2ptIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, ObjId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( ObjId objId )
    {
        ICPElementId id( objId.get() );
        const auto& pairs = pairsGridPerLayer_[0];
        PointToPointAligningTransform p2pt;
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairs[id][j].active )
            {
                const auto& vp = pairs[id][j].vec[idx];
                p2pt.add( vp.srcPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
            for ( size_t idx : pairs[j][id].active )
            {
                const auto& vp = pairs[j][id].vec[idx];
                p2pt.add( vp.tgtPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
        }

        AffineXf3f res;
        switch ( prop_.icpMode )
        {
        default:
            assert( false );
            [[fallthrough]];
        case ICPMode::RigidScale:
            res = AffineXf3f( p2pt.findBestRigidScaleXf() );
            break;
        case ICPMode::AnyRigidXf:
            res = AffineXf3f( p2pt.findBestRigidXf() );
            break;
        case ICPMode::OrthogonalAxis:
            res = AffineXf3f( p2pt.findBestRigidXfOrthogonalRotationAxis( Vector3d{ prop_.fixedRotationAxis } ) );
            break;
        case ICPMode::FixedAxis:
            res = AffineXf3f( p2pt.findBestRigidXfFixedRotationAxis( Vector3d{ prop_.fixedRotationAxis } ) );
            break;
        case ICPMode::TranslationOnly:
            res = AffineXf3f( Matrix3f(), Vector3f( p2pt.findBestTranslation() ) );
            break;
        }
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[objId] = FullSizeBool( false );
            return;
        }
        objs_[objId].xf = res * objs_[objId].xf;
        valid[objId] = FullSizeBool( true );
    } );

    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::p2plIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, ObjId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( ObjId objId )
    {
        ICPElementId id( objId.get() );
        const auto& pairs = pairsGridPerLayer_[0];
        Vector3f centroidRef;
        int activeCount = 0;
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairs[id][j].active )
            {
                const auto& vp = pairs[id][j].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.srcPoint;
                ++activeCount;
            }
            for ( size_t idx : pairs[j][id].active )
            {
                const auto& vp = pairs[j][id].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.tgtPoint;
                ++activeCount;
            }
        }
        if ( activeCount <= 0 )
        {
            valid[objId] = FullSizeBool( false );
            return;
        }
        centroidRef /= float( activeCount * 2 );
        AffineXf3f centroidRefXf = AffineXf3f( Matrix3f(), centroidRef );

        PointToPlaneAligningTransform p2pl;
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairs[id][j].active )
            {
                const auto& vp = pairs[id][j].vec[idx];
                p2pl.add( vp.srcPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.tgtNorm, vp.weight );
            }
            for ( size_t idx : pairs[j][id].active )
            {
                const auto& vp = pairs[j][id].vec[idx];
                p2pl.add( vp.tgtPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.srcNorm, vp.weight );
            }
        }
        p2pl.prepare();

        AffineXf3f res = getAligningXf( p2pl, prop_.icpMode, prop_.p2plAngleLimit, prop_.p2plScaleLimit, prop_.fixedRotationAxis );
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[objId] = FullSizeBool( false );
            return;
        }
        objs_[objId].xf = centroidRefXf * res * centroidRefXf.inverse() * objs_[objId].xf;
        valid[objId] = FullSizeBool( true );
    } );
    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::multiwayIter_( bool p2pl )
{
    MR_TIMER;
    Vector<MultiwayAligningTransform,ObjId> mats( objs_.size() );
    ParallelFor( mats, [&] ( ObjId objId )
    {
        ICPElementId i( objId.get() );
        const auto& pairs = pairsGridPerLayer_[0];
        auto& mat = mats[objId];
        mat.reset( int( objs_.size() ) );
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == i )
                continue;
            for ( auto idx : pairs[i][j].active )
            {
                const auto& data = pairs[i][j].vec[idx];
                if ( p2pl )
                    mat.add( int( i ), data.srcPoint, int( j ), data.tgtPoint, ( data.tgtNorm + data.srcNorm ).normalized(), data.weight );
                else
                    mat.add( int( i ), data.srcPoint, int( j ), data.tgtPoint, data.weight );
            }
        }
    } );

    MultiwayAligningTransform mat;
    mat.reset( int( objs_.size() ) );
    for ( const auto& m : mats )
        mat.add( m );

    mats = {}; // free memory

    MultiwayAligningTransform::Stabilizer stabilizer;
    stabilizer.rot = samplingSize_ * 1e-1f;
    stabilizer.shift = 1e-3f;
    auto res = mat.solve( stabilizer );
    for ( ObjId i( 0 ); i < objs_.size(); ++i )
    {
        auto resI = res[i.get()].rigidXf();
        if ( std::isnan( resI.b.x ) )
            return false;
        objs_[i].xf = AffineXf3f( resI * AffineXf3d( objs_[i].xf ) );
    }
    return true;
}

bool MultiwayICP::cascadeIter_( bool p2pl /*= true */ )
{
    for ( ICPLayer l = 0; l < pairsGridPerLayer_.size(); ++l )
    {
        updateLayerPairs_( l );

        const auto& pairsOnLayer = pairsGridPerLayer_[l];
        auto numHyperGroups = cascadeIndexer_->getNumElements( l + 1 );

        for ( ICPElementId hgI( 0 ); hgI < numHyperGroups; ++hgI )
        {
            auto nodes = cascadeIndexer_->getElementNodes( l + 1, hgI );
            assert( nodes.any() );
            auto numNodes = nodes.count();
            if ( numNodes == 1 )
                continue;

            int indI = 0;
            MultiwayAligningTransform mat;
            mat.reset( int( numNodes ) );
            for ( auto nodeI : nodes )
            {
                int indJ = 0;
                for ( auto nodeJ : nodes )
                {
                    if ( nodeI == nodeJ )
                    {
                        ++indJ;
                        continue;
                    }
                    const auto& pairs = pairsOnLayer[nodeI][nodeJ];
                    for ( auto idx : pairs.active )
                    {
                        const auto& data = pairs.vec[idx];
                        if ( p2pl )
                            mat.add( indI, data.srcPoint, indJ, data.tgtPoint, ( data.tgtNorm + data.srcNorm ).normalized(), data.weight );
                        else
                            mat.add( indI, data.srcPoint, indJ, data.tgtPoint, data.weight );
                    }
                    ++indJ;
                }
                ++indI;
            }

            MultiwayAligningTransform::Stabilizer stabilizer;
            stabilizer.rot = samplingSize_;
            stabilizer.shift = 1e-2f;
            auto res = mat.solve( stabilizer );
            indI = 0;
            for ( auto nodeI : nodes )
            {
                auto resI = res[indI].rigidXf();
                if ( std::isnan( resI.b.x ) )
                    return false;
                const auto& leaves = cascadeIndexer_->getElementLeaves( l, nodeI );
                for ( auto objId : leaves )
                    objs_[objId].xf = AffineXf3f( resI * AffineXf3d( objs_[objId].xf ) );

                ++indI;
            }
        }
    }
    return true;
}

}
