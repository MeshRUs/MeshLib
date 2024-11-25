#pragma once
#include "MRICP.h"
#include "MRGridSampling.h"

namespace MR
{

using ICPObjects = Vector<MeshOrPointsXf, ObjId>;

using ICPLayer = int;
class ICPElemtTag;
using ICPElementId = Id<ICPElemtTag>;
using ICPElementBitSet = TaggedBitSet<ICPElemtTag>;
struct ICPGroupPair : public ICPPairData
{
    ObjVertId srcId;
    ObjVertId tgtClosestId;
};
struct ICPGroupPairs : public IPointPairs
{
    virtual const ICPPairData& operator[]( size_t idx ) const override { return vec[idx]; }
    virtual ICPPairData& operator[]( size_t idx ) override { return vec[idx]; }
    virtual size_t size() const override { return vec.size(); }
    std::vector<ICPGroupPair> vec;
};

using ICPGroupProjector = std::function<void( const Vector3f& p, MeshOrPoints::ProjectionResult& res, ObjId& resId )>;
/// in each pair updates the target data and performs basic filtering (activation)
MRMESH_API void updateGroupPairs( ICPGroupPairs& pairs, const ICPObjects& objs,
    ICPGroupProjector srcProjector, ICPGroupProjector tgtProjector,
    float cosThreshold, float distThresholdSq, bool mutualClosest );

using ICPPairsGrid = Vector<Vector<ICPGroupPairs, ICPElementId>, ICPElementId>;


/// structure to find leafs and groups of each in cascade mode
class IICPTreeIndexer
{
public:
    virtual ~IICPTreeIndexer() = default;
    /// returns true if eI and eJ are from same node
    virtual bool fromSameNode( ICPLayer l, ICPElementId eI, ICPElementId eJ ) const = 0;

    /// returns bitset of leaves of given node
    virtual ObjBitSet getElementLeaves( ICPLayer l, ICPElementId eId ) const = 0;
    /// valid for l > 0, returns bitset of subnodes that is associated with eId
    /// should be valid for l == `getNumLayers`
    virtual ICPElementBitSet getElementNodes( ICPLayer l, ICPElementId eId ) const = 0;

    /// l == 0 - objs_.size()
    /// l == 1 - number of nodes one layer above objects
    /// l == 2 - number of nodes one layer above nodes lvl1
    /// ...
    /// l == `getNumLayers` - 1
    virtual size_t getNumElements( ICPLayer l ) const = 0;
    virtual size_t getNumLayers() const = 0;
};

/// Parameters that are used for sampling of the MultiwayICP objects
struct MultiwayICPSamplingParameters
{
    /// sampling size of each object
    float samplingVoxelSize = 0.0f;

    /// size of maximum icp group to work with
    /// if number of objects exceeds this value, icp is applied in cascade mode
    int maxGroupSize = 64;

    enum class CascadeMode
    {
        Sequential, /// separates objects on groups based on their index in ICPObjects (good if all objects about the size of all objects together)
        AABBTreeBased /// builds AABB tree based on each object bounding box and separates subtrees (good if each object much smaller then all objects together)
    } cascadeMode{ CascadeMode::AABBTreeBased };

    /// callback for progress reports
    ProgressCallback cb;
};

/// This class allows you to register many objects having similar parts
/// and known initial approximations of orientations/locations using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
class MRMESH_CLASS MultiwayICP
{
public:
    MRMESH_API MultiwayICP( const ICPObjects& objects, const MultiwayICPSamplingParameters& samplingParams );

    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformations of all objects to reach registered state
    /// the transformation of the last object is fixed and does not change here
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, ObjId> calculateTransformations( ProgressCallback cb = {} );

    /// runs ICP algorithm given input objects, transformations, and parameters;
    /// \return adjusted transformations of all objects to reach registered state
    /// the transformation of the first object is fixed and does not change here
    [[nodiscard]] MRMESH_API Vector<AffineXf3f, ObjId> calculateTransformationsFixFirst( ProgressCallback cb = {} );

    /// select pairs with origin samples on all objects
    MRMESH_API bool resamplePoints( const MultiwayICPSamplingParameters& samplingParams );

    /// in each pair updates the target data and performs basic filtering (activation)
    /// in cascade mode only useful for stats update
    MRMESH_API bool updateAllPointPairs( ProgressCallback cb = {} );

    /// tune algorithm params before run calculateTransformations()
    void setParams( const ICPProperties& prop ) { prop_ = prop; }
    [[nodiscard]] const ICPProperties& getParams() const { return prop_; }

    /// computes root-mean-square deviation between points
    /// or the standard deviation from given value if present
    [[nodiscard]] MRMESH_API float getMeanSqDistToPoint( std::optional<double> value = {} ) const;

    /// computes root-mean-square deviation from points to target planes
    /// or the standard deviation from given value if present
    [[nodiscard]] MRMESH_API float getMeanSqDistToPlane( std::optional<double> value = {} ) const;

    /// computes the number of samples able to form pairs
    [[nodiscard]] MRMESH_API size_t getNumSamples() const;

    /// computes the number of active point pairs
    [[nodiscard]] MRMESH_API size_t getNumActivePairs() const;

    /// sets callback that will be called for each iteration
    void setPerIterationCallback( std::function<void( int inter )> callback ) { perIterationCb_ = std::move( callback ); }

    /// if in independent equations mode - creates separate equation system for each object
    /// otherwise creates single large equation system for all objects
    bool devIndependentEquationsModeEnabled() const { return maxGroupSize_ == 1; }
    void devEnableIndependentEquationsMode( bool on ) { maxGroupSize_ = on ? 1 : 0; }

    /// returns status info string
    [[nodiscard]] MRMESH_API std::string getStatusInfo() const;

    using PairsPerLayer = Vector<ICPPairsGrid, ICPLayer>;
    /// returns all pairs of all layers
    const PairsPerLayer& getPairsPerLayer() const { return pairsGridPerLayer_; }

    /// returns pointer to class that is used to navigate among layers of cascade registration
    /// if nullptr - cascade mode is not used
    const IICPTreeIndexer* getCascadeIndexer() const { return cascadeIndexer_.get(); }
private:
    ICPObjects objs_;
    PairsPerLayer pairsGridPerLayer_;
    ICPProperties prop_;

    ICPExitType resultType_{ ICPExitType::NotStarted };

    std::function<void( int )> perIterationCb_;

    std::unique_ptr<IICPTreeIndexer> cascadeIndexer_;

    /// reserves space in pairsGridPerLayer_ according to mode and GroupIndexer
    void setupLayers_( MultiwayICPSamplingParameters::CascadeMode mode );

    /// reserves memory for all pairs
    /// if currently in cascade mode (objs.size() > maxGroupSize_) reserves only for pairs inside groups
    bool reservePairsLayer0_( Vector<VertBitSet, ObjId>&& samples, ProgressCallback cb );

    using LayerSamples = Vector<Vector<MultiObjsSamples, ICPElementId>, ICPLayer>;
    std::optional<LayerSamples> resampleUpperLayers_( ProgressCallback cb );
    bool reserveUpperLayerPairs_( LayerSamples&& samples, ProgressCallback cb );

    /// calculates and updates pairs 2nd and next steps of cascade mode
    bool updateLayerPairs_( ICPLayer l, ProgressCallback cb = {} );
    /// deactivate pairs that does not meet farDistFactor criterion, for given layer
    void deactivateFarDistPairs_( ICPLayer l );

    float samplingSize_{ 0.0f };
    // this parameter indicates maximum number of objects that might be aligned simultaneously in multi-way mode
    // N<0 - means all of the objects
    // N=1 - means that all objects are aligned independently
    // N>1 - means that registration is applied cascade with N grouping step:
    //       1) separate all objects to groups with N members
    //       2) align objects inside these groups
    //       3) join groups and create groups of these joined objects (2nd layer groups)
    //       4) align this groups
    // N>number of objects - same as 0
    int maxGroupSize_{ 64 };
    int iter_ = 0;
    bool doIteration_( bool p2pl );
    bool p2ptIter_();
    bool p2plIter_();
    bool multiwayIter_( bool p2pl = true );
    bool cascadeIter_( bool p2pl = true );
};

}
