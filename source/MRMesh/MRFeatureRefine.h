#pragma once

#include "MRFeatureObject.h"

namespace MR
{

/// Optional parameters for \ref refineFeatureObject
struct MRMESH_CLASS RefineParameters
{
    /// Maximum distance from the source model to the feature
    float distanceLimit{ 0.1f };
    /// Maximum angle between the source model's normal and the feature's normal
    float normalTolerance{ 30.f };
    /// (for meshes only) Reference faces used for filtering intermediate results that are too far from it
    const FaceBitSet* faceRegion;
    /// (for meshes only) Reference vertices used for filtering intermediate results that are too far from it
    const VertBitSet* vertRegion;
    /// Maximum amount of iterations performed until a stable set of points is found
    int maxIterations{ 10 };
    /// Progress callback
    ProgressCallback callback;
};

/// Recalculate the feature object's position so it would better fit with the given mesh
MRMESH_API Expected<AffineXf3f> refineFeatureObject( const FeatureObject& featObj, const Mesh& mesh,
                                                     const RefineParameters& params = {} );

/// Recalculate the feature object's position so it would better fit with the given point cloud
MRMESH_API Expected<AffineXf3f> refineFeatureObject( const FeatureObject& featObj, const PointCloud& pointCloud,
                                                     const RefineParameters& params = {} );

} // namespace MR
