#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRId.h"
#include <optional>

namespace MR
{

/// performs sampling of mesh vertices;
/// subdivides mesh bounding box on voxels of approximately given size and returns at most one vertex per voxel;
/// returns std::nullopt if it was terminated by the callback
MRMESH_API std::optional<VertBitSet> verticesGridSampling( const MeshPart& mp, float voxelSize, const ProgressCallback & cb = {} );

/// performs sampling of cloud points;
/// subdivides point cloud bounding box on voxels of approximately given size and returns at most one point per voxel;
/// returns std::nullopt if it was terminated by the callback
MRMESH_API std::optional<VertBitSet> pointGridSampling( const PointCloud& cloud, float voxelSize, const ProgressCallback & cb = {} );


/// structure to contain pointers to model data
struct ModelPointsData
{
    /// all points of model
    const VertCoords* points{ nullptr };
    /// bitset of valid points
    const VertBitSet* validPoints{ nullptr };
    /// model world xf
    const AffineXf3f* xf{ nullptr };
    /// if present this value will override ObjId in result ObjVertId
    ObjId fakeObjId{};
};

struct ObjVertId
{
    ObjId objId;
    VertId vId;
};

using MultiObjsSamples = std::vector<ObjVertId>;

/// performs sampling of several models respecting their world transformations
/// subdivides models bounding box on voxels of approximately given size and returns at most one point per voxel;
/// returns std::nullopt if it was terminated by the callback
MRMESH_API std::optional<MultiObjsSamples> multiModelGridSampling( const Vector<ModelPointsData, ObjId>& models, float voxelSize, const ProgressCallback& cb = {} );

} //namespace MR
