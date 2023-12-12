#pragma once
#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRSignDetectionMode.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <optional>
#include <string>

namespace MR
{

struct BaseShellParameters
{
    /// Size of voxel in grid conversions;
    /// The user is responsible for setting some positive value here
    float voxelSize = 0;

    /// Progress callback
    ProgressCallback callBack;
};

/// computes size of a cubical voxel to get approximately given number of voxels during rasterization
[[nodiscard]] MRMESH_API float suggestVoxelSize( const MeshPart & mp, float approxNumVoxels );

struct OffsetParameters : BaseShellParameters
{
    /// determines the method to compute distance sign
    SignDetectionMode signDetectionMode = SignDetectionMode::OpenVDB;

    /// defines particular implementation of IFastWindingNumber interface that will compute windings. If it is not specified, default FastWindingNumber is used
    std::shared_ptr<IFastWindingNumber> fwn;
};

struct SharpOffsetParameters : OffsetParameters
{
    /// if non-null then created sharp edges will be saved here
    UndirectedEdgeBitSet* outSharpEdges = nullptr;
    /// minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize
    float minNewVertDev = 1.0f / 25;
    /// maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize
    float maxNewRank2VertDev = 5;
    /// maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize
    float maxNewRank3VertDev = 2;
    /// correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;
    /// big correction can be wrong and result from self-intersections in the reference mesh
    float maxOldVertPosCorrection = 0.5f;
};

#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
/// Offsets mesh by converting it to distance field in voxels using OpenVDB library,
/// signDetectionMode = Unsigned(from OpenVDB) | OpenVDB | HoleWindingRule,
/// and then converts back using OpenVDB library (dual marching cubes),
/// so result mesh is always closed
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> offsetMesh( const MeshPart& mp, float offset, const OffsetParameters& params = {} );

/// in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);
/// in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);
/// if your input mesh is closed then please specify params.type == Offset, and you will get closed mesh on output;
/// if your input mesh is open then please specify params.type == Shell, and you will get open mesh on output
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> thickenMesh( const Mesh& mesh, float offset, const OffsetParameters & params = {} );
#endif

/// Offsets mesh by converting it to voxels and back two times
/// only closed meshes allowed (only Offset mode)
/// typically offsetA and offsetB have distinct signs
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> doubleOffsetMesh( const MeshPart& mp, float offsetA, float offsetB, const OffsetParameters& params = {} );

/// Offsets mesh by converting it to distance field in voxels (using OpenVDB library if SignDetectionMode::OpenVDB or our implementation otherwise)
/// and back using standard Marching Cubes, as opposed to Dual Marching Cubes in offsetMesh(...)
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> mcOffsetMesh( const Mesh& mesh, float offset, 
    const OffsetParameters& params = {}, Vector<VoxelId, FaceId>* outMap = nullptr );

/// Constructs a shell around selected mesh region with the properties that every point on the shall must
///  1. be located not further than given distance from selected mesh part,
///  2. be located not closer to not-selected mesh part than to selected mesh part.
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> mcShellMeshRegion( const Mesh& mesh, const FaceBitSet& region, float offset,
    const BaseShellParameters& params, Vector<VoxelId, FaceId> * outMap = nullptr );

/// Offsets mesh by converting it to voxels and back
/// post process result using reference mesh to sharpen features
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> sharpOffsetMesh( const Mesh& mesh, float offset, const SharpOffsetParameters& params = {} );

#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_VOXEL )
/// Offsets polyline by converting it to voxels and building iso-surface
/// do offset in all directions
/// so result mesh is always closed
/// params.type is ignored (always assumed Shell)
[[nodiscard]] MRMESH_API Expected<Mesh, std::string> offsetPolyline( const Polyline3& polyline, float offset, const OffsetParameters& params = {} );
#endif

}
