#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRConstants.h"
#include <cfloat>
#include <functional>

namespace MR
{

/// \defgroup MeshSubdivideGroup Mesh Subdivide
/// \ingroup MeshAlgorithmGroup
/// \{

struct SubdivideSettings
{
    /// Subdivision is stopped when all edges inside or on the boundary of the region are not longer than this value
    float maxEdgeLen = 0;
    /// Maximum number of edge splits allowed
    int maxEdgeSplits = 1000;
    /// Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation
    float maxDeviationAfterFlip = 1;
    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value (in radians)
    float maxAngleChangeAfterFlip = FLT_MAX;
    /// If this value is less than FLT_MAX then edge flips will
    /// ignore dihedral angle check if one of triangles has aspect ratio more than this value
    /// Unit: rad
    float criticalAspectRatioFlip = 1000.0f;
    /// Region on mesh to be subdivided, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// Edges specified by this bit-set will never be flipped, but they can be split so it is updated during the operation
    UndirectedEdgeBitSet* notFlippable = nullptr;
    /// New vertices appeared during subdivision will be added here
    VertBitSet * newVerts = nullptr;
    /// If false do not touch border edges (cannot subdivide lone faces)\n
    /// use \ref MR::findRegionOuterFaces to find boundary faces
    bool subdivideBorder = true;
    /// The subdivision stops as soon as all triangles (in the region) have aspect ratio below or equal to this value
    float maxTriAspectRatio = 0;
    /// An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value.
    /// So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization.
    /// Please set it to a smaller value only if subdivideBorder==false, otherwise many narrow triangles can appear near border
    float maxSplittableTriAspectRatio = FLT_MAX;
    /// Puts new vertices so that they form a smooth surface together with existing vertices.
    /// This option works best for natural surfaces without sharp edges in between triangles
    bool smoothMode = false;
    /// In case of activated smoothMode, the smoothness is locally deactivated at the edges having
    /// dihedral angle at least this value
    float minSharpDihedralAngle = PI_F / 6; // 30 degrees
    /// if true, then every new vertex will be projected on the original mesh (before smoothing)
    bool projectOnOriginalMesh = false;
    /// this function is called each time a new vertex has been created, but before the ring is made Delone
    std::function<void(VertId)> onVertCreated;
    /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
    std::function<void(EdgeId e1, EdgeId e)> onEdgeSplit;
    /// this function is called each time edge (e) is going to split, if it returns false then this split will be skipped
    std::function<bool(EdgeId e)> beforeEdgeSplit;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback = {};
};

/// Split edges in mesh region according to the settings;\n
/// \return The total number of edge splits performed
MRMESH_API int subdivideMesh( Mesh & mesh, const SubdivideSettings & settings = {} );

/// \}

}
