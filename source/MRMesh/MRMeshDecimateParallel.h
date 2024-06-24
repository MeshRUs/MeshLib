#pragma once

#include "MRMeshDecimate.h"

namespace MR
{

/**
 * \struct MR::DecimateParallelSettings
 * \brief Parameters structure for MR::decimateParallelMesh
 * \ingroup DecimateGroup
 *
 * \sa \ref decimateParallelMesh
 */
struct DecimateParallelSettings
{  
    DecimateStrategy strategy = DecimateStrategy::MinimizeError;
    /// for DecimateStrategy::MinimizeError: 
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    float maxError = 0.001f;
    /// Maximal possible edge length created during decimation
    float maxEdgeLen = FLT_MAX;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift = FLT_MAX;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio = 20;
    /// the algorithm will try to eliminate triangles with equal or larger aspect ratio, ignoring normal orientation checks
    float criticalTriAspectRatio = FLT_MAX;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 0.001f;
    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos = true;
    /// Region on mesh to be decimated, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// Whether to allow collapsing edges having at least one vertex on (region) boundary
    bool touchBdVertices = true;
    /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
    /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
    float maxAngleChange = -1;
    /// Subdivides mesh on given number of parts to process them in parallel
    int subdivideParts = 32;
    /**
     * \brief  The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives both vertices of the edge being collapsed: v1 will disappear,
     * and v0 will get new position (provided as the third argument) after collapse;
     * If the callback returns false, then the collapse is prohibited;
     * \note This callback will be called from parallel threads when they process subparts
     */
    std::function<bool( VertId v0, VertId v1, const Vector3f & newV0Pos )> preCollapse;
    /**
     * \brief The user can provide this optional callback for adjusting error introduced by
     * edge collapse between vertices v0 and v1 of the original mesh, and the collapse position.
     * \details On input the callback gets the squared error and position computed by standard means,
     * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
     * This callback can be called from many threads in parallel and must be thread-safe.
     * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
     */
    std::function<void( VertId v0, VertId v1, float & collapseErrorSq, Vector3f & collapsePos )> adjustCollapse;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback = {};
};

/**
 * \brief Collapse edges in mesh region according to the settings
 * \ingroup DecimateGroup
 * \details Analog of decimateMesh for parallel computing. If accuracy is preferable to speed, use \ref decimateMesh.
 * Limitations:
 * 1) original FaceIds and EdgeIds are not preserved;
 * 2) multiple edges are not supported.
 *
 * \sa \ref decimateMesh
 */
MRMESH_API DecimateResult decimateParallelMesh( Mesh & mesh, const DecimateParallelSettings & settings = {} );

} //namespace MR
