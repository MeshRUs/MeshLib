#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRProgressCallback.h"
#include <climits>
#include <functional>

namespace MRE
{
/**
 * \defgroup DecimateGroup Decimate overview
 * \brief This chapter represents documentation about mesh decimation
 */

/// Defines the order of edge collapses inside Decimate algorithm
enum DecimateStrategy
{
    MinimizeError,    // the next edge to collapse will be the one that introduced minimal error to the surface
    ShortestEdgeFirst // the next edge to collapse will be the shortest one
};

/**
 * \struct MRE::DecimateSettings
 * \brief Parameters structure for MRE::decimateMesh
 * \ingroup DecimateGroup
 * 
 * \sa \ref decimateMesh
 */
struct DecimateSettings
{  
    DecimateStrategy strategy = DecimateStrategy::MinimizeError;
    /// for DecimateStrategy::MinimizeError: 
    ///   stop the decimation as soon as the estimated distance deviation from the original mesh is more than this value
    /// for DecimateStrategy::ShortestEdgeFirst only:
    ///   stop the decimation as soon as the longest edge in the mesh is greater than this value
    float maxError = 0.001f;
    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio = 20;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 0.001f;
    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;
    /// Limit on the number of deleted faces
    int maxDeletedFaces = INT_MAX;
    /// Region on mesh to be decimated, it is updated during the operation
    MR::FaceBitSet * region = nullptr;
    /// Whether to allow collapsing edges having at least one vertex on (region) boundary
    bool touchBdVertices = true;
    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    std::function<bool( MR::EdgeId edgeToCollapse, const MR::Vector3f & newEdgeOrgPos)> preCollapse;
    /**
     * \brief  If not null, then
     * on input: if the vector is not empty then it is takes for initialization instead of form computation for all vertices;
     * on output: quadratic form for each remaining vertex is returned there
     */
    MR::Vector<MR::QuadraticForm3f, MR::VertId> * vertForms = nullptr;
    ///  whether to pack mesh at the end
    bool packMesh = false;
    /// callback to report algorithm progress and cancel it by user request
    MR::ProgressCallback progressCallback = {};
};

/**
 * \struct MRE::DecimateResult
 * \brief Results of MRE::decimateMesh
 * \ingroup DecimateGroup
 * 
 * \sa \ref decimateMesh
 * \sa \ref decimateParallelMesh
 * \sa \ref resolveMeshDegenerations
 */
struct DecimateResult
{
    int vertsDeleted = 0; ///< Number deleted verts. Same as the number of performed collapses
    int facesDeleted = 0; ///< Number deleted faces
    /// for DecimateStrategy::MinimizeError:
    ///    estimated distance deviation of decimated mesh from the original mesh
    /// for DecimateStrategy::ShortestEdgeFirst:
    ///    the shortest remaining edge in the mesh
    float errorIntroduced = 0;
    /// whether the algorithm was cancelled by the callback
    bool cancelled = true;
};

/**
 * \brief Collapse edges in mesh region according to the settings
 * \ingroup DecimateGroup
 * \details Have version for parallel computing - \ref decimateParallelMesh
 *
 * \image html decimate/decimate_before.png "Before" width = 350cm
 * \image html decimate/decimate_after.png "After" width = 350cm
 * 
 * \sa \ref decimateParallelMesh
 * \sa \ref resolveMeshDegenerations
 */ 
MREALGORITHMS_API DecimateResult decimateMesh( MR::Mesh & mesh, const DecimateSettings & settings = {} );

/**
 * \brief Computes quadratic form at given vertex of the initial surface before decimation
 * \ingroup DecimateGroup
 */
MREALGORITHMS_API MR::QuadraticForm3f computeFormAtVertex( const MR::MeshPart & mp, MR::VertId v, float stabilizer );

/**
 * \brief Resolves degenerate triangles in given mesh
 * \details This function performs decimation, so it can affect topology
 * \ingroup DecimateGroup
 * \return true if the mesh has been changed
 * 
 * \sa \ref decimateMesh
 */
MREALGORITHMS_API bool resolveMeshDegenerations( MR::Mesh& mesh, int maxIters = 1, float maxDeviation = 0 );

struct RemeshSettings
{
    // the algorithm will try to keep the length of all edges close to this value,
    // splitting twice longer edges, and eliminating twice shorter edges
    float targetEdgeLen = 0.001f;
    // maximum allowed deviation during triangulation optimization
    float maxDeviation = 1e-5f;
    /// Region on mesh to be changed, it is updated during the operation
    MR::FaceBitSet * region = nullptr;
    ///  whether to pack mesh at the end
    bool packMesh = false;
};
// Splits too long and eliminates too short edges from the mesh
MREALGORITHMS_API void remesh( MR::Mesh& mesh, const RemeshSettings & settings );

} //namespace MR
