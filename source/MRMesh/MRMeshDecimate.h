#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"
#include "MRConstants.h"
#include <cfloat>
#include <climits>
#include <functional>

namespace MR
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
 * \struct MR::DecimateSettings
 * \brief Parameters structure for MR::decimateMesh
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
    ///   stop the decimation as soon as the shortest edge in the mesh is greater than this value
    float maxError = 0.001f;

    /// Maximal possible edge length created during decimation
    float maxEdgeLen = FLT_MAX;

    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift = FLT_MAX;

    /// Maximal possible aspect ratio of a triangle introduced during decimation
    float maxTriangleAspectRatio = 20;

    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalTriAspectRatio = FLT_MAX;

    /// edges not longer than this value will be collapsed even if it results in appearance of a triangle with high aspect ratio
    float tinyEdgeLength = -1;

    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 0.001f;

    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos = true;

    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;

    /// Limit on the number of deleted faces
    int maxDeletedFaces = INT_MAX;

    /// Region on mesh to be decimated, it is updated during the operation
    FaceBitSet * region = nullptr;

    /// Edges specified by this bit-set will never be flipped, but they can be collapsed or replaced during collapse of nearby edges so it is updated during the operation
    UndirectedEdgeBitSet* notFlippable = nullptr;

    /// If pointer is not null, then only edges from here can be collapsed (and some nearby edges can disappear)
    const UndirectedEdgeBitSet * edgesToCollapse = nullptr;

    /// Whether to allow collapsing edges having at least one vertex on (region) boundary
    bool touchBdVertices = true;

    /// if touchBdVertices=false then the algorithm needs to know about all boundary vertices;
    /// if the pointer is not null then boundary vertices detection is replaced with testing values in this bit-set
    const VertBitSet * bdVerts = nullptr;

    /// Permit edge flips (in addition to collapsing) to improve Delone quality of the mesh
    /// if it does not change dihedral angle more than on this value (negative value prohibits any edge flips)
    float maxAngleChange = -1;

    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    std::function<bool( EdgeId edgeToCollapse, const Vector3f & newEdgeOrgPos)> preCollapse;

    /**
     * \brief The user can provide this optional callback for adjusting error introduced by this
     * edge collapse and the collapse position.
     * \details On input the callback gets the squared error and position computed by standard means,
     * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
     * This callback can be called from many threads in parallel and must be thread-safe.
     * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
     */
    std::function<void( UndirectedEdgeId ue, float & collapseErrorSq, Vector3f & collapsePos )> adjustCollapse;

    /// this function is called each time edge (e) is deleted;
    /// if valid (e1) is given then dest(e) = dest(e1) and their origins are in different ends of collapsing edge, e1 shall take the place of e
    std::function<void(EdgeId e, EdgeId e1)> onEdgeDel;

    /**
     * \brief  If not null, then vertex quadratic forms are stored there;
     * if on input the vector is not empty then initialization is skipped in favor of values from there;
     * on output: quadratic form for each remaining vertex is returned there
     */
    Vector<QuadraticForm3f, VertId> * vertForms = nullptr;

    ///  whether to pack mesh at the end
    bool packMesh = false;

    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback;

    /// If this value is more than 1, then virtually subdivides the mesh on given number of parts to process them in parallel (using many threads);
    /// unlike \ref decimateParallelMesh it does not create copies of mesh regions, so may take less memory to operate;
    /// IMPORTANT: please call mesh.packOptimally() before calling decimating with subdivideParts > 1, otherwise performance will be bad
    int subdivideParts = 1;

    /// After parallel decimation of all mesh parts is done, whether to perform final decimation of whole mesh region
    /// to eliminate small edges near the border of individual parts
    bool decimateBetweenParts = true;

    /// if not null, then it contains the faces of each subdivision part on input, which must not overlap,
    /// and after decimation of all parts, the region inside each part is put here;
    /// decimateBetweenParts=true or packMesh=true are not compatible with this option
    std::vector<FaceBitSet> * partFaces = nullptr;

    /// minimum number of faces in one subdivision part for ( subdivideParts > 1 ) mode
    int minFacesInPart = 0;
};

/**
 * \struct MR::DecimateResult
 * \brief Results of MR::decimateMesh
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
MRMESH_API DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings = {} );

/**
 * \brief Computes quadratic form at given vertex of the initial surface before decimation
 * \ingroup DecimateGroup
 */
[[nodiscard]] MRMESH_API QuadraticForm3f computeFormAtVertex( const MeshPart & mp, VertId v, float stabilizer );

/**
 * \brief Computes quadratic forms at every vertex of mesh part before decimation
 * \ingroup DecimateGroup
 */
[[nodiscard]] MRMESH_API Vector<QuadraticForm3f, VertId> computeFormsAtVertices( const MeshPart & mp, float stabilizer );

/**
 * \brief returns given subdivision part of all valid faces;
 * parallel threads shall be able to safely modify these bits because they do not share any block with other parts
 * \ingroup DecimateGroup
 */
[[nodiscard]] MRMESH_API FaceBitSet getSubdividePart( const FaceBitSet & valids, size_t subdivideParts, size_t myPart );

struct ResolveMeshDegenSettings
{
    [[deprecated]]
    int maxIters = 1;
    /// maximum permitted deviation from the original surface
    float maxDeviation = 0;
    /// edges not longer than this value will be collapsed ignoring normals and aspect ratio checks
    float tinyEdgeLength = 0;
    /// Permit edge flips if it does not change dihedral angle more than on this value
    float maxAngleChange = PI_F / 3;
    /// the algorithm will ignore dihedral angle check if one of triangles had aspect ratio equal or more than this value;
    /// and the algorithm will permit temporary increase in aspect ratio after collapse, if before collapse one of the triangles had larger aspect ratio
    float criticalAspectRatio = 10000;
    /// Small stabilizer is important to achieve good results on completely planar mesh parts,
    /// if your mesh is not-planer everywhere, then you can set it to zero
    float stabilizer = 1e-6f;
    /// degenerations will be fixed only in given region, which is updated during the processing
    FaceBitSet * region = nullptr;
};

/**
 * \brief Resolves degenerate triangles in given mesh
 * \details This function performs decimation, so it can affect topology
 * \ingroup DecimateGroup
 * \return true if the mesh has been changed
 * 
 * \sa \ref decimateMesh
 */
MRMESH_API bool resolveMeshDegenerations( Mesh& mesh, const ResolveMeshDegenSettings & settings = {} );
[[deprecated(" use the version with parameter struct instead" )]]
MRMESH_API bool resolveMeshDegenerations( Mesh& mesh, int maxIters, float maxDeviation = 0, float maxAngleChange = PI_F / 3, float criticalAspectRatio = 10000 );


struct RemeshSettings
{
    /// the algorithm will try to keep the length of all edges close to this value,
    /// splitting the edges longer than targetEdgeLen, and then eliminating the edges shorter than targetEdgeLen
    float targetEdgeLen = 0.001f;
    /// Maximum number of edge splits allowed during subdivision
    int maxEdgeSplits = 10'000'000;
    /// Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value
    float maxAngleChangeAfterFlip = 30 * PI_F / 180.0f;
    /// Maximal shift of a boundary during one edge collapse
    float maxBdShift = FLT_MAX;
    /// This option in subdivision works best for natural surfaces, where all triangles are close to equilateral and have similar area,
    /// and no sharp edges in between
    bool useCurvature = false;
    /// the number of iterations of final relaxation of mesh vertices;
    /// few iterations can give almost perfect uniformity of the vertices and edge lengths but deviate from the original surface
    int finalRelaxIters = 0;
    /// if true prevents the surface from shrinkage after many iterations
    bool finalRelaxNoShrinkage = false;
    /// Region on mesh to be changed, it is updated during the operation
    FaceBitSet * region = nullptr;
    /// Edges specified by this bit-set will never be flipped or collapsed, but they can be replaced during collapse of nearby edges so it is updated during the operation;
    /// also the vertices incident to these edges are excluded from relaxation
    UndirectedEdgeBitSet* notFlippable = nullptr;
    ///  whether to pack mesh at the end
    bool packMesh = false;
    /// if true, then every new vertex after subdivision will be projected on the original mesh (before smoothing);
    /// this does not affect the vertices moved on other stages of the processing
    bool projectOnOriginalMesh = false;
    /// this function is called each time edge (e) is split into (e1->e), but before the ring is made Delone
    std::function<void(EdgeId e1, EdgeId e)> onEdgeSplit;
    /// if valid (e1) is given then dest(e) = dest(e1) and their origins are in different ends of collapsing edge, e1 shall take the place of e
    std::function<void(EdgeId e, EdgeId e1)> onEdgeDel;
    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    std::function<bool( EdgeId edgeToCollapse, const Vector3f& newEdgeOrgPos )> preCollapse;
    /// callback to report algorithm progress and cancel it by user request
    ProgressCallback progressCallback;
};
// Splits too long and eliminates too short edges from the mesh
MRMESH_API bool remesh( Mesh& mesh, const RemeshSettings & settings );

} //namespace MR
