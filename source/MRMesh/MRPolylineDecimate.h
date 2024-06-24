#pragma once

#include "MRMeshFwd.h"
#include <cfloat>
#include <climits>
#include <functional>

namespace MR
{

/**
 * \struct MR::DecimatePolylineSettings
 * \brief Parameters structure for MR::decimatePolyline
 * \ingroup DecimateGroup
 *
 * \sa \ref decimatePolyline
 */
template<typename V>
struct DecimatePolylineSettings
{
    /// Limit from above on the maximum distance from moved vertices to original contour
    float maxError = 0.001f;

    /// Maximal possible edge length created during decimation
    float maxEdgeLen = FLT_MAX;

    /// Stabilizer is dimensionless coefficient.
    /// The larger is stabilizer, the more Decimator will strive to retain the density of input points.
    /// If stabilizer is zero, then only the shape of input line will be preserved.
    float stabilizer = 0.001f;

    /// if true then after each edge collapse the position of remaining vertex is optimized to
    /// minimize local shape change, if false then the edge is collapsed in one of its vertices, which keeps its position
    bool optimizeVertexPos = true;

    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;

    /// Region of the polyline to be decimated, it is updated during the operation
    /// Remain nullptr to include the whole polyline
    VertBitSet* region = nullptr;

    /// Whether to allow collapsing edges with at least one vertex on the end of not-closed polyline
    /// (or on region boundary if region is given);
    /// if touchBdVertices is false then boundary vertices are strictly fixed
    bool touchBdVertices = true;

    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    std::function<bool( EdgeId edgeToCollapse, const V & newEdgeOrgPos )> preCollapse;

    /**
     * \brief The user can provide this optional callback for adjusting error introduced by this
     * edge collapse and the collapse position.
     * \details On input the callback gets the squared error and position computed by standard means,
     * and callback can modify any of them. The larger the error, the later this edge will be collapsed.
     * This callback can be called from many threads in parallel and must be thread-safe.
     * This callback can be called many times for each edge before real collapsing, and it is important to make the same adjustment.
     */
    std::function<void( UndirectedEdgeId ue, float & collapseErrorSq, V & collapsePos )> adjustCollapse;

    /**
     * \brief  If not null, then
     * on input: if the vector is not empty then it is taken for initialization instead of form computation for all vertices;
     * on output: quadratic form for each remaining vertex is returned there
     */
    Vector<QuadraticForm<V>, VertId>* vertForms = nullptr;
};

using DecimatePolylineSettings2 = DecimatePolylineSettings<Vector2f>;
using DecimatePolylineSettings3 = DecimatePolylineSettings<Vector3f>;

/**
 * \struct MR::DecimatePolylineResult
 * \brief Results of MR::decimateContour
 */
struct DecimatePolylineResult
{
    int vertsDeleted = 0; ///< Number deleted verts. Same as the number of performed collapses
    float errorIntroduced = 0; ///< Max different (as distance) between original contour and result contour
};

/**
 * \brief Collapse edges in the polyline according to the settings
 * \ingroup DecimateGroup
 */
MRMESH_API DecimatePolylineResult decimatePolyline( Polyline2& polyline, const DecimatePolylineSettings2& settings = {} );
MRMESH_API DecimatePolylineResult decimatePolyline( Polyline3& polyline, const DecimatePolylineSettings3& settings = {} );

/**
 * \brief Collapse edges in the contour according to the settings
 * \ingroup DecimateGroup
 */ 
MRMESH_API DecimatePolylineResult decimateContour( Contour2f& contour, const DecimatePolylineSettings2& settings = {} );
MRMESH_API DecimatePolylineResult decimateContour( Contour3f& contour, const DecimatePolylineSettings3& settings = {} );

} //namespace MR
