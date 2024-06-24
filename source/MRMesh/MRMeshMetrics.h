#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"
#include <functional>

namespace MR
{
/// \addtogroup FillHoleGroup
/// \{

using FillTriangleMetric = std::function<double( VertId, VertId, VertId )>;
using FillEdgeMetric = std::function<double( VertId, VertId, VertId, VertId )>;
using FillCombineMetric = std::function<double( double, double )>;

/// Big value, but less then DBL_MAX, to be able to pass some bad triangulations instead of breaking it
/// e10 - real metrics to have weight in triangulation, if it would be more than e15+ some metrics will be less than double precision
MRMESH_API extern const double BadTriangulationMetric;

/** \struct MR::FillHoleMetric
  * \brief Holds metrics for fillHole and buildCylinderBetweenTwoHoles triangulation\n
  * 
  * This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions\n
  * 
  * \sa \ref getCircumscribedMetric
  * \sa \ref getPlaneFillMetric
  * \sa \ref getEdgeLengthFillMetric
  * \sa \ref getEdgeLengthStitchMetric
  * \sa \ref getComplexStitchMetric
  * \sa \ref fillHole
  * \sa \ref buildCylinderBetweenTwoHoles
  */
struct FillHoleMetric
{
    /// is called for each triangle, if it is set
    FillTriangleMetric triangleMetric;
    /// is called for each edge, if it is set
    FillEdgeMetric edgeMetric;
    /// is called to combine metrics from different candidates, if it is not set it just summarizes input
    FillCombineMetric combineMetric;
};

/// Computes combined metric after filling a hole
MRMESH_API double calcCombinedFillMetric( const Mesh & mesh, const FaceBitSet & filledRegion, const FillHoleMetric & metric );

/// This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.
/// It is rather fast to calculate, and it results in typically good triangulations.
MRMESH_API FillHoleMetric getCircumscribedMetric( const Mesh& mesh );

/// Same as getCircumscribedFillMetric, but with extra penalty for the triangles having
/// normals looking in the opposite side of plane containing left of (e).
MRMESH_API FillHoleMetric getPlaneFillMetric( const Mesh& mesh, EdgeId e );

/// Similar to getPlaneFillMetric with extra penalty for the triangles having
/// normals looking in the opposite side of plane containing left of (e),
/// but the metric minimizes the sum of circumcircle radius times aspect ratio for all triangles in the triangulation.
MRMESH_API FillHoleMetric getPlaneNormalizedFillMetric( const Mesh& mesh, EdgeId e );

/// This metric minimizes the sum of triangleMetric for all triangles in the triangulation
/// plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n
/// Where\n
/// triangleMetric is proportional to triangle aspect ratio\n
/// edgeMetric is proportional to ( 1 - dihedralAngleCos )
MRMESH_API FillHoleMetric getComplexStitchMetric( const Mesh& mesh );

/// Simple metric minimizing the sum of all edge lengths
MRMESH_API FillHoleMetric getEdgeLengthFillMetric( const Mesh& mesh );

/// Forbids connecting vertices from the same hole \n
/// Simple metric minimizing edge length
MRMESH_API FillHoleMetric getEdgeLengthStitchMetric( const Mesh& mesh );

/// Forbids connecting vertices from the same hole \n
/// All new faces should be parallel to given direction
MRMESH_API FillHoleMetric getVerticalStitchMetric( const Mesh& mesh, const Vector3f& upDir );

/// This metric minimizes the sum of triangleMetric for all triangles in the triangulation
/// plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n
/// Where\n
/// triangleMetric is proportional to weighted triangle area and triangle aspect ratio\n
/// edgeMetric grows with angle between triangles as ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4.
MRMESH_API FillHoleMetric getComplexFillMetric( const Mesh& mesh, EdgeId e );

/// This metric minimizes summary projection of new edges to plane normal, (try do produce edges parallel to plane)
MRMESH_API FillHoleMetric getParallelPlaneFillMetric( const Mesh& mesh, EdgeId e, const Plane3f* plane = nullptr );

/// This metric minimizes the maximal dihedral angle between the faces in the triangulation
/// and on its boundary
MRMESH_API FillHoleMetric getMaxDihedralAngleMetric( const Mesh& mesh );

/// This metric minimizes the maximal dihedral angle between the faces in the triangulation
/// and on its boundary, and it avoids creating too degenerate triangles;
///  for planar holes it is the same as getCircumscribedMetric
MRMESH_API FillHoleMetric getUniversalMetric( const Mesh& mesh );

/// This metric maximizes the minimal angle among all faces in the triangulation
MRMESH_API FillHoleMetric getMinTriAngleMetric( const Mesh& mesh );

/// This metric is for triangulation construction with minimal summed area of triangles.
/// Warning: this metric can produce degenerated triangles
MRMESH_API FillHoleMetric getMinAreaMetric( const Mesh& mesh );

/// \}

}