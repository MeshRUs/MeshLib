#pragma once

#include "MRMeshFwd.h"
#include "MRMeshMetrics.h"
#include <functional>
#include <memory>

namespace MR
{

/** \defgroup FillHoleGroup Fill/Stitch Holes overview
  * \brief This chapter represents documentation about hole triangulations or stitching two holes
  * \ingroup MeshAlgorithmGroup
  * \{
  */

/** \struct MR::FillHoleParams
  * \brief Parameters structure for MR::fillHole\n
  * Structure has some options to control MR::fillHole
  * 
  * \sa \ref fillHole
  * \sa \ref FillHoleMetric
  */
struct FillHoleParams
{
    /** Specifies triangulation metric\n
      * default for MR::fillHole: getCircumscribedFillMetric\n
      * \sa \ref FillHoleMetric
      */
    FillHoleMetric metric;
    /// If not nullptr accumulate new faces
    FaceBitSet* outNewFaces{ nullptr };
    /** If Strong makes additional efforts to avoid creating multiple edges, 
      * in some rare cases it is not possible (cases with extremely bad topology), 
      * if you faced one try to use \ref MR::duplicateMultiHoleVertices before \ref MR::fillHole
      * 
      * If Simple avoid creating edges that already exist in topology (default)
      * 
      * If None do not avoid multiple edges
      */
    enum class MultipleEdgesResolveMode
    {
        None,
        Simple,
        Strong
    } multipleEdgesResolveMode{ MultipleEdgesResolveMode::Simple };

    /** If true creates degenerate faces band around hole to have sharp angle visualization
      * \warning This flag bad for result topology, most likely you do not need it
      */
    bool makeDegenerateBand{ false };

    /** The maximum number of polygon subdivisions on a triangle and two smaller polygons,
      * must be 2 or larger
      */
    int maxPolygonSubdivisions{ 20 };

    /** Input/output value, if it is present: 
      * returns true if triangulation was bad and do not actually fill hole, 
      * if triangulation is ok returns false; 
      * if it is not present fill hole trivially in case of bad triangulation, (or leaves bad triangulation, depending on metric)
      */
    bool* stopBeforeBadTriangulation{ nullptr };
};

/** \struct MR::StitchHolesParams
  * \brief Parameters structure for MR::buildCylinderBetweenTwoHoles\n
  * Structure has some options to control MR::buildCylinderBetweenTwoHoles
  *
  * \sa \ref buildCylinderBetweenTwoHoles
  * \sa \ref FillHoleMetric
  */
struct StitchHolesParams
{
    /** Specifies triangulation metric\n
      * default for MR::buildCylinderBetweenTwoHoles: getComplexStitchMetric
      * \sa \ref FillHoleMetric
      */
    FillHoleMetric metric;
    /// If not nullptr accumulate new faces
    FaceBitSet* outNewFaces{ nullptr };
};

/** \brief Stitches two holes in Mesh\n
  *
  * Build cylindrical patch to fill space between two holes represented by one of their edges each,\n
  * default metric: ComplexStitchMetric
  *
  * \image html fill/before_stitch.png "Before" width = 250cm
  * \image html fill/stitch.png "After" width = 250cm
  * 
  * Next picture show, how newly generated faces can be smoothed
  * \ref MR::positionVertsSmoothly
  * \ref MR::subdivideMesh
  * \image html fill/stitch_smooth.png "Stitch with smooth" width = 250cm
  * 
  * \param mesh mesh with hole
  * \param a EdgeId which represents 1st hole (should not have valid left FaceId)
  * \param b EdgeId which represents 2nd hole (should not have valid left FaceId)
  * \param params parameters of holes stitching
  *
  * \sa \ref fillHole
  * \sa \ref StitchHolesParams
  */
MRMESH_API void buildCylinderBetweenTwoHoles( Mesh & mesh, EdgeId a, EdgeId b, const StitchHolesParams& params = {} );

/// this version finds holes in the mesh by itself and returns false if they are not found
MRMESH_API bool buildCylinderBetweenTwoHoles( Mesh & mesh, const StitchHolesParams& params = {} );


/** \brief Fills hole in mesh\n
  * 
  * Fills given hole represented by one of its edges (having no valid left face),\n
  * uses fillHoleTrivially if cannot fill hole without multiple edges,\n
  * default metric: CircumscribedFillMetric
  * 
  * \image html fill/before_fill.png "Before" width = 250cm
  * \image html fill/fill.png "After" width = 250cm
  *
  * Next picture show, how newly generated faces can be smoothed
  * \ref MR::positionVertsSmoothly
  * \ref MR::subdivideMesh
  * \image html fill/fill_smooth.png "Fill with smooth" width = 250cm
  * 
  * \param mesh mesh with hole
  * \param a EdgeId which represents hole (should not have valid left FaceId)
  * \param params parameters of hole filling
  * 
  * \sa \ref buildCylinderBetweenTwoHoles
  * \sa \ref fillHoleTrivially
  * \sa \ref FillHoleParams
  */
MRMESH_API void fillHole( Mesh& mesh, EdgeId a, const FillHoleParams& params = {} );

/// fill all holes given by their representative edges in \param as
MRMESH_API void fillHoles( Mesh& mesh, const std::vector<EdgeId> & as, const FillHoleParams& params = {} );

/// returns true if given loop is a boundary of one hole in given mesh topology:
/// * every edge in the loop does not have left face,
/// * next/prev edges in the loop are related as follows: next = topology.prev( prev.sym() )
/// if the function returns true, then any edge from the loop passed to \ref fillHole will fill the same hole
[[nodiscard]] MRMESH_API bool isHoleBd( const MeshTopology & topology, const EdgeLoop & loop );

struct FillHoleItem
{
    // if not-negative number then it is edgeid;
    // otherwise it refers to the edge created recently
    int edgeCode1, edgeCode2;
};

/// concise representation of proposed hole triangulation
struct HoleFillPlan
{
    std::vector<FillHoleItem> items;
    int numTris = 0; // the number of triangles in the filling
};

/// prepares the plan how to triangulate the face or hole to the left of (e) (not filling it immediately),
/// several getHoleFillPlan can work in parallel
MRMESH_API HoleFillPlan getHoleFillPlan( const Mesh& mesh, EdgeId e, const FillHoleParams& params = {} );

/// prepares the plan how to triangulate the planar face or planar hole to the left of (e) (not filling it immediately),
/// several getPlanarHoleFillPlan can work in parallel
MRMESH_API HoleFillPlan getPlanarHoleFillPlan( const Mesh& mesh, EdgeId e );

/// quickly triangulates the face or hole to the left of (e) given the plan (quickly compared to fillHole function)
MRMESH_API void executeHoleFillPlan( Mesh & mesh, EdgeId a0, HoleFillPlan & plan, FaceBitSet * outNewFaces = nullptr );

/** \brief Triangulates face of hole in mesh trivially\n
  * \ingroup FillHoleGroup
  *
  * Fills given hole represented by one of its edges (having no valid left face)\n
  * by creating one new vertex in the centroid of boundary vertices and connecting new vertex with all boundary vertices.
  *
  * \image html fill/before_fill.png "Before" width = 250cm
  * \image html fill/fill_triv.png "After" width = 250cm
  *
  * Next picture show, how newly generated faces can be smoothed
  * \ref MR::positionVertsSmoothly
  * \ref MR::subdivideMesh
  * \image html fill/fill_triv_smooth.png "Trivial fill with smooth" width = 250cm
  *
  * \param mesh mesh with hole
  * \param a EdgeId points on the face or hole to the left that will be triangulated
  * \param outNewFaces optional output newly generated faces
  * \return new vertex
  * 
  * \sa \ref fillHole
  */
MRMESH_API VertId fillHoleTrivially( Mesh& mesh, EdgeId a, FaceBitSet * outNewFaces = nullptr );

/// adds cylindrical extension of given hole represented by one of its edges (having no valid left face)
/// by adding new vertices located in given plane and 2 * number_of_hole_edge triangles;
/// \return the edge of new hole opposite to input edge (a)
MRMESH_API EdgeId extendHole( Mesh& mesh, EdgeId a, const Plane3f & plane, FaceBitSet * outNewFaces = nullptr );

/// adds cylindrical extension of too all holes of the mesh by calling extendHole(...);
/// \return representative edges of one per every hole after extension
MRMESH_API std::vector<EdgeId> extendAllHoles( Mesh& mesh, const Plane3f & plane, FaceBitSet * outNewFaces = nullptr );

/// adds extension of given hole represented by one of its edges (having no valid left face)
/// by adding new vertices located at getVertPos( existing vertex position );
/// \return the edge of new hole opposite to input edge (a)
MRMESH_API EdgeId extendHole( Mesh& mesh, EdgeId a, std::function<Vector3f(const Vector3f &)> getVertPos, FaceBitSet * outNewFaces = nullptr );

/// adds cylindrical extension of given hole represented by one of its edges (having no valid left face)
/// by adding new vertices located in lowest point of the hole -dir*holeExtension and 2 * number_of_hole_edge triangles;
/// \return the edge of new hole opposite to input edge (a)
MRMESH_API EdgeId buildBottom( Mesh& mesh, EdgeId a, Vector3f dir, float holeExtension, FaceBitSet* outNewFaces = nullptr );

/// creates a band of degenerate triangles around given hole;
/// \return the edge of new hole opposite to input edge (a)
MRMESH_API EdgeId makeDegenerateBandAroundHole( Mesh& mesh, EdgeId a, FaceBitSet * outNewFaces = nullptr );

/// creates a bridge between two boundary edges a and b (both having no valid left face);
/// bridge consists of two triangles in general or of one triangle if a and b are neighboring edges on the boundary;
/// \return false if bridge cannot be created because otherwise multiple edges appear
MRMESH_API bool makeBridge( MeshTopology & topology, EdgeId a, EdgeId b, FaceBitSet * outNewFaces = nullptr );

/// creates a new bridge edge between origins of two boundary edges a and b (both having no valid left face);
/// \return invalid id if bridge cannot be created because otherwise multiple edges appear
MRMESH_API EdgeId makeBridgeEdge( MeshTopology & topology, EdgeId a, EdgeId b );

/// \}

} // namespace MR
