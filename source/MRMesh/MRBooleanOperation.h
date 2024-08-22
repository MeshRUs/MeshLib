#pragma once

#include "MRVector.h"
#include "MRExpected.h"
#include "MRId.h"
#include <array>

namespace MR
{

/// \addtogroup BooleanGroup
/// \{

/**
  * Enum class of available CSG operations
  * \image html boolean/no_bool.png "Two separate meshes" width = 300cm
  * \sa \ref MR::boolean
  */
enum class BooleanOperation
{
    /// Part of mesh `A` that is inside of mesh `B`
    /// \image html boolean/inside_a.png "Inside A" width = 300cm
    InsideA,
    /// Part of mesh `B` that is inside of mesh `A`
    /// \image html boolean/inside_b.png "Inside B" width = 300cm
    InsideB,
    /// Part of mesh `A` that is outside of mesh `B`
    /// \image html boolean/outside_a.png "Outside A" width = 300cm
    OutsideA,
    /// Part of mesh `B` that is outside of mesh `A`
    /// \image html boolean/outside_b.png "Outside B" width = 300cm
    OutsideB,
    /// Union surface of two meshes (outside parts)
    /// \image html boolean/union.png "Union" width = 300cm
    Union,
    /// Intersection surface of two meshes (inside parts)
    /// \image html boolean/intersection.png "Intersection" width = 300cm
    Intersection,
    /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
    /// \image html boolean/b-a.png "Difference B-A" width = 300cm
    DifferenceBA,
    /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
    /// \image html boolean/a-b.png "Difference A-B" width = 300cm
    DifferenceAB,

    Count ///< not a valid operation
};

/** \struct MR::BooleanResultMapper
  * \brief Structure to map old mesh BitSets to new
  * \details Structure to easily map topology of MR::boolean input meshes to result mesh
  *
  * This structure allows to map faces, vertices and edges of mesh `A` and mesh `B` input of MR::boolean to result mesh topology primitives
  * \sa \ref MR::boolean
  */
struct BooleanResultMapper
{
    /// Input object index enum
    enum class MapObject { A, B, Count };

    BooleanResultMapper() = default;

    /// Returns faces bitset of result mesh corresponding input one
    MRMESH_API FaceBitSet map( const FaceBitSet& oldBS, MapObject obj ) const;
    /// Returns vertices bitset of result mesh corresponding input one
    MRMESH_API VertBitSet map( const VertBitSet& oldBS, MapObject obj ) const;
    /// Returns edges bitset of result mesh corresponding input one
    MRMESH_API EdgeBitSet map( const EdgeBitSet& oldBS, MapObject obj ) const;
    /// Returns only new faces that are created during boolean operation
    MRMESH_API FaceBitSet newFaces() const;

    /// returns updated oldBS leaving only faces that has corresponding ones in result mesh
    MRMESH_API FaceBitSet filteredOldFaceBitSet( const FaceBitSet& oldBS, MapObject obj );

    struct Maps
    {
        /// "after cut" faces to "origin" faces
        /// this map is not 1-1, but N-1
        FaceMap cut2origin;
        /// "after cut" faces to "after stitch" faces (1-1)
        FaceMap cut2newFaces;
        /// "origin" edges to "after stitch" edges (1-1)
        WholeEdgeMap old2newEdges;
        /// "origin" vertices to "after stitch" vertices (1-1)
        VertMap old2newVerts;
        /// old topology indexes are valid if true
        bool identity{false};
    };
    std::array<Maps, size_t( MapObject::Count )> maps;

    [[nodiscard]] const Maps& getMaps( MapObject index ) const { return maps[ int( index ) ]; }
};

/// Parameters will be useful if specified
struct BooleanInternalParameters
{
    /// Instance of original mesh with tree for better speed
    const Mesh* originalMeshA{ nullptr };
    /// Instance of original mesh with tree for better speed
    const Mesh* originalMeshB{ nullptr };
    /// Optional output cut edges of booleaned meshes
    std::vector<EdgeLoop>* optionalOutCut{ nullptr };
};

/// Perform boolean operation on cut meshes
/// \return mesh in space of meshA or error.
/// \note: actually this function is meant to be internal, use "boolean" instead
MRMESH_API Expected<Mesh> doBooleanOperation( Mesh&& meshACut, Mesh&& meshBCut,
    const std::vector<EdgePath>& cutEdgesA, const std::vector<EdgePath>& cutEdgesB,
    BooleanOperation operation, const AffineXf3f* rigidB2A = nullptr,
    BooleanResultMapper* mapper = nullptr,
    bool mergeAllNonIntersectingComponents = false,
    const BooleanInternalParameters& intParams = {} );

/// \}

} //namespace MR
