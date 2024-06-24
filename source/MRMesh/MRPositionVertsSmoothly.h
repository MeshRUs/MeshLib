#pragma once

#include "MRMeshFwd.h"
#include "MREnums.h"

namespace MR
{

/// Puts given vertices in such positions to make smooth surface both inside verts-region and on its boundary;
/// \param verts must not include all vertices of a mesh connected component
/// \param fixedSharpVertices in these vertices the surface can be not-smooth
MRMESH_API void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts,
    EdgeWeights edgeWeightsType = EdgeWeights::Cotan,
    const VertBitSet * fixedSharpVertices = nullptr );

/// Puts given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary;
/// \param verts must not include all vertices of a mesh connected component unless vertStabilizers are given
/// \param vertShifts optional additional shifts of each vertex relative to smooth position
/// \param vertStabilizers optional per-vertex stabilizers: the more the value, the bigger vertex attraction to its original position
MRMESH_API void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts,
    const Vector<Vector3f, VertId>* vertShifts = nullptr,
    const VertScalars* vertStabilizers = nullptr );

struct SpacingSettings
{
    /// vertices to be moved by the algorithm, nullptr means all valid vertices
    const VertBitSet* region = nullptr;

    /// target distance of each edge in the mesh (for at least one edge's vertex in the region)
    UndirectedEdgeMetric dist; // must be defined by the caller

    /// the algorithm is iterative, the more iterations the closer result to exact solution
    int numIters = 10;

    /// too small number here can lead to instability, too large - to slow convergence
    float stabilizer = 3;

    /// maximum sum of minus negative weights, if it is exceeded then stabilizer is increased automatically
    float maxSumNegW = 0.1f;

    /// if this predicated is given, then all inverted faces will be converted in degenerate faces at the end of each iteration
    FacePredicate isInverted;
};

/// Moves given vertices to make the distances between them as specified
MRMESH_API void positionVertsWithSpacing( Mesh& mesh, const SpacingSettings & settings );

struct InflateSettings
{
    /// the amount of pressure applied to mesh region:
    /// positive pressure moves the vertices outside, negative - inside;
    /// please specify a value by magnitude about the region diagonal
    float pressure = 0;
    /// the number of internal iterations (>=1);
    /// larger number of iterations makes the performance slower, but the quality better
    int iterations = 3;
    /// smooths the area before starting inflation;
    /// please set to false only if the region is known to be already smooth
    bool preSmooth = true;
    /// whether to increase the pressure gradually during the iterations (recommended for best quality)
    bool gradualPressureGrowth = true;
};

/// Inflates (in one of two sides) given mesh region,
/// putting given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary;
/// \param verts must not include all vertices of a mesh connected component
MRMESH_API void inflate( Mesh& mesh, const VertBitSet& verts, const InflateSettings & settings );

} //namespace MR
