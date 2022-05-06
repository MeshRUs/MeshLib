#pragma once
#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

struct RelaxParams
{
    // number of iterations
    int iterations{ 1 };
    // region to relax
    const VertBitSet* region{ nullptr };
    // speed of relaxing, typical values (0.0, 0.5]
    float force{ 0.5f };
};

// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
// returns true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relax( Mesh& mesh, const RelaxParams params = {}, ProgressCallback cb = {} );

// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
// do not really keeps volume but tries hard
// returns true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxKeepVolume( Mesh& mesh, const RelaxParams params = {}, ProgressCallback cb = {} );

enum class RelaxApproxType 
{
    Planar,
    Quadric
};

struct MeshApproxRelaxParams : RelaxParams
{
    // radius to find neighbors by surface
    // 0.0f - default = 1e-3 * sqrt(surface area)
    float surfaceDilateRadius{ 0.0f };
    RelaxApproxType type{ RelaxApproxType::Planar };
};

// applies given number of relaxation iterations to the whole mesh ( or some region if it is specified )
// approx neighborhoods
// returns true if was finished successfully, false if was interrupted by progress callback
MRMESH_API bool relaxApprox( Mesh& mesh, const MeshApproxRelaxParams params = {}, ProgressCallback cb = {} );

// applies at most given number of relaxation iterations the spikes detected by given threshold
MRMESH_API void removeSpikes( Mesh & mesh, int maxIterations, float minSumAngle, const VertBitSet * region = nullptr );

}
