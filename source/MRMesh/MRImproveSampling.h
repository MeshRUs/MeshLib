#pragma once

#include "MRMeshFwd.h"
#include "MRProgressCallback.h"

namespace MR
{

struct ImproveSamplingSettings
{
    /// the number of algorithm iterations to perform
    int numIters = 1;

    /// if a sample represents less than this number of input points than such sample will be discarded
    int minPointsInSample = 1;

    /// output progress status and receive cancel signal
    ProgressCallback progress;
};

/// Finds more representative sampling starting from a given one following k-means method;
/// \param samples input and output selected sample points from \param cloud;
/// \return false if it was terminated by the callback
/// \ingroup PointCloudGroup
MRMESH_API bool improveSampling( const PointCloud & cloud, VertBitSet & samples, const ImproveSamplingSettings & settings );

} //namespace MR
