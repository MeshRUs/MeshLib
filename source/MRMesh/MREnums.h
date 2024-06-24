#pragma once

namespace MR
{

/// determines the weight of each edge in applications like Laplacian
enum class EdgeWeights
{
    /// all edges have same weight=1
    Unit = 0,

    /// edge weight depends on local geometry and uses cotangent values
    Cotan,

    /// [deprecated] edge weight is equal to edge length times cotangent weight
    CotanTimesLength,

    /// cotangent edge weights and equation weights inversely proportional to square root of local area
    CotanWithAreaEqWeight
};

} //namespace MR
