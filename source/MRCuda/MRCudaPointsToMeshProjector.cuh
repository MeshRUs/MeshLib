#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"
namespace MR
{

namespace Cuda
{

// GPU analog of CPU MeshProjectionResult struct
struct MeshProjectionResult
{
    PointOnFace proj;
    MeshTriPoint tp;
    float distSq;
};

// calls mesh projection kernel for each point in parallel
void meshProjectionKernel( int maxThreadsPerBlock, const float3* points,
                           const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size );

}

}