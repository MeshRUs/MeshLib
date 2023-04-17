#pragma once
#include "MRCudaFloat.cuh"
#include "MRCudaBasic.cuh"

namespace MR
{

namespace Cuda
{

// struct simular to MR::Box3 with minimal needed functions
struct Box3
{
    float3 min;
    float3 max;

    __device__ float3 getBoxClosestPointTo( const float3& pt ) const;
    __device__ void include( const float3& pt );
};

// struct simular to AABBTreeNode<FaceTreeTraits3> with minimal needed functions
struct Node3
{
    Box3 box;
    int l, r;

    __device__ bool leaf() const;
    __device__ int leafId() const;
};

// mesh topology data, maby can be simplified to have less data transfers between CPU and GPU
struct HalfEdgeRecord
{
    int next;
    int prev;
    int org;
    int left;
};

// GPU analog of CPU PointOnFace struct
struct PointOnFace
{
    int faceId;
    float3 point;
};

// GPU analog of CPU MeshTriPoint struct
struct MeshTriPoint
{
    int edgeId;
    float a;
    float b;
};

// GPU analog of CPU MeshProjectionResult struct
struct MeshProjectionResult
{
    PointOnFace proj;
    MeshTriPoint mtp;
    float distSq;
};

// GPU analog of CPU AffineXf3f
struct Matrix4
{
    float3 x = { 1.f, 0.f, 0.f };
    float3 y = { 0.f, 1.f, 0.f };
    float3 z = { 0.f, 0.f, 1.f };
    float3 b = { 0.f, 0.f, 0.f };
    bool isIdentity = true;
    /// application of the transformation to a point
    __device__ float3 transform( const float3& pt ) const;
    /// application of the transformation to a box
    __device__ Box3 transform( const Box3& box ) const;

    __device__ Matrix4 inverse() const;
};

// calls mesh projection kernel for each point in parallel
void meshProjectionKernel( const float3* points,
                           const Node3* nodes, const float3* meshPoints, const HalfEdgeRecord* edges, const int* edgePerFace,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size );

}

}