#include "MRCudaPointsToMeshProjector.cuh"
#include "MRMesh/MRAABBTree.h"
#include "device_launch_parameters.h"

namespace MR { namespace Cuda {

__global__ void kernel( const float3* points,
    const Node3* __restrict__ nodes, const float3* __restrict__ meshPoints, const FaceToThreeVerts* __restrict__ faces,
    MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    if ( size == 0 )
    {
        assert( false );
        return;
    }

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= size )
        return;

    const auto pt = xf.isIdentity ? points[index] : xf.transform( points[index] );
    MeshProjectionResult res;
    res.distSq = upDistLimitSq;
    res.proj.faceId = -1;
    struct SubTask
    {
        int n;
        float distSq = 0;
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( int n )
    {
        const auto box = refXf.isIdentity ? nodes[n].box : refXf.transform( nodes[n].box );
        float distSq = lengthSq( box.getBoxClosestPointTo( pt ) - pt );
        return SubTask{ n, distSq };
    };

    addSubTask( getSubTask( 0 ) );
    
    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = nodes[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            const auto face = node.leafId();
            const auto & vs = faces[face].verts;
            float3 a = meshPoints[vs[0]];
            float3 b = meshPoints[vs[1]];
            float3 c = meshPoints[vs[2]];

            if ( !refXf.isIdentity )
            {
                a = refXf.transform( a );
                b = refXf.transform( b );
                c = refXf.transform( c );
            }
            
            // compute the closest point in double-precision, because float might be not enough
            const auto closestPointRes = closestPointInTriangle( pt, a, b, c );

            float distSq = lengthSq( closestPointRes.proj - pt );
            if ( distSq < res.distSq )
            {
                res.distSq = distSq;
                res.proj.point = closestPointRes.proj;
                res.proj.faceId = face;
                res.tp = MeshTriPoint{ -1, closestPointRes.bary.x, closestPointRes.bary.y };
                if ( distSq <= loDistLimitSq )
                    break;
            }
            continue;
        }

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
        if ( s1.distSq < s2.distSq )
        {
            const auto temp = s1;
            s1 = s2;
            s2 = temp;
        }
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }
    resVec[index] = res;
}

void meshProjectionKernel( int maxThreadsPerBlock, const float3* points, 
                           const Node3* nodes, const float3* meshPoints, const FaceToThreeVerts* faces,
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;
    kernel << <numBlocks, maxThreadsPerBlock >> > ( points, nodes, meshPoints, faces, resVec, xf, refXf, upDistLimitSq, loDistLimitSq, size );
}

}}

