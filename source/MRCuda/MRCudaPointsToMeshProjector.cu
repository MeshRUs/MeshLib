#include "MRCudaPointsToMeshProjector.cuh"
#include "MRMesh/MRAABBTree.h"
#include "device_launch_parameters.h"

namespace MR { namespace Cuda {

__device__ float3 Matrix4::transform( const float3& pt ) const
{
    float3 res = { dot( x, pt ), dot( y, pt ), dot( z, pt ) };
    res = res + b;
    return res;
}

__device__ Box3 Matrix4::transform( const Box3& box ) const
{
    Box3 res;
    res.include( transform( float3{ box.min.x, box.min.y, box.min.z } ) );
    res.include( transform( float3{ box.min.x, box.min.y, box.max.z } ) );
    res.include( transform( float3{ box.min.x, box.max.y, box.min.z } ) );
    res.include( transform( float3{ box.min.x, box.max.y, box.max.z } ) );
    res.include( transform( float3{ box.max.x, box.min.y, box.min.z } ) );
    res.include( transform( float3{ box.max.x, box.min.y, box.max.z } ) );
    res.include( transform( float3{ box.max.x, box.max.y, box.min.z } ) );
    res.include( transform( float3{ box.max.x, box.max.y, box.max.z } ) );
    return res;
}

__device__ Matrix4 Matrix4::inverse() const
{
    const float det = x.x * ( y.y * z.z - y.z * z.y )
        - x.y * ( y.x * z.z - y.z * z.x )
        + x.z * ( y.x * z.y - y.y * z.x );

    if ( det == 0 )
        return {};

    return
    {
        { ( y.y * z.z - y.z * z.y ) / det, ( x.z * z.y - x.y * z.z ) / det, ( x.y * y.z - x.z * y.y ) / det },
        { ( y.z * z.x - y.x * z.z ) / det, ( x.x * z.z - x.z * z.x ) / det, ( x.z * y.x - x.x * y.z ) / det },
        { ( y.x * z.y - y.y * z.x ) / det, ( x.y * z.x - x.x * z.y ) / det, ( x.x * y.y - x.y * y.x ) / det }
    };
}

__device__ bool Node3::leaf() const
{
    return r < 0;
}

__device__ int Node3::leafId() const
{
    return l;
}

__device__ float3 Box3::getBoxClosestPointTo( const float3& pt ) const
{
    return { clamp( pt.x, min.x, max.x ), clamp( pt.y, min.y, max.y ), clamp( pt.z, min.z, max.z ) };
}

__device__ void Box3::include( const float3& pt )
{
    if ( pt.x < min.x ) min.x = pt.x;
    if ( pt.x > max.x ) max.x = pt.x;
    if ( pt.y < min.y ) min.y = pt.y;
    if ( pt.y > max.y ) max.y = pt.y;
    if ( pt.z < min.z ) min.z = pt.z;
    if ( pt.z > max.z ) max.z = pt.z;
}

struct ClosestPointRes
{
    float2 bary;
    float3 proj;
};

__device__ ClosestPointRes closestPointInTriangle( const float3& p, const float3& a, const float3& b, const float3& c )
{
    const float3 ab = b - a;
    const float3 ac = c - a;

    const Matrix4 transform{ ab, ac, cross( ab, ac ) };
    const Matrix4 invTransform = transform.inverse();
    const float3 pp = invTransform.transform( p - a );
    if ( pp.x < 0 )
    {
        if ( pp.y < 0 )
            return { { 0, 0 }, a + transform.transform( float3{ 0, 0 ,0 } ) };
        if ( pp.y < 1 )
            return { { 0, pp.y }, a + transform.transform( float3 {0, pp.y, 0 } ) };

        return { { 0, 1 }, a + transform.transform( float3{ 0, 1, 0 } ) };
    }

    if ( pp.y < 0 )
    {
        if ( pp.x < 1 )
            return { { pp.x, 0 }, a + transform.transform( float3 { pp.x, 0, 0 } ) };

        return { { 1, 0 }, a + transform.transform( float3{ 1, 0, 0 } ) };
    }

    if ( pp.y > 1 - pp.x )
    {
        if ( pp.y < pp.x - 1 )
            return { { 1, 0 }, a + transform.transform( float3{ 1, 0, 0 } ) };

        if ( pp.y > pp.x + 1 )
            return { { 0, 1 }, a + transform.transform( float3{ 0, 1, 0 } ) };

        const float x = ( pp.x - pp.y + 1 ) * 0.5f;
        const float y = ( pp.y - pp.x + 1 ) * 0.5f;
        return { { x, y }, a + transform.transform( float3{ x, y ,0 } ) };
    }
        
    return { {pp.x, pp.y}, a + transform.transform( float3{ pp.x,pp.y, 0 } ) };
}

__global__ void kernel( const float3* points, const Node3* nodes, const float3* meshPoints, const HalfEdgeRecord* edges, const int* edgePerFace, MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
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
    auto& res = resVec[index];
    res.distSq = upDistLimitSq;    
    res.mtp.edgeId = -1;
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
           
            int edge = edgePerFace[face];
            float3 a = meshPoints[edges[edge].org];
            edge = edges[ edge ^ 1 ].prev;
            float3 b = meshPoints[edges[edge].org];
            edge = edges[edge ^ 1].prev;
            float3 c = meshPoints[edges[edge].org];

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
                res.mtp = MeshTriPoint{ edgePerFace[face], closestPointRes.bary.x, closestPointRes.bary.y };
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
}

void meshProjectionKernel( const float3* points, 
                           const Node3* nodes, const float3* meshPoints, const HalfEdgeRecord* edges, const int* edgePerFace, 
                           MeshProjectionResult* resVec, const Matrix4 xf, const Matrix4 refXf, float upDistLimitSq, float loDistLimitSq, size_t size )
{
    int maxThreadsPerBlock = 0;
    cudaDeviceGetAttribute( &maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0 );
    int numBlocks = ( int( size ) + maxThreadsPerBlock - 1 ) / maxThreadsPerBlock;    
    kernel << <numBlocks, maxThreadsPerBlock >> > ( points, nodes, meshPoints, edges, edgePerFace, resVec, xf, refXf, upDistLimitSq, loDistLimitSq, size );
}

}}

