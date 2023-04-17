#pragma once
#include "cuda_runtime.h"

namespace MR
{

namespace Cuda
{

__device__ inline float clamp( float x, float l, float u )
{
    return ( ( x < l ) ? l :
             ( ( x > u ) ? u : x ) );
}

__device__ inline float2 operator+( const float2& a, const float2& b )
{
    return { a.x + b.x, a.y + b.y };
}

__device__ inline float2 operator-( const float2& a, const float2& b )
{
    return { a.x - b.x, a.y - b.y };
}

__device__ inline float2 operator*( const float2& a, const float k )
{
    return { k * a.x , k * a.y };
}

__device__ inline float lengthSq( const float2& a )
{
    return a.x * a.x + a.y * a.y;
}

__device__ inline float dot( const float2& a, const float2& b )
{
    return a.x * b.x + a.y * b.y;
}

__device__ inline float3 operator+( const float3& a, const float3& b )
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ inline float3 operator-( const float3& a, const float3& b )
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ inline float3 operator*( const float3& a, const float k )
{
    return { k * a.x , k * a.y, k * a.z };
}

__device__ inline float lengthSq( const float3& a )
{
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ inline float dot( const float3& a, const float3& b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross( const float3& a, const float3& b )
{
    return { a.y* b.z - a.z * b.y,
             a.z* b.x - a.x * b.z,
             a.x* b.y - a.y * b.x };
}


}
}