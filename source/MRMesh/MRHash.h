#pragma once

#include "MRId.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include <cstdint>
#include <cstring>
#include <functional>

namespace std
{

template <typename T>
struct hash<MR::Id<T>>
{
    size_t operator() ( MR::Id<T> const& p ) const noexcept
    {
        return (int)p;
    }
};

template<>
struct hash<MR::Vector2f>
{
    size_t operator()( MR::Vector2f const& p ) const noexcept
    {
        std::uint64_t xy;
        static_assert( sizeof( float ) == sizeof( std::uint32_t ) );
        std::memcpy( &xy, &p.x, sizeof( std::uint64_t ) );
        return size_t( xy );
    }
};

template<>
struct hash<MR::Vector3f>
{
    size_t operator()( MR::Vector3f const& p ) const noexcept
    {
        // standard implementation is slower:
        // phmap::HashState().combine(phmap::Hash<float>()(p.x), p.y, p.z);
        std::uint64_t xy;
        std::uint32_t z;
        static_assert( sizeof( float ) == sizeof( std::uint32_t ) );
        std::memcpy( &xy, &p.x, sizeof( std::uint64_t ) );
        std::memcpy( &z, &p.z, sizeof( std::uint32_t ) );
        return size_t( xy ) ^ ( size_t( z ) << 16 );
    }
};

} // namespace std
