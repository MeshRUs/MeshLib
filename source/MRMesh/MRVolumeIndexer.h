#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRId.h"

#include <array>

namespace MR
{

/// \defgroup VoxelGroup Voxel
/// \brief This chapter represents documentation about Volume (Voxel)
/// \{

/// \defgroup VolumeIndexerGroup Volume Indexer
/// \ingroup VoxelGroup
/// \{

enum class OutEdge : signed char
{
    Invalid = -1,
    PlusZ = 0,
    MinusZ,
    PlusY,
    MinusY,
    PlusX,
    MinusX,
    Count
};

static_assert( sizeof( OutEdge ) == 1 );

static const std::initializer_list<OutEdge> all6Edges = { OutEdge::PlusZ, OutEdge::MinusZ, OutEdge::PlusY, OutEdge::MinusY, OutEdge::PlusX, OutEdge::MinusX };

inline OutEdge opposite( OutEdge e )
{
    const static std::array<OutEdge, 7> map{ OutEdge::Invalid, OutEdge::MinusZ, OutEdge::PlusZ, OutEdge::MinusY, OutEdge::PlusY, OutEdge::MinusX, OutEdge::PlusX };
    return map[ (size_t)e + 1 ];
}

static constexpr int OutEdgeCount = 6;

class VolumeIndexer
{
public:
    VolumeIndexer( const Vector3i & dims );
    const Vector3i & dims() const { return dims_; }
    size_t size() const { return size_; }
    size_t sizeXY() const { return sizeXY_; }
    Vector3i toPos( VoxelId id ) const;
    VoxelId toVoxelId( const Vector3i & pos ) const;
    /// returns true if this voxel is within dimensions
    bool isInDims( const Vector3i& pos ) const { return pos.x >= 0 && pos.x < dims_.x && pos.y >= 0 && pos.y < dims_.y && pos.z >= 0 && pos.z < dims_.z; }
    /// returns true if this voxel is on the boundary of the volume
    bool isBdVoxel( const Vector3i & pos ) const { return pos.x == 0 || pos.x + 1 == dims_.x || pos.y == 0 || pos.y + 1 == dims_.y || pos.z == 0 || pos.z + 1 == dims_.z; }
    /// returns true if v1 is within at most 6 neighbors of v0
    bool areNeigbors( VoxelId v0, VoxelId v1 ) const { return areNeigbors( toPos( v0 ), toPos( v1 ) ); }
    bool areNeigbors( const Vector3i & pos0, const Vector3i & pos1 ) const { return ( pos0 - pos1 ).lengthSq() == 1; }
    /// returns id of v's neighbor specified by the edge
    VoxelId getNeighbor( VoxelId v, OutEdge toNei ) const { return getNeighbor( v, toPos( v ), toNei ); }
    MRMESH_API VoxelId getNeighbor( VoxelId v, const Vector3i & pos, OutEdge toNei ) const;
    /// returns id of v's neighbor specified by the edge, which is known to exist (so skipping a lot of checks)
    VoxelId getExistingNeighbor( VoxelId v, OutEdge toNei ) const;
    VoxelId getNeighbor( VoxelId v, const Vector3i & pos, bool bdPos, OutEdge toNei ) const { return bdPos ? getNeighbor( v, pos, toNei ) : getExistingNeighbor( v, toNei ); }

protected:
    Vector3i dims_;
    size_t sizeXY_ = 0; ///< = dims_.x * dims_.y
    size_t size_ = 0; ///< = dims_.x * dims_.y * dims_.z
    int neiInc_[ OutEdgeCount ] = {};
};

inline VolumeIndexer::VolumeIndexer( const Vector3i & dims ) 
    : dims_( dims )
    , sizeXY_( size_t( dims_.x ) * dims_.y ) 
    , size_( sizeXY_ * dims_.z ) 
{ 
    neiInc_[(int)OutEdge::PlusZ]  = (int)sizeXY_;
    neiInc_[(int)OutEdge::MinusZ] =-(int)sizeXY_;
    neiInc_[(int)OutEdge::PlusY] =  dims_.x;
    neiInc_[(int)OutEdge::MinusY] =-dims_.x;
    neiInc_[(int)OutEdge::PlusX] =  1;
    neiInc_[(int)OutEdge::MinusX] =-1;
}

inline Vector3i VolumeIndexer::toPos( VoxelId id ) const
{
    assert( id.valid() );
    int z = int( id / sizeXY_ );
    int sumZ = int( id % sizeXY_ );
    int y = sumZ / dims_.x;
    int x = sumZ % dims_.x;
    return {x,y,z};
}

inline VoxelId VolumeIndexer::toVoxelId( const Vector3i & pos ) const
{
    return VoxelId{ pos.x + pos.y * size_t(dims_.x) + pos.z * sizeXY_ };
}

inline VoxelId VolumeIndexer::getExistingNeighbor( VoxelId v, OutEdge toNei ) const
{ 
    assert( toNei > OutEdge::Invalid && toNei < OutEdge::Count );
    return VoxelId( size_t( v ) + neiInc_[(int)toNei] ); 
}

/// expands VoxelBitSet with given number of steps
MRMESH_API void expandVoxelsMask( VoxelBitSet& mask, const VolumeIndexer& indexer, int expansion = 1 );
/// shrinks VoxelBitSet with given number of steps
MRMESH_API void shrinkVoxelsMask( VoxelBitSet& mask, const VolumeIndexer& indexer, int shrinkage = 1 );

/// \}

} // namespace MR
