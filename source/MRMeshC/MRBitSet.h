#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/// creates a copy of a BitSet object
MRMESHC_API MRBitSet* mrBitSetCopy( const MRBitSet* bs );

/// creates bitset of given size filled with given value
MRMESHC_API MRBitSet* mrBitSetNew( size_t numBits, bool fillValue );

/// gets read-only access to the underlying blocks of a bitset
MRMESHC_API const uint64_t* mrBitSetBlocks( const MRBitSet* bs );

/// gets count of the underlying blocks of a bitset
MRMESHC_API size_t mrBitSetBlocksNum( const MRBitSet* bs );

/// gets total length of a bitset
MRMESHC_API size_t mrBitSetSize( const MRBitSet* bs );

/// returns the number of bits in this bitset that are set
MRMESHC_API size_t mrBitSetCount( const MRBitSet* bs );

/// checks if two bitsets are equal (have the same length and identical bit values)
MRMESHC_API bool mrBitSetEq( const MRBitSet* a, const MRBitSet* b );

/// returns the value of the specified bit
MRMESHC_API bool mrBitSetTest( const MRBitSet* bs, size_t index );

/// sets the value of the specified bit
MRMESHC_API void mrBitSetSet( MRBitSet* bs, size_t index, bool value );

/// returns special value representing the non-existent index
MRMESHC_API size_t mrBitSetNpos( void );

/// return the lowest index i such as bit i is set, or npos if *this has no on bits.
MRMESHC_API size_t mrBitSetFindFirst( const MRBitSet* bs );

/// return the highest index i such as bit i is set, or npos if *this has no on bits.
MRMESHC_API size_t mrBitSetFindLast( const MRBitSet* bs );

/// resizes the bitset
MRMESHC_API void mrBitSetResize( MRBitSet* bs, size_t size, bool value );

/// sets element pos to given value, adjusting the size of the set to include new element if necessary
MRMESHC_API void mrBitSetAutoResizeSet( MRBitSet* bs, size_t pos, bool value );

/// creates a new bitset including a's bits and excluding b's bits
MRMESHC_API MRBitSet* mrBitSetSub( const MRBitSet* a, const MRBitSet* b );

/// creates a new bitset including both a's bits and b's bits
MRMESHC_API MRBitSet* mrBitSetOr( const MRBitSet* a, const MRBitSet* b );

/// deallocates a BitSet object
MRMESHC_API void mrBitSetFree( MRBitSet* bs );

/// creates a new EdgeBitSet object
MRMESHC_API MREdgeBitSet* mrEdgeBitSetNew( size_t numBits, bool fillValue );

/// creates a copy of a EdgeBitSet object
MRMESHC_API MREdgeBitSet* mrEdgeBitSetCopy( const MREdgeBitSet* ebs );

/// deallocates a EdgeBitSet object
MRMESHC_API void mrEdgeBitSetFree( MREdgeBitSet* ebs );

/// creates a new UndirectedEdgeBitSet object
MRMESHC_API MRUndirectedEdgeBitSet* mrUndirectedEdgeBitSetNew( size_t numBits, bool fillValue );

/// creates a copy of a UndirectedEdgeBitSet object
MRMESHC_API MRUndirectedEdgeBitSet* mrUndirectedEdgeBitSetCopy( const MRUndirectedEdgeBitSet* uebs );

/// deallocates a UndirectedEdgeBitSet object
MRMESHC_API void mrUndirectedEdgeBitSetFree( MRUndirectedEdgeBitSet* uebs );

/// creates a new FaceBitSet object
MRMESHC_API MRFaceBitSet* mrFaceBitSetNew( size_t numBits, bool fillValue );

/// creates a copy of a FaceBitSet object
MRMESHC_API MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs );

/// deallocates a FaceBitSet object
MRMESHC_API void mrFaceBitSetFree( MRFaceBitSet* fbs );

/// creates a new VertBitSet object
MRMESHC_API MRVertBitSet* mrVertBitSetNew( size_t numBits, bool fillValue );

/// creates a copy of a VertBitSet object
MRMESHC_API MRVertBitSet* mrVertBitSetCopy( const MRVertBitSet* vbs );

/// deallocates a VertBitSet object
MRMESHC_API void mrVertBitSetFree( MRVertBitSet* vbs );

MR_EXTERN_C_END
