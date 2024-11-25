#include "MRBitSet.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRBitSet.h"

using namespace MR;

REGISTER_AUTO_CAST( BitSet )
REGISTER_AUTO_CAST( EdgeBitSet )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST( VertBitSet )

MRBitSet* mrBitSetCopy( const MRBitSet* bs_ )
{
    ARG( bs );
    RETURN_NEW( bs );
}

const uint64_t* mrBitSetBlocks( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.bits().data();
}

size_t mrBitSetBlocksNum( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.bits().size();
}

size_t mrBitSetSize( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.size();
}

void mrBitSetFree( MRBitSet* bs_ )
{
    ARG_PTR( bs );
    delete bs;
}

MRFaceBitSet* mrFaceBitSetCopy( const MRFaceBitSet* fbs_ )
{
    ARG( fbs );
    RETURN_NEW( fbs );
}

void mrFaceBitSetFree( MRFaceBitSet* fbs_ )
{
    ARG_PTR( fbs );
    delete fbs;
}

bool mrBitSetEq( const MRBitSet* a_, const MRBitSet* b_ )
{
    ARG( a ); ARG( b );
    return a == b;
}

MRFaceBitSet* mrFaceBitSetNew( size_t numBits, bool fillValue )
{
    RETURN_NEW( FaceBitSet( numBits, fillValue ) );
}

size_t mrBitSetCount( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.count();
}

size_t mrBitSetFindFirst( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.find_first();
}

size_t mrBitSetFindLast( const MRBitSet* bs_ )
{
    ARG( bs );
    return bs.find_last();
}

void mrBitSetResize( MRBitSet* bs_, size_t size, bool value )
{
    ARG( bs );
    bs.resize( size, value );
}

void mrBitSetAutoResizeSet( MRBitSet* bs_, size_t pos, bool value )
{
    ARG( bs );
    bs.autoResizeSet( pos, value );
}

MRBitSet* mrBitSetSub( const MRBitSet* a_, const MRBitSet* b_ )
{
    ARG( a ); ARG( b );
    RETURN_NEW( a - b );
}

MRBitSet* mrBitSetOr( const MRBitSet* a_, const MRBitSet* b_ )
{
    ARG( a ); ARG( b );
    RETURN_NEW( a | b );
}

MRBitSet* mrBitSetNew( size_t numBits, bool fillValue )
{
    RETURN_NEW( BitSet( numBits, fillValue ) );
}

MREdgeBitSet* mrEdgeBitSetNew( size_t numBits, bool fillValue )
{
    RETURN_NEW( EdgeBitSet( numBits, fillValue ) );
}

MREdgeBitSet* mrEdgeBitSetCopy( const MREdgeBitSet* ebs_ )
{
    ARG( ebs );
    RETURN_NEW( ebs );
}

void mrEdgeBitSetFree( MREdgeBitSet* ebs_ )
{
    ARG_PTR( ebs );
    delete ebs;
}

MRVertBitSet* mrVertBitSetNew( size_t numBits, bool fillValue )
{
    RETURN_NEW( VertBitSet( numBits, fillValue ) );
}

MRVertBitSet* mrVertBitSetCopy( const MRVertBitSet* vbs_ )
{
    ARG( vbs );
    RETURN_NEW( vbs );
}

void mrVertBitSetFree( MRVertBitSet* vbs_ )
{
    ARG_PTR( vbs );
    delete vbs;
}

MRUndirectedEdgeBitSet* mrUndirectedEdgeBitSetNew( size_t numBits, bool fillValue )
{
    RETURN_NEW( UndirectedEdgeBitSet( numBits, fillValue ) );
}

MRUndirectedEdgeBitSet* mrUndirectedEdgeBitSetCopy( const MRUndirectedEdgeBitSet* uebs_ )
{
    ARG( uebs );
    RETURN_NEW( uebs );
}

void mrUndirectedEdgeBitSetFree( MRUndirectedEdgeBitSet* uebs_ )
{
    ARG_PTR( uebs );
    delete uebs;
}

bool mrBitSetTest( const MRBitSet* bs_, size_t index )
{
    ARG( bs );
    return bs.test( index );
}

void mrBitSetSet( MRBitSet* bs_, size_t index, bool value )
{
    ARG( bs );
    bs.set( index, value );
}

size_t mrBitSetNpos( void )
{
    return BitSet::npos;
}
