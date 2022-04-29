#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRphmap.h"
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#include <boost/dynamic_bitset.hpp>
#include <iterator>

namespace MR
{

/**
 * \defgroup BasicGroup Basic elements overview
 * \brief This chapter represents documentation about basic elements
 * \{
 */

/// container of bits
class BitSet : public boost::dynamic_bitset<std::uint64_t>
{
    using base = boost::dynamic_bitset<std::uint64_t>;
public:
    using base::base;
    using IndexType = size_t;

    [[nodiscard]] bool test( IndexType n ) const { return n < size() && base::test( n ); }

    MRMESH_API BitSet & operator &= ( const BitSet & b );
    MRMESH_API BitSet & operator |= ( const BitSet & b );
    MRMESH_API BitSet & operator ^= ( const BitSet & b );
    MRMESH_API BitSet & operator -= ( const BitSet & b );

    /// return the highest index i such as bit i is set, or npos if *this has no on bits. 
    [[nodiscard]] MRMESH_API IndexType find_last() const;

    /// this accessor automatically adjusts the size of the set to include i-th element
    void autoResizeSet( size_t pos, bool val = true )
    {
        auto sz = size();
        if ( pos == sz )
        {
            push_back( val );
            return;
        }
        if ( pos > sz )
        {
            if ( capacity() <= pos && sz > 0 )
            {
                while ( pos < sz )
                    sz <<= 1;
                reserve( sz );
            }
            resize( pos + 1 );
        }
        set( pos, val );
    }

    /// same as \ref autoResizeSet and returns previous value of pos-bit
    [[nodiscard]] bool autoResizeTestSet( size_t pos, bool val = true )
    {
        bool const b = test( pos );
        if ( b != val )
            autoResizeSet( pos, val );
        return b;
    }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return capacity() / 8; }
};

/// container of bits representing specific indices (faces, verts or edges)
template <typename T>
class TaggedBitSet : public BitSet
{
    using base = BitSet;
public:
    using base::base;
    using IndexType = Id<T>;

    TaggedBitSet & set( IndexType n, size_type len, bool val ) { base::set( n, len, val ); return * this; }
    TaggedBitSet & set( IndexType n, bool val = true ) { base::set( n, val ); return * this; }
    TaggedBitSet & set() { base::set(); return * this; }
    TaggedBitSet & reset( IndexType n, size_type len ) { base::reset( n, len ); return * this; }
    TaggedBitSet & reset( IndexType n ) { base::reset( n ); return * this; }
    TaggedBitSet & reset() { base::reset(); return * this; }
    TaggedBitSet & flip( IndexType n, size_type len ) { base::flip( n, len ); return * this; }
    TaggedBitSet & flip( IndexType n ) { base::flip( n ); return * this; }
    TaggedBitSet & flip() { base::flip(); return * this; }
    [[nodiscard]] bool test( IndexType n ) const { return base::test( n ); }
    [[nodiscard]] bool test_set( IndexType n, bool val = true ) { return base::test_set( n, val ); }

    [[nodiscard]] reference operator[]( IndexType pos ) { return base::operator[]( pos ); }
    [[nodiscard]] bool operator[]( IndexType pos ) const { return base::operator[]( pos ); }

    [[nodiscard]] IndexType find_first() const { return IndexType( base::find_first() ); }
    [[nodiscard]] IndexType find_next( IndexType pos ) const { return IndexType( base::find_next( pos ) ); }
    [[nodiscard]] IndexType find_last() const { return IndexType( base::find_last() ); }

    TaggedBitSet & operator &= ( const TaggedBitSet & b ) { base::operator &= ( b ); return * this; }
    TaggedBitSet & operator |= ( const TaggedBitSet & b ) { base::operator |= ( b ); return * this; }
    TaggedBitSet & operator ^= ( const TaggedBitSet & b ) { base::operator ^= ( b ); return * this; }
    TaggedBitSet & operator -= ( const TaggedBitSet & b ) { base::operator -= ( b ); return * this; }

    void autoResizeSet( IndexType pos, bool val = true ) { base::autoResizeSet( pos, val ); }
    [[nodiscard]] bool autoResizeTestSet( IndexType pos, bool val = true ) { return base::autoResizeTestSet( pos, val ); }

    /// constructs another bit set from this where every set bit index is transformed using given map
    [[nodiscard]] TaggedBitSet getMapping( const Vector<IndexType, IndexType> & map ) const;
    [[nodiscard]] TaggedBitSet getMapping( const HashMap<IndexType, IndexType> & map ) const;
    /// this is a faster version if the result size is known beforehand
    [[nodiscard]] TaggedBitSet getMapping( const Vector<IndexType, IndexType> & map, size_t resSize ) const;
    [[nodiscard]] TaggedBitSet getMapping( const HashMap<IndexType, IndexType> & map, size_t resSize ) const;
};

template <typename T>
[[nodiscard]] inline bool contains( const TaggedBitSet<T> * bitset, Id<T> id )
{
    return id.valid() && ( !bitset || bitset->test( id ) );
}

template <typename T>
[[nodiscard]] inline bool contains( const TaggedBitSet<T> & bitset, Id<T> id )
{
    return id.valid() && bitset.test( id );
}

/// iterator to enumerate all indices with set bits in BitSet class or its derivatives
template <typename T>
class SetBitIteratorT
{
public:
    using IndexType = typename T::IndexType;
    
    using iterator_category = std::forward_iterator_tag;
    using value_type        = IndexType;
    using difference_type   = std::ptrdiff_t;
    using reference         = const IndexType; ///< intentionally not a reference
    using pointer           = const IndexType *;

    /// constructs end iterator
    SetBitIteratorT() = default;
    /// constructs begin iterator
    SetBitIteratorT( const T & bitset )
        : bitset_( &bitset ), index_( bitset.find_first() )
    {
    }
    SetBitIteratorT & operator++( )
    {
        index_ = bitset_->find_next( index_ );
        return * this;
    }
    [[nodiscard]] SetBitIteratorT operator++( int )
    {
        SetBitIteratorT ret = *this;
        operator++();
        return ret;
    }
    
    [[nodiscard]] const T * bitset() const { return bitset_; }
    [[nodiscard]] reference operator *() const { return index_; }

private:
    const T * bitset_ = nullptr;
    IndexType index_ = IndexType(-1);
};

template <typename T>
[[nodiscard]] inline bool operator ==( const SetBitIteratorT<T> & a, const SetBitIteratorT<T> & b )
    { return *a == *b; }

template <typename T>
[[nodiscard]] inline bool operator !=( const SetBitIteratorT<T> & a, const SetBitIteratorT<T> & b )
    { return *a != *b; }


[[nodiscard]] inline auto begin( const BitSet & a )
    { return SetBitIteratorT<BitSet>(a); }
[[nodiscard]] inline auto end( const BitSet & )
    { return SetBitIteratorT<BitSet>(); }

template <typename T>
[[nodiscard]] inline auto begin( const TaggedBitSet<T> & a )
    { return SetBitIteratorT<TaggedBitSet<T>>(a); }
template <typename T>
[[nodiscard]] inline auto end( const TaggedBitSet<T> & )
    { return SetBitIteratorT<TaggedBitSet<T>>(); }

template <typename T>
[[nodiscard]] TaggedBitSet<T> TaggedBitSet<T>::getMapping( const Vector<IndexType, IndexType> & map ) const
{
    TaggedBitSet<T> res;
    for ( auto b : *this )
        if ( auto mapped = map[b] )
            res.autoResizeSet( mapped );
    return res;
}

template <typename T>
[[nodiscard]] TaggedBitSet<T> TaggedBitSet<T>::getMapping( const HashMap<IndexType, IndexType> & map ) const
{
    TaggedBitSet<T> res;
    for ( auto b : *this )
        if ( auto mapped = getAt( map, b ) )
            res.autoResizeSet( mapped );
    return res;
}

template <typename T>
[[nodiscard]] TaggedBitSet<T> TaggedBitSet<T>::getMapping( const Vector<IndexType, IndexType> & map, size_t resSize ) const
{
    TaggedBitSet<T> res( resSize );
    for ( auto b : *this )
        if ( auto mapped = map[b] )
            res.set( mapped );
    return res;
}

template <typename T>
[[nodiscard]] TaggedBitSet<T> TaggedBitSet<T>::getMapping( const HashMap<IndexType, IndexType> & map, size_t resSize ) const
{
    TaggedBitSet<T> res( resSize );
    for ( auto b : *this )
        if ( auto mapped = getAt( map, b ) )
            res.set( mapped );
    return res;
}

[[nodiscard]] inline BitSet operator & ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res &= b; return res; }
[[nodiscard]] inline BitSet operator | ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res |= b; return res; }
[[nodiscard]] inline BitSet operator ^ ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res ^= b; return res; }
[[nodiscard]] inline BitSet operator - ( const BitSet & a, const BitSet & b ) { BitSet res{ a }; res -= b; return res; }

template <typename T> [[nodiscard]] inline TaggedBitSet<T> operator & ( const TaggedBitSet<T> & a, const TaggedBitSet<T> & b ) { auto res{ a }; res &= b; return res; }
template <typename T> [[nodiscard]] inline TaggedBitSet<T> operator | ( const TaggedBitSet<T> & a, const TaggedBitSet<T> & b ) { auto res{ a }; res |= b; return res; }
template <typename T> [[nodiscard]] inline TaggedBitSet<T> operator ^ ( const TaggedBitSet<T> & a, const TaggedBitSet<T> & b ) { auto res{ a }; res ^= b; return res; }
template <typename T> [[nodiscard]] inline TaggedBitSet<T> operator - ( const TaggedBitSet<T> & a, const TaggedBitSet<T> & b ) { auto res{ a }; res -= b; return res; }

/// \}

} // namespace MR
