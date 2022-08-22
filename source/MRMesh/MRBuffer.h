#pragma once

#include "MRMeshFwd.h"
#include <cassert>
#include <memory>

namespace MR
{

/**
 * \brief std::vector<T>-like container that is
 *  1) resized without initialization of its elements,
 *  2) much simplified: no push_back and many other methods
 * \tparam T type of stored elements
 * \ingroup BasicGroup
 */
template <typename T>
class Buffer
{
public:
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    Buffer() = default;
    explicit Buffer( size_t size ) { resize( size ); }
    Buffer( std::vector<T> && vec ) : data_( std::move( data_ ) ), size_( std::move( size_ ) ) { }

    void clear() { data_.reset(); size_ = 0; }

    [[nodiscard]] auto size() const { return size_; }

    void resize( size_t newSize ) 
    {
        if ( newSize == 0 )
            clear();
        else if ( newSize != size_ )
            data_ = std::make_unique_for_overwrite<T[]>( size_ = newSize );
    }

    [[nodiscard]] const_reference operator[]( size_t i ) const
    {
        assert( i < size_ );
        return data_[i];
    }
    [[nodiscard]] reference operator[]( size_t i )
    {
        assert( i < size_ );
        return data_[i];
    }

    [[nodiscard]] auto data() { return data_.get(); }
    [[nodiscard]] auto data() const { return data_.get(); }

    void swap( Buffer & b ) { data_.swap( b.data_ ); std::swap( size_, b.size_ ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return size() * sizeof(T); }

private:
    std::unique_ptr<T[]> data_;
    size_t size_ = 0;
};

template <typename T, typename I>
[[nodiscard]] inline auto begin( const Buffer<T> & a )
    { return a.data(); }

template <typename T, typename I>
[[nodiscard]] inline auto begin( Buffer<T> & a )
    { return a.data(); }

template <typename T, typename I>
[[nodiscard]] inline auto end( const Buffer<T> & a )
    { return a.data() + a.size(); }

template <typename T, typename I>
[[nodiscard]] inline auto end( Buffer<T> & a )
    { return a.data() + a.size(); }

} // namespace MR
