#pragma once

#include "MRTimer.h"
#include <vector>
#include <algorithm>

namespace MR
{

/// similar to std::priority_queue, but with ability to access underlying vector to custom modify its elements
template <typename T, typename P = std::less<T>>
class PriorityQueue
{
public:
    using value_type = T;
    using Container = std::vector<T>;
    using size_type = typename Container::size_type;

    /// constructs empty queue
    PriorityQueue() {}
    explicit PriorityQueue( const P& pred ) : pred_( pred ) {}

    /// initializes queue elements from given vector
    explicit PriorityQueue( const P& pred, Container&& v );

    /// checks if the queue has no elements
    bool empty() const { return c.empty(); }

    /// returns the number of elements
    size_type size() const { return c.size(); }

    /// accesses the top element
    const T & top() const { return c.front(); }

    /// inserts element in the queue
    void push( const value_type& value ) { c.push_back( value ); onPush_(); }
    void push( value_type&& value ) { c.push_back( std::move( value ) ); onPush_(); }
    template< class... Args >
    void emplace( Args&&... args ) { c.emplace_back( std::forward<Args>(args)... ); onPush_(); }

    /// removes the top element from the priority queue
    void pop() { std::pop_heap( c.begin(), c.end(), pred_ ); c.pop_back(); }

    /// intentionally left public to allow user access to it,
    /// but the user is responsible for restore heap-property of this vector before calling of any method
    Container c;

private:
    void onPush_() { std::push_heap( c.begin(), c.end(), pred_ ); };

private:
    [[no_unique_address]] P pred_;
};

template <typename T, typename P>
PriorityQueue<T, P>::PriorityQueue( const P& pred, Container&& v ) : c( std::move( v ) ), pred_( pred )
{
    MR_TIMER
    std::make_heap( c.begin(), c.end(), pred_ );
}

} // namespace MR
