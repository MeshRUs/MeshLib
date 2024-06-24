#include "MRAABBTreeMaker.h"
#include "MRAABBTreeNode.h"
#include "MRBuffer.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include <stack>
#include <thread>

namespace MR
{

template<typename T>
struct Subtree
{
    Subtree( NodeId root, int f, int n ) : root( root ), firstLeaf( f ), numLeaves( n ) { }
    NodeId root; // of subtree
    int firstLeaf = 0;
    int numLeaves = 0;
    NodeId lastNode() const { return root + getNumNodes( numLeaves ); }
    bool leaf() const { assert( numLeaves >= 1 );  return numLeaves == 1; }
};

template<typename T>
class AABBTreeMaker
{
public:
    using Node = AABBTreeNode<T>;
    using NodeVec = Vector<Node, NodeId>;
    using Subtree = MR::Subtree<T>;
    using BoxT = typename T::BoxT;

    NodeVec construct( Buffer<BoxedLeaf<T>> boxedLeaves );

private:
    Buffer<BoxedLeaf<T>> boxedLeaves_;
    NodeVec nodes_;

private:
    // [firstLeaf, result) will go to left child and [result, lastLeaf) - to the right child
    int partitionLeaves_( const BoxT & box, int firstLeaf, int lastLeaf );
    // constructs not-leaf node
    std::pair<Subtree, Subtree> makeNode_( const Subtree & s );
    // constructs given subtree, optionally splitting the job on given number of threads
    void makeSubtree_( const Subtree & s, int numThreads );
};

template<typename T>
int AABBTreeMaker<T>::partitionLeaves_( const BoxT & box, int firstLeaf, int lastLeaf )
{
    assert( firstLeaf + 1 < lastLeaf );
    auto boxDiag = box.max - box.min;
    const int splitDim = int( std::max_element( begin( boxDiag ), end( boxDiag ) ) - begin( boxDiag ) );

    int midLeaf = firstLeaf + ( lastLeaf - firstLeaf ) / 2;
    std::nth_element( boxedLeaves_.data() + firstLeaf, boxedLeaves_.data() + midLeaf, boxedLeaves_.data() + lastLeaf,
        [&]( const BoxedLeaf<T> & a, const BoxedLeaf<T> & b )
        {
            return a.box.min[splitDim] + a.box.max[splitDim] < b.box.min[splitDim] + b.box.max[splitDim];
        } );
    return midLeaf;
}

template<typename T>
auto AABBTreeMaker<T>::makeNode_( const Subtree & s ) -> std::pair<Subtree, Subtree>
{
    assert( !s.leaf() );
    auto & node = nodes_[s.root];
    assert( !node.box.valid() );
    for ( size_t i = 0; i < s.numLeaves; ++i )
        node.box.include( boxedLeaves_[s.firstLeaf + i].box );

    const int midLeaf = partitionLeaves_( node.box, s.firstLeaf, s.firstLeaf + s.numLeaves );
    const int leftNumLeaves = midLeaf - s.firstLeaf;
    const int rightNumLeaves = s.numLeaves - leftNumLeaves;
    node.l = s.root + 1;
    node.r = s.root + 1 + getNumNodes( leftNumLeaves );
    return
    {
        Subtree( node.l, s.firstLeaf, leftNumLeaves ),
        Subtree( node.r, midLeaf,     rightNumLeaves )
    };
}

template<typename T>
void AABBTreeMaker<T>::makeSubtree_( const Subtree & s, int numThreads )
{
    assert( s.root + getNumNodes( s.numLeaves ) <= nodes_.size() );

    if ( numThreads >= 2 && s.numLeaves >= 32 )
    {
        // split subtree between two threads
        const auto& lr = makeNode_( s );
        const int rThreads = numThreads / 2;
        const int lThreads = numThreads - rThreads;
        tbb::task_group group;
        group.run( [&] () { makeSubtree_( lr.second, rThreads ); } );
        makeSubtree_( lr.first, lThreads );
        group.wait();
        return;
    }

    // process subtree in this thread only
    Timer t( "finishing" );
    std::stack<Subtree> stack;
    stack.push( s );

    while ( !stack.empty() )
    {
        const Subtree x = stack.top();
        stack.pop();
        if ( x.leaf() )
        {
            auto & node = nodes_[x.root];
            node.setLeafId( boxedLeaves_[x.firstLeaf].leafId );
            node.box = boxedLeaves_[x.firstLeaf].box;
            continue;
        }

        const auto & [ls, rs] = makeNode_( x );
        assert( ls.root < rs.root );
        stack.push( rs );
        stack.push( ls ); // to process it first
    }
}

template<typename T>
auto AABBTreeMaker<T>::construct( Buffer<BoxedLeaf<T>> boxedLeaves ) -> NodeVec
{
    MR_TIMER;

    boxedLeaves_ = std::move( boxedLeaves );

    const auto numLeaves = (int)boxedLeaves_.size();
    nodes_.resize( getNumNodes( numLeaves ) );

    // to equally balance the load on threads, subdivide the task on
    // a power of two subtasks, which is at least twice the hardware concurrency
    int numThreads = 1;
    int target = std::thread::hardware_concurrency();
    if ( target > 1 )
        numThreads *= 2;
    while ( target > 1 )
    {
        numThreads *= 2;
        target = ( target + 1 ) / 2;
    }
    makeSubtree_( Subtree( NodeId{ 0 }, 0, numLeaves ), numThreads );

    return std::move( nodes_ );
}

template<typename T>
AABBTreeNodeVec<T> makeAABBTreeNodeVec( Buffer<BoxedLeaf<T>> boxedLeaves )
{
    return AABBTreeMaker<T>().construct( std::move( boxedLeaves ) );
}

} //namespace MR
