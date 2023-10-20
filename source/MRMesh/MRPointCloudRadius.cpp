#include "MRPointCloudRadius.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRBox.h"
#include "MRBitSetParallelFor.h"
#include "MRPointsInBall.h"

namespace MR
{
// TODO: this function can be more precise if remake it with reduction of all aabb nodes
float findAvgPointsRadius( const PointCloud& pointCloud, int avgPoints )
{
    const auto& pointsAABBTree = pointCloud.getAABBTree();
    const auto& nodes = pointsAABBTree.nodes();
    int N = AABBTreePoints::MaxNumPointsInLeaf;
    float radiusNSq = 0.0f;
    int counter = 0;
    for ( const auto& node : nodes )
    {
        if ( !node.leaf() )
            continue;
        auto [first, last] = node.getLeafPointRange();
        if ( last - first != N )
            continue;
        radiusNSq += sqr( node.box.diagonal() * 0.5f );
        ++counter;
    }
    if ( counter == 0 )
    {
        N = int( pointsAABBTree.orderedPoints().size() );
        counter = 1;
        radiusNSq = sqr( pointsAABBTree.getBoundingBox().diagonal() * 0.5f );
        assert( N > 0 );
    }
    radiusNSq = radiusNSq / float( counter );
    return sqrt( radiusNSq / N * avgPoints * 0.5f );
}

bool dilateRegion( const PointCloud& pointCloud, VertBitSet& region, float dilation, ProgressCallback cb, const AffineXf3f* xf )
{
    return BitSetParallelForAll( region, [&] ( VertId testVertex )
    {
        if ( region.test( testVertex ) )
            return;

        findPointsInBall( pointCloud, pointCloud.points[testVertex], dilation, [&] ( VertId v, const Vector3f& )
        {
            if ( region.test( v ) )
                region.set( testVertex, true );
        }, xf );
    }, cb );
}

bool erodeRegion( const PointCloud& pointCloud, VertBitSet& region, float erosion, ProgressCallback cb, const AffineXf3f* xf )
{
    return BitSetParallelForAll( region, [&] ( VertId testVertex )
    {
        if ( !region.test( testVertex ) )
            return;

        findPointsInBall( pointCloud, pointCloud.points[testVertex], erosion, [&] ( VertId v, const Vector3f& )
        {
            if ( !region.test( v ) )
                region.set( testVertex, false );
        }, xf );
    }, cb );
}
}
