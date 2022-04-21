#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRComputeBoundingBox.h"

namespace MR
{

Box3f PointCloud::getBoundingBox() const
{ 
    return getAABBTree().getBoundingBox();
}

Box3f PointCloud::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return MR::computeBoundingBox( points, validPoints, toWorld );
}

void PointCloud::addPartByMask( const PointCloud& from, const VertBitSet& fromVerts )
{
    const auto& fromPoints = from.points;
    const auto& fromNormals = from.normals;

    const bool canUseNormals = normals.size() == 0 || ( points.size() == normals.size() && fromPoints.size() == fromNormals.size() );
    assert( canUseNormals );
    if ( !canUseNormals )
        return;

    const bool useNormals = fromPoints.size() == fromNormals.size();

    VertBitSet fromValidVerts = fromVerts & from.validPoints;
    VertId idIt = VertId( points.size() );
    const auto newSize = points.size() + fromValidVerts.count();
    points.resize( newSize );
    validPoints.resize( newSize, true );
    if ( useNormals )
        normals.resize( newSize );
    for ( auto v : fromValidVerts )
    {
        points[idIt] = fromPoints[v];
        if ( useNormals )
            normals[idIt] = fromNormals[v];
        idIt++;
    }

    invalidateCaches();
}

const AABBTreePoints& PointCloud::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePoints( *this ); } );
}

size_t PointCloud::heapBytes() const
{
    return points.heapBytes()
        + normals.heapBytes()
        + validPoints.heapBytes()
        + AABBTreeOwner_.heapBytes();
}

} //namespace MR
