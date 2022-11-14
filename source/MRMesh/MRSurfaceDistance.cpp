#include "MRSurfaceDistance.h"
#include "MRSurfaceDistanceBuilder.h"
#include "MRMesh.h"
#include "MRTimer.h"

namespace MR
{

Vector<float,VertId> computeSurfaceDistances( const Mesh & mesh, const VertBitSet & startVertices, float maxDist, 
                                              const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStartRegion( startVertices, 0 );
    while ( b.doneDistance() < maxDist )
    {
        b.growOne();
    }
    return b.takeDistanceMap();
}

Vector<float,VertId> computeSurfaceDistances( const Mesh & mesh, const VertBitSet & startVertices, const VertBitSet& targetVertices,
    float maxDist, const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStartRegion( startVertices, 0 );

    auto toReachVerts = targetVertices - startVertices;
    auto toReachCount = toReachVerts.count();

    while ( toReachCount > 0 && b.doneDistance() < maxDist )
    {
        auto v = b.growOne();
        if ( toReachVerts.test( v ) )
            --toReachCount;
    }
    return b.takeDistanceMap();
}

Vector<float, VertId> computeSurfaceDistances( const Mesh& mesh, const HashMap<VertId, float>& startVertices, float maxDist,
                                               const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStartVertices( startVertices );
    while ( b.doneDistance() < maxDist )
    {
        b.growOne();
    }
    return b.takeDistanceMap();
}

Vector<float,VertId> computeSurfaceDistances( const Mesh & mesh, const MeshTriPoint & start, const MeshTriPoint & end, 
                                              const VertBitSet* region, bool * endReached )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStart( start );

    VertId stopVerts[3];
    int numStopVerts = 0;
    mesh.topology.forEachVertex( end, [&]( VertId v )
    {
        stopVerts[ numStopVerts++ ] = v;
    } );

    if ( endReached )
        *endReached = true;
    while ( numStopVerts > 0 )
    {
        auto v = b.growOne();
        if ( !v )
        {
            if ( endReached )
                *endReached = false;
            break;
        }

        auto it = std::find( stopVerts, stopVerts + numStopVerts, v );
        if ( it != stopVerts + numStopVerts )
        {
            for ( ; it + 1 < stopVerts + numStopVerts; ++it )
                *it = *(it + 1);
            --numStopVerts;
        }
    }
    return b.takeDistanceMap();
}

Vector<float,VertId> computeSurfaceDistances( const Mesh& mesh, const MeshTriPoint & start, float maxDist,
                                              const VertBitSet* region )
{
    MR_TIMER;

    SurfaceDistanceBuilder b( mesh, region );
    b.addStart( start );
    while ( b.doneDistance() < maxDist )
    {
        b.growOne();
    }
    return b.takeDistanceMap();
}

} //namespace MR
