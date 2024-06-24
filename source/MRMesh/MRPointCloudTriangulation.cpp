#include "MRPointCloudTriangulation.h"
#include "MRPointCloud.h"
#include "MRVector.h"
#include "MRId.h"
#include "MRPointCloudRadius.h"
#include "MRPointCloudMakeNormals.h"
#include "MRBitSetParallelFor.h"
#include "MRMeshDelone.h"
#include "MRMeshBuilder.h"
#include "MRMeshFillHole.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRVector3.h"
#include "MRBox.h"
#include "MRTimer.h"
#include "MRPointCloudTriangulationHelpers.h"
#include "MRRegionBoundary.h"
#include "MRParallelFor.h"
#include "MRLocalTriangulations.h"
#include "MRMeshFixer.h"
#include "MREdgePaths.h"
#include <parallel_hashmap/phmap.h>

namespace MR
{

class PointCloudTriangulator
{
public:
    PointCloudTriangulator( const PointCloud& pointCloud, const TriangulationParameters& params );

    std::optional<Mesh> triangulate( ProgressCallback progressCb );

private:
    /// constructs mesh from given triangles
    std::optional<Mesh> makeMesh_( Triangulation && t3, Triangulation && t2, ProgressCallback progressCb );

    const PointCloud& pointCloud_;
    TriangulationParameters params_;
};

PointCloudTriangulator::PointCloudTriangulator( const PointCloud& pointCloud, const TriangulationParameters& params ) :
    pointCloud_{pointCloud},
    params_{params}
{
}

std::optional<Mesh> PointCloudTriangulator::triangulate( ProgressCallback progressCb )
{
    MR_TIMER
    assert( ( params_.numNeighbours <= 0 && params_.radius > 0 )
         || ( params_.numNeighbours > 0 && params_.radius <= 0 ) );

    auto optLocalTriangulations = TriangulationHelpers::buildUnitedLocalTriangulations( pointCloud_,
        {
            .radius = params_.radius,
            .numNeis = params_.numNeighbours,
            .critAngle = params_.critAngle,
            .boundaryAngle = params_.boundaryAngle,
            .trustedNormals = pointCloud_.hasNormals() ? &pointCloud_.normals : nullptr,
            .automaticRadiusIncrease = params_.automaticRadiusIncrease,
            .searchNeighbors = params_.searchNeighbors
        }, subprogress( progressCb, 0.0f, pointCloud_.hasNormals() ? 0.4f : 0.3f ) );
    if ( !optLocalTriangulations )
        return {};
    auto & localTriangulations = *optLocalTriangulations;

    Triangulation t3, t2;
    if ( pointCloud_.hasNormals() )
        findRepeatedOrientedTriangles( localTriangulations, &t3, &t2 );
    else
        autoOrientLocalTriangulations( pointCloud_, localTriangulations, pointCloud_.validPoints, subprogress( progressCb, 0.3f, 0.5f ), &t3, &t2 );

    return makeMesh_( std::move( t3 ), std::move( t2 ), subprogress( progressCb, 0.5f, 1.0f ) );
}

std::optional<Mesh> PointCloudTriangulator::makeMesh_( Triangulation && t3, Triangulation && t2, ProgressCallback progressCb )
{
    MR_TIMER

    Mesh mesh;
    mesh.points = pointCloud_.points;

    auto compare = [] ( const auto& l, const auto& r )->bool
    {
        if ( l[0] < r[0] )
            return true;
        if ( l[0] > r[0] )
            return false;
        if ( l[1] < r[1] )
            return true;
        if ( l[1] > r[1] )
            return false;
        return l[2] < r[2];
    };
    tbb::parallel_sort( t3.vec_.begin(), t3.vec_.end(), compare );
    tbb::parallel_sort( t2.vec_.begin(), t2.vec_.end(), compare );
    auto t3Size = t3.size();
    t3.vec_.insert( t3.vec_.end(), std::make_move_iterator( t2.vec_.begin() ), std::make_move_iterator( t2.vec_.end() ) );
    FaceBitSet region3( t3Size );
    region3.flip();
    FaceBitSet region2( t3.size() );
    region2.flip();
    region2 -= region3;

    // create topology
    MeshBuilder::addTriangles( mesh.topology, t3, { .region = &region3, .allowNonManifoldEdge = false } );
    if ( !reportProgress( progressCb, 0.1f ) )
        return {};
    region2 |= region3;
    MeshBuilder::addTriangles( mesh.topology, t3, { .region = &region2, .allowNonManifoldEdge = false } );
    if ( !reportProgress( progressCb, 0.2f ) )
        return {};

    // remove bad triangles
    mesh.deleteFaces( findHoleComplicatingFaces( mesh ) );

    // fill small holes
    const auto maxHolePerimeterToFill = params_.critHoleLength >= 0.0f ?
        params_.critHoleLength :
        pointCloud_.getBoundingBox().diagonal() * 0.1f;
    auto boundaries = findRightBoundary( mesh.topology );
    // setup parameters to prevent any appearance of multiple edges during hole filling
    FillHoleParams fillHoleParams;
    fillHoleParams.multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong;
    bool stoppedFlag = false;
    fillHoleParams.stopBeforeBadTriangulation = &stoppedFlag;
    for ( int i = 0; i < boundaries.size(); ++i )
    {
        const auto& boundary = boundaries[i];

        if ( (float)calcPathLength( boundary, mesh ) <= maxHolePerimeterToFill )
            fillHole( mesh, boundary.front(), fillHoleParams );

        if ( !reportProgress( progressCb, [&]{ return 0.3f + 0.7f * float( i + 1 ) / float( boundaries.size() ); } ) ) // 30% - 100%
            return {};
    }

    return mesh;
}

std::optional<Mesh> triangulatePointCloud( const PointCloud& pointCloud, const TriangulationParameters& params /*= {} */,
    ProgressCallback progressCb )
{
    MR_TIMER
    PointCloudTriangulator triangulator( pointCloud, params );
    return triangulator.triangulate( progressCb );
}

} //namespace MR
