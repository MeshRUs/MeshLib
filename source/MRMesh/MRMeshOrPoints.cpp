#include "MRMeshOrPoints.h"
#include "MRMesh.h"
#include "MRPointCloud.h"
#include "MRBox.h"
#include "MRGridSampling.h"
#include "MRMeshProject.h"
#include "MRPointsProject.h"
#include "MRObjectMesh.h"
#include "MRObjectPoints.h"
#include "MRBestFit.h"

namespace MR
{

Box3f MeshOrPoints::computeBoundingBox( const AffineXf3f * toWorld ) const
{
    return std::visit( overloaded{
        [toWorld]( const MeshPart & mp ) { return mp.mesh.computeBoundingBox( mp.region, toWorld ); },
        [toWorld]( const PointCloud * pc ) { return pc->computeBoundingBox( toWorld ); }
    }, var_ );
}

void MeshOrPoints::accumulate( PointAccumulator& accum, const AffineXf3f* xf ) const
{
    return std::visit( overloaded{
        [&accum, xf]( const MeshPart & mp ) { accumulateFaceCenters( accum, mp, xf ); },
        [&accum, xf]( const PointCloud * pc ) { accumulatePoints( accum, *pc, xf ); }
    }, var_ );
}

std::optional<VertBitSet> MeshOrPoints::pointsGridSampling( float voxelSize, size_t maxVoxels, const ProgressCallback & cb ) const
{
    assert( voxelSize > 0 );
    assert( maxVoxels > 0 );
    auto bboxDiag = computeBoundingBox().size() / voxelSize;
    auto nSamples = bboxDiag[0] * bboxDiag[1] * bboxDiag[2];
    if ( nSamples > maxVoxels )
        voxelSize *= std::cbrt( float(nSamples) / float(maxVoxels) );
    return std::visit( overloaded{
        [voxelSize, cb]( const MeshPart & mp ) { return verticesGridSampling( mp, voxelSize, cb ); },
        [voxelSize, cb]( const PointCloud * pc ) { return pointGridSampling( *pc, voxelSize, cb ); }
    }, var_ );
}

const VertCoords & MeshOrPoints::points() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> const VertCoords & { return mp.mesh.points; },
        []( const PointCloud * pc ) -> const VertCoords & { return pc->points; }
    }, var_ );
}

std::function<Vector3f(VertId)> MeshOrPoints::normals() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<Vector3f(VertId)>
        {
            return [&mesh = mp.mesh]( VertId v ) { return mesh.pseudonormal( v ); };
        },
        []( const PointCloud * pc ) -> std::function<Vector3f(VertId)>
        { 
            return !pc->hasNormals() ? std::function<Vector3f(VertId)>{} : [pc]( VertId v ) { return pc->normals[v]; };
        }
    }, var_ );
}

std::function<float(VertId)> MeshOrPoints::weights() const
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> std::function<float(VertId)>
        {
            return [&mesh = mp.mesh]( VertId v ) { return mesh.dblArea( v ); };
        },
        []( const PointCloud * ) { return std::function<float(VertId)>{}; }
    }, var_ );
}

auto MeshOrPoints::projector() const -> std::function<ProjectionResult( const Vector3f & )>
{
    return [lp = limitedProjector()]( const Vector3f & p )
    {
        ProjectionResult res;
        lp( p, res );
        return res;
    };
}

auto MeshOrPoints::limitedProjector() const -> LimitedProjectorFunc
{
    return std::visit( overloaded{
        []( const MeshPart & mp ) -> LimitedProjectorFunc
        {
            return [&mp]( const Vector3f & p, ProjectionResult & res )
            {
                MeshProjectionResult mpr = findProjection( p, mp, res.distSq );
                if ( mpr.distSq < res.distSq )
                    res = ProjectionResult
                    {
                        .point = mpr.proj.point,
                        .normal = mp.mesh.pseudonormal( mpr.mtp ),
                        .isBd = mpr.mtp.isBd( mp.mesh.topology ),
                        .distSq = mpr.distSq,
                        .closestVert = mp.mesh.getClosestVertex( mpr.proj )
                    };
            };
        },
        []( const PointCloud * pc ) -> LimitedProjectorFunc
        {
            return [pc]( const Vector3f & p, ProjectionResult & res )
            {
                PointsProjectionResult ppr = findProjectionOnPoints( p, *pc, res.distSq );
                if ( ppr.distSq < res.distSq )
                    res = ProjectionResult
                    {
                        .point = pc->points[ppr.vId],
                        .normal = ppr.vId < pc->normals.size() ? pc->normals[ppr.vId] : std::optional<Vector3f>{},
                        .distSq = ppr.distSq,
                        .closestVert = ppr.vId
                    };
            };
        }
    }, var_ );
}

std::optional<MeshOrPoints> getMeshOrPoints( const VisualObject * obj )
{
    if ( auto objMesh = dynamic_cast<const ObjectMesh*>( obj ) )
        return MeshOrPoints( objMesh->meshPart() );
    if ( auto objPnts = dynamic_cast<const ObjectPoints*>( obj ) )
        return MeshOrPoints( *objPnts->pointCloud() );
    return {};
}

} // namespace MR
