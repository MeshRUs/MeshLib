#include "MRGridSampling.h"
#include "MRBox.h"
#include "MRMesh.h"
#include "MRMeshPart.h"
#include "MRPointCloud.h"
#include "MRRegionBoundary.h"
#include "MRVolumeIndexer.h"
#include "MRTimer.h"
#include "MRMakeSphereMesh.h"
#include "MRGTest.h"
#include "MRComputeBoundingBox.h"

namespace MR
{

struct GridElement
{
    ObjId oid;
    VertId vid;
    float centerDistSq = FLT_MAX;
};

class Grid : public VolumeIndexer
{
public:
    Grid( const Box3f & box, const Vector3i & dims );
    // finds voxel containing given point
    Vector3i pointPos( const Vector3f & p ) const;
    // finds center of given voxel
    Vector3f voxelCenter( const Vector3i & pos ) const;
    // if given point is closer to the center of its voxel, then it is remembered
    void addVertex( const Vector3f& p, VertId vid, ObjId oid = {} );
    // returns all sampled points after addition
    VertBitSet getSamples() const;
    // returns all sampled points after addition
    std::vector<ObjVertId> setSamplesPerModel() const;

private:
    Box3f box_; 
    Vector3f voxelSize_;
    Vector3f recipVoxelSize_;
    std::vector<GridElement> voxels_;
};

Grid::Grid( const Box3f & box, const Vector3i & dims )
    : VolumeIndexer( dims )
    , box_( box )
{
    voxels_.resize( size_ );
    const auto boxSz = box.max - box.min;
    voxelSize_.x = boxSz.x / dims.x;
    voxelSize_.y = boxSz.y / dims.y;
    voxelSize_.z = boxSz.z / dims.z;
    recipVoxelSize_.x = 1 / voxelSize_.x;
    recipVoxelSize_.y = 1 / voxelSize_.y;
    recipVoxelSize_.z = 1 / voxelSize_.z;
}

inline Vector3i Grid::pointPos( const Vector3f & p ) const
{
    return
    {
        std::clamp( (int)( ( p.x - box_.min.x ) * recipVoxelSize_.x ), 0, dims_.x - 1 ),
        std::clamp( (int)( ( p.y - box_.min.y ) * recipVoxelSize_.y ), 0, dims_.y - 1 ),
        std::clamp( (int)( ( p.z - box_.min.z ) * recipVoxelSize_.z ), 0, dims_.z - 1 )
    };
}

inline Vector3f Grid::voxelCenter( const Vector3i & pos ) const
{
    return
    {
        box_.min.x + ( pos.x + 0.5f ) * voxelSize_.x,
        box_.min.y + ( pos.y + 0.5f ) * voxelSize_.y,
        box_.min.z + ( pos.z + 0.5f ) * voxelSize_.z
    };
}

void Grid::addVertex( const Vector3f& p, VertId vid, ObjId oid )
{
    const auto pos = pointPos( p );
    auto & ge = voxels_[ toVoxelId( pos ) ];
    const auto distSq = ( p - voxelCenter( pos ) ).lengthSq();
    if ( distSq < ge.centerDistSq )
    {
        ge.centerDistSq = distSq;
        ge.vid = vid;
        ge.oid = oid;
    }
}

VertBitSet Grid::getSamples() const
{
    VertId maxId;
    for ( const auto & ge : voxels_ )
        maxId = std::max( maxId, ge.vid );

    VertBitSet res( (size_t)maxId + 1 );
    for ( const auto & ge : voxels_ )
        if ( ge.vid )
            res.set( ge.vid );

    return res;
}

std::vector<MR::ObjVertId> Grid::setSamplesPerModel() const
{
    size_t counter = 0;
    for ( const auto& ge : voxels_ )
        if ( ge.vid )
            ++counter;

    std::vector<MR::ObjVertId> res( counter );
    counter = 0;
    for ( const auto& ge : voxels_ )
    {
        if ( !ge.vid )
            continue;
        res[counter] = { ge.oid,ge.vid };
        ++counter;
    }
    return res;
}

std::optional<VertBitSet> verticesGridSampling( const MeshPart & mp, float voxelSize, const ProgressCallback & cb )
{
    MR_TIMER
    if (voxelSize <= 0.f)
    {
        if ( mp.region )
            return getIncidentVerts( mp.mesh.topology, *mp.region );

        return mp.mesh.topology.getValidVerts();
    }

    const auto bbox = mp.mesh.computeBoundingBox( mp.region );
    const auto bboxSz = bbox.max - bbox.min;
    constexpr float maxVoxelsInOneDim = 1 << 10;
    const Vector3i dims
    {
        (int) std::min( std::ceil( bboxSz.x / voxelSize ), maxVoxelsInOneDim ),
        (int) std::min( std::ceil( bboxSz.y / voxelSize ), maxVoxelsInOneDim ),
        (int) std::min( std::ceil( bboxSz.z / voxelSize ), maxVoxelsInOneDim )
    };

    Grid grid( bbox, dims );
    if ( cb && !cb( 0.1f ) )
        return {};

    VertBitSet store;
    const auto& regionVerts = getIncidentVerts( mp.mesh.topology, mp.region, store );
    int counter = 0;
    int size = int( regionVerts.count() );
    for ( auto v : regionVerts )
    {
        grid.addVertex( mp.mesh.points[v], v );
        if ( !reportProgress( cb, [&]{ return 0.1f + 0.8f * float( counter ) / float( size ); }, counter++, 1024 ) )
            return {};
    }

    const auto res =  grid.getSamples();
    if ( cb && !cb( 1.0f ) )
        return {};

    return res;
}

std::optional<VertBitSet> pointGridSampling( const PointCloud & cloud, float voxelSize, const ProgressCallback & cb )
{
    if (voxelSize <= 0.f)
        return cloud.validPoints;
    MR_TIMER

    const auto bbox = cloud.getBoundingBox();
    const auto bboxSz = bbox.max - bbox.min;
    constexpr float maxVoxelsInOneDim = 1 << 10;
    const Vector3i dims
    {
        ( int )std::clamp( std::ceil( bboxSz.x / voxelSize ),1.0f, maxVoxelsInOneDim ),
        ( int )std::clamp( std::ceil( bboxSz.y / voxelSize ),1.0f, maxVoxelsInOneDim ),
        ( int )std::clamp( std::ceil( bboxSz.z / voxelSize ),1.0f, maxVoxelsInOneDim )
    };

    Grid grid( bbox, dims );
    if ( cb && !cb( 0.1f ) )
        return {};

    int counter = 0;
    int size = int( cloud.validPoints.count() );
    for ( auto v : cloud.validPoints )
    {
        grid.addVertex( cloud.points[v], v );
        if ( !reportProgress( cb, [&]{ return 0.1f + 0.8f * float( counter ) / float( size ); }, counter++, 1024 ) )
            return {};
    }

    const auto res = grid.getSamples();
    if ( cb && !cb( 1.0f ) )
        return {};
    
    return res;
}

std::optional<std::vector<ObjVertId>> multiModelGridSampling( const Vector<ModelPointsData, ObjId>& models, float voxelSize, const ProgressCallback& cb )
{
    if ( voxelSize <= 0.f )
        return {};
    MR_TIMER;

    Box3f bbox;
    for ( const auto& model : models )
    {
        if ( !model.points || !model.validPoints )
            continue;
        bbox.include( computeBoundingBox( *model.points, *model.validPoints, model.xf ) );
    }
    const auto bboxSz = bbox.max - bbox.min;
    constexpr float maxVoxelsInOneDim = 1 << 10;
    const Vector3i dims
    {
        ( int )std::clamp( std::ceil( bboxSz.x / voxelSize ),1.0f, maxVoxelsInOneDim ),
        ( int )std::clamp( std::ceil( bboxSz.y / voxelSize ),1.0f, maxVoxelsInOneDim ),
        ( int )std::clamp( std::ceil( bboxSz.z / voxelSize ),1.0f, maxVoxelsInOneDim )
    };

    Grid grid( bbox, dims );
    if ( cb && !cb( 0.1f ) )
        return {};

    ObjId oId;
    auto sb = subprogress( cb, 0.1f, 0.8f );
    for ( const auto& model : models )
    {
        ++oId;
        if ( !model.points || !model.validPoints )
            continue;
        for ( auto v : *model.validPoints )
        {
            auto useOId = model.fakeObjId ? model.fakeObjId : oId;
            if ( model.xf )
                grid.addVertex( ( *model.xf )( ( *model.points )[v] ), v, useOId );
            else
                grid.addVertex( ( *model.points )[v], v, useOId );
        }
       

        if ( !reportProgress( sb, float( oId + 1 ) / models.size() ) )
            return {};
    }

    std::vector<ObjVertId> res = grid.setSamplesPerModel();

    if ( cb && !cb( 1.0f ) )
        return {};

    return res;
}

TEST( MRMesh, GridSampling )
{
    auto sphereMesh = makeUVSphere();
    auto numVerts = sphereMesh.topology.numValidVerts();
    auto samples = verticesGridSampling( sphereMesh, 0.5f );
    auto sampleCount = samples->count();
    EXPECT_LE( sampleCount, numVerts );
}

} //namespace MR
