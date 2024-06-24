#include "MRObjectVoxels.h"
#ifndef MRMESH_NO_OPENVDB
#include "MRObjectFactory.h"
#include "MRMesh.h"
#include "MRVDBConversions.h"
#include "MRVDBFloatGrid.h"
#include "MRFloatGrid.h"
#include "MRSimpleVolume.h"
#include "MRVoxelsSave.h"
#include "MRVoxelsLoad.h"
#include "MRSerializer.h"
#include "MRMeshNormals.h"
#include "MRTimer.h"
#include "MRSceneColors.h"
#include "MRStringConvert.h"
#include "MROpenVDBHelper.h"
#include "MRVoxelsConversions.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRJson.h"
#include "MRPch/MRAsyncLaunchType.h"
#include "MRPch/MRFmt.h"
#include <filesystem>
#include <thread>

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectVoxels )

constexpr size_t cVoxelsHistogramBinsNumber = 256;

void ObjectVoxels::construct( const SimpleVolume& volume, ProgressCallback cb )
{
    mesh_.reset();
    vdbVolume_.data = simpleVolumeToDenseGrid( volume, cb );
    vdbVolume_.dims = volume.dims;
    vdbVolume_.voxelSize = volume.voxelSize;
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    activeBox_ = Box3i( Vector3i(), vdbVolume_.dims );
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    volumeRenderActiveVoxels_.clear();

    updateHistogram_( volume.min, volume.max );
    if ( volumeRendering_ )
        dirty_ |= ( DIRTY_PRIMITIVES | DIRTY_TEXTURE | DIRTY_SELECTION );
}

void ObjectVoxels::construct( const FloatGrid& grid, const Vector3f& voxelSize, ProgressCallback cb )
{
    if ( !grid )
        return;
    vdbVolume_.data = grid;

    auto vdbDims = vdbVolume_.data->evalActiveVoxelDim();
    vdbVolume_.dims = {vdbDims.x(),vdbDims.y(),vdbDims.z()};
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    activeBox_ = Box3i( Vector3i(), vdbVolume_.dims );
    vdbVolume_.voxelSize = voxelSize;
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    volumeRenderActiveVoxels_.clear();

    updateHistogramAndSurface( cb );
    if ( volumeRendering_ )
        dirty_ |= ( DIRTY_PRIMITIVES | DIRTY_TEXTURE | DIRTY_SELECTION );
}

void ObjectVoxels::construct( const VdbVolume& volume, ProgressCallback cb )
{
    construct( volume.data, volume.voxelSize, cb );
}

void ObjectVoxels::updateHistogramAndSurface( ProgressCallback cb )
{
    if ( !vdbVolume_.data )
        return;

    float min{0.0f}, max{0.0f};

    evalGridMinMax( vdbVolume_.data, min, max );

    const float progressTo = ( mesh_ && cb ) ? 0.5f : 1.f;
    updateHistogram_( min, max, subprogress( cb, 0.f, progressTo ) );
    vdbVolume_.min = min;
    vdbVolume_.max = max;
    if ( mesh_ )
    {
        mesh_.reset();

        const float progressFrom = cb ? 0.5f : 0.f;
        setIsoValue( isoValue_, subprogress( cb, progressFrom, 1.f ) );
    }
}

Expected<bool, std::string> ObjectVoxels::setIsoValue( float iso, ProgressCallback cb, bool updateSurface )
{
    if ( !vdbVolume_.data )
        return false; // no volume presented in this
    if ( mesh_ && iso == isoValue_ )
        return false; // current iso surface represents required iso value

    isoValue_ = iso;
    if ( updateSurface )
    {
        auto recRes = recalculateIsoSurface( isoValue_, cb );
        if ( !recRes.has_value() )
            return unexpected( recRes.error() );
        updateIsoSurface( *recRes );
    }
    if ( volumeRendering_ )
        dirty_ |= DIRTY_TEXTURE;
    return updateSurface;
}

std::shared_ptr<Mesh> ObjectVoxels::updateIsoSurface( std::shared_ptr<Mesh> mesh )
{
    if ( mesh != mesh_ )
    {
        mesh_.swap( mesh );
        setDirtyFlags( DIRTY_ALL );
        isoSurfaceChangedSignal();
    }
    return mesh;

}

VdbVolume ObjectVoxels::updateVdbVolume( VdbVolume vdbVolume )
{
    auto oldVdbVolume = std::move( vdbVolume_ );
    vdbVolume_ = std::move( vdbVolume );
    indexer_ = VolumeIndexer( vdbVolume_.dims );
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x, 1 / vdbVolume_.voxelSize.y, 1 / vdbVolume_.voxelSize.z };
    volumeRenderActiveVoxels_.clear();
    setDirtyFlags( DIRTY_ALL );
    if ( volumeRendering_ )
        dirty_ |= DIRTY_SELECTION; // This flag is not set in ObjectMeshHolder::setDirtyFlags
    return oldVdbVolume;
}

Histogram ObjectVoxels::updateHistogram( Histogram histogram )
{
    auto oldHistogram = std::move( histogram_ );
    histogram_ = std::move( histogram );
    return oldHistogram;
}

Box3i ObjectVoxels::updateActiveBounds( const Box3i& box )
{
    Box3i oldBox = activeBox_;
    activeBox_ = box;
    return oldBox;
}

Expected<std::shared_ptr<Mesh>, std::string> ObjectVoxels::recalculateIsoSurface( float iso, ProgressCallback cb /*= {} */ ) const
{
    MR_TIMER
    if ( !vdbVolume_.data )
        return unexpected("No VdbVolume available");

    float startProgress = 0;   // where the current iteration has started
    float reachedProgress = 0; // maximum progress reached so far
    ProgressCallback myCb;
    if ( cb )
        myCb = [&startProgress, &reachedProgress, cb]( float p )
        {
            reachedProgress = startProgress + ( 1 - startProgress ) * p;
            return cb( reachedProgress );
        };

    auto vdbVolume = vdbVolume_;
    for (;;)
    {
        // continue progress bar from the value where it stopped on the previous iteration
        startProgress = reachedProgress;
        Expected<Mesh, std::string> meshRes;
        if ( dualMarchingCubes_ )
        {
            meshRes = gridToMesh( vdbVolume.data, GridToMeshSettings{
                .voxelSize = vdbVolume.voxelSize,
                .isoValue = iso,
                .maxVertices = maxSurfaceVertices_,
                .cb = myCb
            } );
        }
        else
        {
            MarchingCubesParams vparams;
            vparams.iso = iso;
            vparams.maxVertices = maxSurfaceVertices_;
            vparams.cb = myCb;
            vparams.positioner = positioner_;
            if ( vdbVolume.data->getGridClass() == openvdb::GridClass::GRID_LEVEL_SET )
                vparams.lessInside = true;
            meshRes = marchingCubes( vdbVolume, vparams );
        }
        if ( meshRes.has_value() )
            return std::make_shared<Mesh>( std::move( meshRes.value() ) );
        if ( !meshRes.has_value() && meshRes.error() == stringOperationCanceled() )
            return unexpectedOperationCanceled();
        vdbVolume.data = resampled( vdbVolume.data, 2.0f );
        vdbVolume.voxelSize *= 2.0f;
        auto vdbDims = vdbVolume.data->evalActiveVoxelDim();
        vdbVolume.dims = {vdbDims.x(),vdbDims.y(),vdbDims.z()};
    }
}

void ObjectVoxels::setDualMarchingCubes( bool on, bool updateSurface, ProgressCallback cb )
{
    MR_TIMER
    dualMarchingCubes_ = on;
    if ( updateSurface )
    {
        auto recRes = recalculateIsoSurface( isoValue_, cb );
        if ( recRes.has_value() )
            updateIsoSurface( *recRes );
    }
}

void ObjectVoxels::setActiveBounds( const Box3i& activeBox, ProgressCallback cb, bool updateSurface )
{
    if ( !vdbVolume_.data )
        return;
    if ( !activeBox.valid() )
        return;

    activeBox_ = activeBox;
    auto accessor = vdbVolume_.data->getAccessor();

    size_t counter = 0;
    float volume = float( vdbVolume_.dims.x ) * vdbVolume_.dims.y * vdbVolume_.dims.z;
    float cbModifier = 1.0f;
    if ( updateSurface && volumeRendering_ )
        cbModifier = 1.0f / 3.0f;
    else if ( updateSurface || volumeRendering_ )
        cbModifier = 1.0f / 2.0f;
    float lastProgress = 0.0f;

    bool insideX = false;
    bool insideY = false;
    bool insideZ = false;
    for ( int z = 0; z < vdbVolume_.dims.z; ++z )
    for ( int y = 0; y < vdbVolume_.dims.y; ++y )
    for ( int x = 0; x < vdbVolume_.dims.x; ++x )
    {
        insideX = ( x >= activeBox_.min.x && x < activeBox_.max.x );
        insideY = ( y >= activeBox_.min.y && y < activeBox_.max.y );
        insideZ = ( z >= activeBox_.min.z && z < activeBox_.max.z );
        accessor.setActiveState( {x,y,z}, insideX && insideY && insideZ );
        reportProgress( cb, [&]{ return cbModifier * float( counter ) / volume; }, ++counter, 256 );
    }

    volumeRenderActiveVoxels_.clear();
    dirty_ |= DIRTY_SELECTION;

    lastProgress = cbModifier;
    if ( updateSurface )
    {
        ProgressCallback isoProgressCallback = subprogress( cb, lastProgress, 2.0f * cbModifier );
        lastProgress = 2.0f * cbModifier;
        auto recRes = recalculateIsoSurface( isoValue_, isoProgressCallback );
        std::shared_ptr<Mesh> recMesh;
        if ( recRes.has_value() )
            recMesh = *recRes;
        updateIsoSurface( recMesh );
    }
    if ( volumeRendering_ )
    {
        prepareDataForVolumeRendering( subprogress( cb, lastProgress, 1.0f ) );
        setDirtyFlags( DIRTY_PRIMITIVES );
    }
}

void ObjectVoxels::setVolumeRenderActiveVoxels( const VoxelBitSet& activeVoxels )
{
    auto box = activeBox_.size();
    const bool valid = activeVoxels.empty() || activeVoxels.size() == box.x * box.y * box.z;
    assert( valid );
    if ( !valid )
        return;
    volumeRenderActiveVoxels_ = activeVoxels;
    dirty_ |= DIRTY_SELECTION;
}

VoxelId ObjectVoxels::getVoxelIdByCoordinate( const Vector3i& coord ) const
{
    return indexer_.toVoxelId( coord );
}

VoxelId ObjectVoxels::getVoxelIdByPoint( const Vector3f& point ) const
{
    return getVoxelIdByCoordinate( Vector3i( mult( point, reverseVoxelSize_ ) ) );
}

Vector3i ObjectVoxels::getCoordinateByVoxelId( VoxelId id ) const
{
    return indexer_.toPos( id );
}

bool ObjectVoxels::prepareDataForVolumeRendering( ProgressCallback cb /*= {} */ ) const
{
    if ( !vdbVolume_.data )
        return false;
    auto res = vdbVolumeToSimpleVolumeNorm( vdbVolume_, getActiveBounds(), cb );
    if ( !res || res->data.empty() )
    {
        volumeRenderingData_.reset();
        return false;
    }
    volumeRenderingData_ = std::make_unique<SimpleVolume>( std::move( *res ) );
    return true;
}

void ObjectVoxels::enableVolumeRendering( bool on )
{
    if ( volumeRendering_ == on )
        return;
    volumeRendering_ = on;
    if ( volumeRendering_ )
    {
        if ( !volumeRenderingData_ )
            prepareDataForVolumeRendering();
        renderObj_ = createRenderObject<ObjectVoxels>( *this );
    }
    else
    {
        renderObj_ = createRenderObject<ObjectMeshHolder>( *this );
    }
    setDirtyFlags( DIRTY_ALL );
}

void ObjectVoxels::setVolumeRenderingParams( const VolumeRenderingParams& params )
{
    if ( params == volumeRenderingParams_ )
        return;
    volumeRenderingParams_ = params;
    if ( isVolumeRenderingEnabled() )
        dirty_ |= DIRTY_TEXTURE;
}

bool ObjectVoxels::hasVisualRepresentation() const
{
    if ( isVolumeRenderingEnabled() )
        return false;
    return bool( mesh_ );
}

void ObjectVoxels::setMaxSurfaceVertices( int maxVerts )
{
    if ( maxVerts == maxSurfaceVertices_ )
        return;
    maxSurfaceVertices_ = maxVerts;
    if ( !mesh_ || mesh_->topology.numValidVerts() <= maxSurfaceVertices_ )
        return;
    mesh_.reset();
    setIsoValue( isoValue_ );
}

std::shared_ptr<Object> ObjectVoxels::clone() const
{
    auto res = std::make_shared<ObjectVoxels>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = std::make_shared<Mesh>( *mesh_ );
    if ( vdbVolume_.data )
        res->vdbVolume_.data = MakeFloatGrid( vdbVolume_.data->deepCopy() );
    return res;
}

std::shared_ptr<Object> ObjectVoxels::shallowClone() const
{
    auto res = std::make_shared<ObjectVoxels>( ProtectedStruct{}, *this );
    if ( mesh_ )
        res->mesh_ = mesh_;
    if ( vdbVolume_.data )
        res->vdbVolume_ = vdbVolume_;
    return res;
}

void ObjectVoxels::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectMeshHolder::setDirtyFlags( mask, invalidateCaches );

    if ( invalidateCaches && ( mask & DIRTY_POSITION || mask & DIRTY_FACE ) && mesh_ )
        mesh_->invalidateCaches();
}

size_t ObjectVoxels::heapBytes() const
{
    return ObjectMeshHolder::heapBytes()
        + vdbVolume_.heapBytes()
        + histogram_.heapBytes()
        + MR::heapBytes( volumeRenderingData_ );
}

void ObjectVoxels::swapBase_( Object& other )
{
    if ( auto otherVoxels = other.asType<ObjectVoxels>() )
        std::swap( *this, *otherVoxels );
    else
        assert( false );
}

void ObjectVoxels::swapSignals_( Object& other )
{
    ObjectMeshHolder::swapSignals_( other );
    if ( auto otherVoxels = other.asType<ObjectVoxels>() )
        std::swap( isoSurfaceChangedSignal, otherVoxels->isoSurfaceChangedSignal );
    else
        assert( false );
}

/// @brief class to parallel reduce histogram calculation
template<typename TreeT>
class HistogramCalcProc
{
public:
    using ValueT = typename TreeT::ValueType;
    using TreeAccessor = openvdb::tree::ValueAccessor<const TreeT>;
    using LeafIterT = typename TreeT::LeafCIter;
    using TileIterT = typename TreeT::ValueAllCIter;

    HistogramCalcProc( float min, float max ) :
        hist( min, max, cVoxelsHistogramBinsNumber )
    {}

    HistogramCalcProc( const HistogramCalcProc& other ) :
        hist( Histogram( other.hist.getMin(), other.hist.getMax(), cVoxelsHistogramBinsNumber ) )
    {}

    void action( const LeafIterT&, const TreeAccessor& treeAcc, const openvdb::math::CoordBBox& bbox )
    {
        for ( auto it = bbox.begin(); it != bbox.end(); ++it )
        {
            ValueT value = ValueT();
            if ( treeAcc.probeValue( *it, value ) )
                hist.addSample( value );
        }
    }

    void action( const TileIterT& iter, const TreeAccessor&, const openvdb::math::CoordBBox& bbox )
    {
        ValueT value = iter.getValue();
        const size_t count = size_t( bbox.volume() );
        hist.addSample( value, count );
    }

    void join( const HistogramCalcProc& other )
    {
        hist.addHistogram( other.hist );
    }

    Histogram hist;
};


void ObjectVoxels::updateHistogram_( float min, float max, ProgressCallback cb /*= {}*/ )
{
    MR_TIMER;

    RangeSize size = calculateRangeSize( *vdbVolume_.data );
    
    using HistogramCalcProcFT = HistogramCalcProc<openvdb::FloatTree>;
    HistogramCalcProcFT histCalcProc( min, max );
    using HistRangeProcessorOne = RangeProcessorSingle<openvdb::FloatTree, HistogramCalcProcFT>;
    HistRangeProcessorOne calc( vdbVolume_.data->evalActiveVoxelBoundingBox(), vdbVolume_.data->tree(), histCalcProc );
    

    if ( size.tile > 0 )
    {
        typename HistRangeProcessorOne::TileIterT tileIterMain = vdbVolume_.data->tree().cbeginValueAll();
        tileIterMain.setMaxDepth( tileIterMain.getLeafDepth() - 1 ); // skip leaf nodes
        typename HistRangeProcessorOne::TileRange tileRangeMain( tileIterMain );
        auto sb = size.leaf > 0 ? subprogress( cb, 0.0f, 0.5f ) : cb;
        calc.setProgressHolder( std::make_shared<RangeProgress>( sb, size.tile, RangeProgress::Mode::Tiles ) );
        tbb::parallel_reduce( tileRangeMain, calc );
    }

    if ( size.leaf > 0 )
    {
        typename HistRangeProcessorOne::LeafRange leafRangeMain( vdbVolume_.data->tree().cbeginLeaf() );
        auto sb = size.tile > 0 ? subprogress( cb, 0.5f, 1.0f ) : cb;
        calc.setProgressHolder( std::make_shared<RangeProgress>( sb, size.leaf, RangeProgress::Mode::Leaves ) );
        tbb::parallel_reduce( leafRangeMain, calc );
    }

    histogram_ = std::move( calc.mProc.hist );
}

void ObjectVoxels::setDefaultColors_()
{
    setFrontColor( SceneColors::get( SceneColors::SelectedObjectVoxels ), true );
    setFrontColor( SceneColors::get( SceneColors::UnselectedObjectVoxels ), false );
}

void ObjectVoxels::setDefaultSceneProperties_()
{
    setDefaultColors_();
}

ObjectVoxels::ObjectVoxels()
{
    setDefaultSceneProperties_();
}

void ObjectVoxels::applyScale( float scaleFactor )
{
    vdbVolume_.voxelSize *= scaleFactor;
    reverseVoxelSize_ = { 1 / vdbVolume_.voxelSize.x,1 / vdbVolume_.voxelSize.y,1 / vdbVolume_.voxelSize.z };

    ObjectMeshHolder::applyScale( scaleFactor );
}

void ObjectVoxels::serializeFields_( Json::Value& root ) const
{
    VisualObject::serializeFields_( root );
    serializeToJson( vdbVolume_.voxelSize, root["VoxelSize"] );

    serializeToJson( vdbVolume_.dims, root["Dimensions"] );
    serializeToJson( activeBox_.min, root["MinCorner"] );
    serializeToJson( activeBox_.max, root["MaxCorner"] );
    serializeToJson( selectedVoxels_, root["SelectionVoxels"] );

    root["IsoValue"] = isoValue_;
    root["DualMarchingCubes"] = dualMarchingCubes_;
    root["Type"].append( ObjectVoxels::TypeName() );
}

Expected<std::future<VoidOrErrStr>> ObjectVoxels::serializeModel_( const std::filesystem::path& path ) const
{
    if ( ancillary_ || !vdbVolume_.data )
        return {};

    return std::async( getAsyncLaunchType(),
        [this, filename = utf8string( path ) + ".raw"] ()
    {
        return MR::VoxelsSave::toRawAutoname( vdbVolume_, pathFromUtf8( filename ) );
    } );
}

void ObjectVoxels::deserializeFields_( const Json::Value& root )
{
    VisualObject::deserializeFields_( root );
    if ( root["VoxelSize"].isDouble() )
        vdbVolume_.voxelSize = Vector3f::diagonal( ( float )root["VoxelSize"].asDouble() );
    else
        deserializeFromJson( root["VoxelSize"], vdbVolume_.voxelSize );

    deserializeFromJson( root["Dimensions"], vdbVolume_.dims );

    deserializeFromJson( root["MinCorner"], activeBox_.min );

    deserializeFromJson( root["MaxCorner"], activeBox_.max );

    deserializeFromJson( root["SelectionVoxels"], selectedVoxels_ );

    if ( root["IsoValue"].isNumeric() )
        isoValue_ = root["IsoValue"].asFloat();

    if ( root["DualMarchingCubes"].isBool() )
        dualMarchingCubes_ = root["DualMarchingCubes"].asBool();

    if ( !activeBox_.valid() )
        activeBox_ = Box3i( Vector3i(), vdbVolume_.dims );

    if ( activeBox_.min != Vector3i() || activeBox_.max != vdbVolume_.dims )
        setActiveBounds( activeBox_ );
    else 
        setIsoValue( isoValue_ );

    if ( root["UseDefaultSceneProperties"].isBool() && root["UseDefaultSceneProperties"].asBool() )
        setDefaultSceneProperties_();
}

VoidOrErrStr ObjectVoxels::deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb )
{
    auto res = VoxelsLoad::fromRaw( pathFromUtf8( utf8string( path ) + ".raw" ), progressCb );
    if ( !res.has_value() )
        return unexpected( res.error() );
    
    construct( res.value().data, res.value().voxelSize );
    if ( !vdbVolume_.data )
        return unexpected( "No grid loaded" );

    return {};
}

std::vector<std::string> ObjectVoxels::getInfoLines() const
{
    std::vector<std::string> res = ObjectMeshHolder::getInfoLines();
    res.push_back( fmt::format( "dims: ({}, {}, {})", vdbVolume_.dims.x, vdbVolume_.dims.y, vdbVolume_.dims.z ) );
    res.push_back( fmt::format( "voxel size: ({:.3}, {:.3}, {:.3})", vdbVolume_.voxelSize.x, vdbVolume_.voxelSize.y, vdbVolume_.voxelSize.z ) );
    res.push_back( fmt::format( "volume: ({:.3}, {:.3}, {:.3})", 
        vdbVolume_.dims.x * vdbVolume_.voxelSize.x,
        vdbVolume_.dims.y * vdbVolume_.voxelSize.y,
        vdbVolume_.dims.z * vdbVolume_.voxelSize.z ) );
    res.push_back( fmt::format( "min-value: {:.3}", vdbVolume_.min ) );
    res.push_back( fmt::format( "iso-value: {:.3}", isoValue_ ) );
    res.push_back( fmt::format( "max-value: {:.3}", vdbVolume_.max ) );
    res.push_back( dualMarchingCubes_ ? "visual: dual marching cubes" : "visual: standard marching cubes" );
    return res;
}

}
#endif
