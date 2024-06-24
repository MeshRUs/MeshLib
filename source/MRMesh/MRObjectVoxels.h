#pragma once
#include "MRMeshFwd.h"
#ifndef MRMESH_NO_OPENVDB
#include "MRObjectMeshHolder.h"
#include "MRProgressCallback.h"
#include "MRHistogram.h"
#include "MRVolumeIndexer.h"
#include "MRSimpleVolume.h"
#include "MRMarchingCubes.h"

namespace MR
{

/// This class stores information about voxels object
/// \ingroup DataModelGroup
class MRMESH_CLASS ObjectVoxels : public ObjectMeshHolder
{
public:
    MRMESH_API ObjectVoxels();
    ObjectVoxels& operator = ( ObjectVoxels&& ) noexcept = default;
    ObjectVoxels( ObjectVoxels&& ) noexcept = default;
    virtual ~ObjectVoxels() = default;

    constexpr static const char* TypeName() noexcept { return "ObjectVoxels"; }
    virtual const char* typeName() const override { return TypeName(); }

    MRMESH_API virtual void applyScale( float scaleFactor ) override;

    /// Returns iso surface, empty if iso value is not set
    const std::shared_ptr<Mesh>& surface() const { return mesh_; }

    /// Return VdbVolume
    const VdbVolume& vdbVolume() const { return vdbVolume_; };

    /// Returns Float grid which contains voxels data, see more on openvdb::FloatGrid
    const FloatGrid& grid() const { return vdbVolume_.data; }

    [[nodiscard]] virtual bool hasModel() const override { return bool( vdbVolume_.data ); }

    /// Returns dimensions of voxel objects
    const Vector3i& dimensions() const
    { return vdbVolume_.dims; }
    /// Returns current iso value
    float getIsoValue() const
    { return isoValue_; }
    /// Returns histogram
    const Histogram& histogram() const
    { return histogram_; }

    const Vector3f& voxelSize() const
    { return vdbVolume_.voxelSize; }

    MRMESH_API virtual std::vector<std::string> getInfoLines() const override;
    virtual std::string getClassName() const override { return "Voxels"; }

    /// Clears all internal data and then creates grid and calculates histogram (surface is not built, call \ref updateHistogramAndSurface)
    MRMESH_API void construct( const SimpleVolume& simpleVolume, ProgressCallback cb = {} );
    /// Clears all internal data and calculates histogram
    MRMESH_API void construct( const FloatGrid& grid, const Vector3f& voxelSize, ProgressCallback cb = {} );
    /// Clears all internal data and calculates histogram
    MRMESH_API void construct( const VdbVolume& vdbVolume, ProgressCallback cb = {} );
    /// Updates histogram, by stored grid (evals min and max values from grid)
    /// rebuild iso surface if it is present
    MRMESH_API void updateHistogramAndSurface( ProgressCallback cb = {} );

    /// Sets iso value and updates iso-surfaces if needed: 
    /// Returns true if iso-value was updated, false - otherwise
    MRMESH_API virtual Expected<bool, std::string> setIsoValue( float iso, ProgressCallback cb = {}, bool updateSurface = true );

    /// Sets external surface mesh for this object
    /// and returns back previous mesh of this
    MRMESH_API std::shared_ptr<Mesh> updateIsoSurface( std::shared_ptr<Mesh> mesh );
    /// Sets external vdb volume for this object
    /// and returns back previous vdb volume of this
    MRMESH_API VdbVolume updateVdbVolume( VdbVolume vdbVolume );
    /// Sets external histogram for this object
    /// and returns back previous histogram of this
    MRMESH_API Histogram updateHistogram( Histogram histogram );
    /// Sets active bounds for this object
    /// and returns back previous active bounds of this
    MRMESH_API Box3i updateActiveBounds( const Box3i &box );

    /// Calculates and return new mesh or error message
    MRMESH_API Expected<std::shared_ptr<Mesh>, std::string> recalculateIsoSurface( float iso, ProgressCallback cb = {} ) const;
    /// returns true if the iso-surface is built using Dual Marching Cubes algorithm or false if using Standard Marching Cubes
    bool getDualMarchingCubes() const { return dualMarchingCubes_; }
    /// sets whether to use Dual Marching Cubes algorithm for visualization (true) or Standard Marching Cubes (false);
    /// \param updateSurface forces immediate update
    MRMESH_API virtual void setDualMarchingCubes( bool on, bool updateSurface = true, ProgressCallback cb = {} );
    /// set voxel point positioner for Marching Cubes (only for Standard Marching Cubes)
    virtual void setVoxelPointPositioner( VoxelPointPositioner positioner ) { positioner_ = positioner; }


    /// Sets active bounds for some simplifications (max excluded)
    /// active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty 
    MRMESH_API virtual void setActiveBounds( const Box3i& activeBox, ProgressCallback cb = {}, bool updateSurface = true );
    /// Returns active bounds (max excluded)
    /// active bounds is box in voxel coordinates, note that voxels under (0,0,0) and voxels over (dimensions) are empty 
    const Box3i& getActiveBounds() const
    { return activeBox_; }

    const VoxelBitSet& getSelectedVoxels() const { return selectedVoxels_; }
    void selectVoxels( const VoxelBitSet& selectedVoxels ) { selectedVoxels_ = selectedVoxels; }
    
    /// get active (visible) voxels
    const VoxelBitSet& getVolumeRenderActiveVoxels() const { return volumeRenderActiveVoxels_; }
    /// set active (visible) voxels (using only in Volume Rendering mode)
    MRMESH_API void setVolumeRenderActiveVoxels( const VoxelBitSet& activeVoxels );

    /// VoxelId is numerical representation of voxel
    /// Coordinate is {x,y,z} indices of voxels in box (base dimensions space, NOT active dimensions)
    /// Point is local space coordinate of point in scene
    MRMESH_API VoxelId getVoxelIdByCoordinate( const Vector3i& coord ) const;
    MRMESH_API VoxelId getVoxelIdByPoint( const Vector3f& point ) const;
    MRMESH_API Vector3i getCoordinateByVoxelId( VoxelId id ) const;

    /// Returns indexer with more options
    const VolumeIndexer& getVolumeIndexer() const { return indexer_; }

    // prepare data for volume rendering
    // returns false if canceled or voxel data is empty
    MRMESH_API bool prepareDataForVolumeRendering( ProgressCallback cb = {} ) const;

    bool isVolumeRenderingEnabled() const { return volumeRendering_; }
    // this function should only be called from GUI thread because it changes rendering object,
    // it can take some time to prepare data, so you can prepare data with progress callback
    // by calling `prepareDataForVolumeRendering(cb)` function before calling this one
    MRMESH_API void enableVolumeRendering( bool on );
    // move volume rendering data to caller: basically used in RenderVolumeObject 
    [[nodiscard]] std::unique_ptr<SimpleVolume> getVolumeRenderingData() const { return std::move( volumeRenderingData_ ); }

    // struct to control volume rendering texture
    struct VolumeRenderingParams
    {
        // volume texture smoothing
        FilterType volumeFilterType{ FilterType::Linear };
        // shading model
        enum class ShadingType
        {
            None,
            ValueGradient,
            AlphaGradient
        } shadingType{ ShadingType::None };
        // coloring type
        enum class LutType
        {
            GrayShades,
            Rainbow,
            OneColor
        } lutType{ LutType::Rainbow };
        // color that is used for OneColor mode
        Color oneColor{ Color::white() };
        // minimum colored value (voxels with lower values are transparent)
        float min{ 0.0f };
        // maximum colored value (voxels with higher values are transparent)
        float max{ 0.0f };
        // type of alpha function on texture
        enum class AlphaType
        {
            Constant,
            LinearIncreasing,
            LinearDecreasing
        } alphaType{ AlphaType::Constant };
        uint8_t alphaLimit{ 10 };
        bool operator==( const VolumeRenderingParams& )const = default;
    };
    const VolumeRenderingParams& getVolumeRenderingParams() const { return volumeRenderingParams_; }
    MRMESH_API void setVolumeRenderingParams( const VolumeRenderingParams& params );

    MRMESH_API virtual bool hasVisualRepresentation() const override;

    /// sets top limit on the number of vertices in the iso-surface
    MRMESH_API void setMaxSurfaceVertices( int maxVerts );
    /// gets top limit on the number of vertices in the iso-surface
    int getMaxSurfaceVertices() const { return maxSurfaceVertices_; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    MRMESH_API virtual void setDirtyFlags( uint32_t mask, bool invalidateCaches = true ) override;

    /// \note this ctor is public only for std::make_shared used inside clone()
    ObjectVoxels( ProtectedStruct, const ObjectVoxels& obj ) : ObjectVoxels( obj ) {}

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const override;

    /// signal about Iso-surface changes (from updateIsoSurface)
    using IsoSurfaceChangedSignal = Signal<void()>;
    IsoSurfaceChangedSignal isoSurfaceChangedSignal;

private:
    VolumeRenderingParams volumeRenderingParams_;
    mutable UniquePtr<SimpleVolume> volumeRenderingData_;

    int maxSurfaceVertices_{ 5'000'000 };
    VdbVolume vdbVolume_;
    float isoValue_{0.0f};
    bool dualMarchingCubes_{true};
    VoxelPointPositioner positioner_ = {};
    Histogram histogram_;
    Box3i activeBox_;

    /// Service data
    VolumeIndexer indexer_ = VolumeIndexer( vdbVolume_.dims );
    Vector3f reverseVoxelSize_;

    void updateHistogram_( float min, float max, ProgressCallback cb = {} );


    /// this is private function to set default colors of this type (ObjectVoxels) in constructor only
    void setDefaultColors_();

    /// set default scene-related properties
    void setDefaultSceneProperties_();

protected:
    VoxelBitSet selectedVoxels_;
    VoxelBitSet volumeRenderActiveVoxels_;

    ObjectVoxels( const ObjectVoxels& other ) = default;
    bool volumeRendering_{ false };

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    MRMESH_API void deserializeFields_( const Json::Value& root ) override;

    MRMESH_API VoidOrErrStr deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} ) override;

    MRMESH_API virtual Expected<std::future<VoidOrErrStr>> serializeModel_( const std::filesystem::path& path ) const override;
};


}
#endif
