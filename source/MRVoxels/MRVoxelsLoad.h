#pragma once
#include "MRVoxelsFwd.h"
#include "MRScalarConvert.h"
#include "MRVoxelsVolume.h"

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRObject.h"
#include <MRMesh/MRLoadedObjects.h>

#include <filesystem>

namespace MR
{

namespace VoxelsLoad
{

/// \defgroup VoxelsLoadGroup Voxels Load
/// \ingroup IOGroup
/// \{

struct RawParameters
{
    Vector3i dimensions;
    Vector3f voxelSize;
    bool gridLevelSet{ false }; ///< OpenVDB GridClass set as GRID_LEVEL_SET (need to set right surface normals direction)
    ScalarType scalarType{ ScalarType::Float32 };
};
/// Load raw voxels from file with provided parameters
MRVOXELS_API Expected<VdbVolume> fromRaw( const std::filesystem::path& file, const RawParameters& params,
                                                         const ProgressCallback& cb = {} );
/// Load raw voxels from stream with provided parameters;
/// important on Windows: in stream must be open in binary mode
MRVOXELS_API Expected<VdbVolume> fromRaw( std::istream& in, const RawParameters& params,
                                                         const ProgressCallback& cb = {} );

/// finds raw voxels file and its encoding parameters
/// \param file on input: file name probably without suffix with parameters
///             on output: if success existing file name
MRVOXELS_API Expected<RawParameters> findRawParameters( std::filesystem::path& file );

/// Load raw voxels file, parsing parameters from name 
MRVOXELS_API Expected<VdbVolume> fromRaw( const std::filesystem::path& file,
                                                         const ProgressCallback& cb = {} );

/// Load all voxel volumes from OpenVDB file
MRVOXELS_API Expected<std::vector<VdbVolume>> fromVdb( const std::filesystem::path& file,
                                                         const ProgressCallback& cb = {} );

/// Load voxel from Gav-file with micro CT reconstruction
MRVOXELS_API Expected<VdbVolume> fromGav( const std::filesystem::path& file, const ProgressCallback& cb = {} );
/// Load voxel from Gav-stream with micro CT reconstruction
MRVOXELS_API Expected<VdbVolume> fromGav( std::istream& in, const ProgressCallback& cb = {} );


/// Detects the format from file extension and loads voxels from it
MRVOXELS_API Expected<std::vector<VdbVolume>> fromAnySupportedFormat( const std::filesystem::path& file,
                                                                        const ProgressCallback& cb = {} );

/// \}

// Determines iso-surface orientation
enum class GridType
{
    // consider values less than iso as outer area
    DenseGrid = 0,
    // consider values less than iso as inner area
    LevelSet = 1
};

#ifndef MRVOXELS_NO_TIFF
struct LoadingTiffSettings
{
    std::filesystem::path dir;
    Vector3f voxelSize = { 1.0f, 1.0f, 1.0f };
    GridType gridType = GridType::DenseGrid;
    ProgressCallback cb = {};
};
/// Load voxels from a set of TIFF files
MRVOXELS_API Expected<VdbVolume> loadTiffDir( const LoadingTiffSettings& settings );
#endif // MRVOXELS_NO_TIFF

#ifndef MR_PARSING_FOR_PB11_BINDINGS
using VoxelsLoader = Expected<std::vector<VdbVolume>>( * )( const std::filesystem::path&, const ProgressCallback& );

MR_FORMAT_REGISTRY_EXTERNAL_DECL( MRVOXELS_API, VoxelsLoader )
#endif

}

/// loads voxels from given file in new object
MRVOXELS_API Expected<std::vector<std::shared_ptr<ObjectVoxels>>> makeObjectVoxelsFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

MRVOXELS_API Expected<LoadedObjects> makeObjectFromVoxelsFile( const std::filesystem::path& file, const ProgressCallback& callback = {} );

}
