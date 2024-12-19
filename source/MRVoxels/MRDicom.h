#pragma once

#include "MRVoxelsFwd.h"
#ifndef MRVOXELS_NO_DICOM
#include "MRVoxelsVolume.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRLoadedObjects.h"

#include <filesystem>
#include <optional>

namespace MR
{

namespace VoxelsLoad
{

/// check if file is a valid DICOM dataset file
/// \param seriesUid - if set, the extracted series instance UID is copied to the variable
MRVOXELS_API bool isDicomFile( const std::filesystem::path& path, std::string* seriesUid = nullptr );

/// check if given folder contains at least one DICOM file
MRVOXELS_API bool isDicomFolder( const std::filesystem::path& dirPath );

template <typename T>
struct DicomVolumeT
{
    T vol;
    std::string name;
    AffineXf3f xf;
};

using DicomVolume = DicomVolumeT<SimpleVolumeMinMax>;
using DicomVolumeAsVdb = DicomVolumeT<VdbVolume>;


/// Loads 3D all volumetric data from DICOM files in a folder
/// @note Explicitly instantiated for T = SimpleVolumeMinMax and T = VdbVolume
template <typename T = SimpleVolumeMinMax>
MRVOXELS_API std::vector<Expected<DicomVolumeT<T>>> loadDicomsFolder( const std::filesystem::path& path,
                                                                      unsigned maxNumThreads = 4, const ProgressCallback& cb = {}  );

/// Loads 3D first volumetric data from DICOM files in a folder
/// @note Explicitly instantiated for T = SimpleVolumeMinMax and T = VdbVolume
template <typename T = SimpleVolumeMinMax>
MRVOXELS_API Expected<DicomVolumeT<T>> loadDicomFolder( const std::filesystem::path& path,
                                                    unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads every subfolder with DICOM volume as new object
MRVOXELS_API std::vector<Expected<DicomVolumeAsVdb>> loadDicomsFolderTreeAsVdb( const std::filesystem::path& path,
                                                                     unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// converts DicomVolumeAsVdb in ObjectVoxels
MRVOXELS_API Expected<std::shared_ptr<ObjectVoxels>> createObjectVoxels( const DicomVolumeAsVdb & dcm, const ProgressCallback & cb = {} );

/// Loads 3D volumetric data from dicom-files in given folder, and converts them into an ObjectVoxels
/// \param scaleFactor multiplier for voxel size
MRVOXELS_API Expected<LoadedObjectVoxels> makeObjectVoxelsFromDicomFolder( const std::filesystem::path& folder,
    float scaleFactor, const ProgressCallback& callback = {} );

/// Loads 3D volumetric data from a single DICOM file
template <typename T = SimpleVolumeMinMax>
MRVOXELS_API Expected<DicomVolumeT<T>> loadDicomFile( const std::filesystem::path& path, const ProgressCallback& cb = {} );

} // namespace VoxelsLoad

namespace VoxelsSave
{

/// Save voxels objet to a single 3d DICOM file
MRVOXELS_API Expected<void> toDicom( const VdbVolume& vdbVolume, const std::filesystem::path& path, ProgressCallback cb = {} );
/// Saves object to a single 3d DICOM file. \p sourceScale specifies the true scale of the voxel data
/// which will be saved with "slope" and "intercept" parameters of the output dicom.
template <typename T>
MRVOXELS_API Expected<void> toDicom( const VoxelsVolume<std::vector<T>>& volume, const std::filesystem::path& path, const std::optional<MinMaxf>& sourceScale = {}, const ProgressCallback& cb = {} );

} // namespace VoxelsSave

} // namespace MR
#endif