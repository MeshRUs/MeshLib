#pragma once

#include "MRVoxelsFwd.h"
#ifndef MRVOXELS_NO_DICOM
#include "MRVoxelsVolume.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"

#include <filesystem>
#include <optional>

namespace MR
{

namespace VoxelsLoad
{

/// check if file is a valid DICOM dataset file
/// \param seriesUid - if set, the extracted series instance UID is copied to the variable
MRVOXELS_API bool isDicomFile( const std::filesystem::path& path, std::string* seriesUid = nullptr );

struct DicomVolume
{
    SimpleVolumeMinMax vol;
    std::string name;
    AffineXf3f xf;
};

struct DicomVolumeAsVdb
{
    VdbVolume vdbVolume;
    std::string name;
    AffineXf3f xf;
};

/// Loads 3D all volumetric data from DICOM files in a folder
MRVOXELS_API std::vector<Expected<DicomVolumeAsVdb>> loadDicomsFolderAsVdb( const std::filesystem::path& path,
                                                                  unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );
/// Loads 3D first volumetric data from DICOM files in a folder
MRVOXELS_API Expected<DicomVolumeAsVdb> loadDicomFolderAsVdb( const std::filesystem::path& path,
                                                    unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads 3D all volumetric data from DICOM files in a folder
MRVOXELS_API std::vector<Expected<DicomVolume>> loadDicomsFolder( const std::filesystem::path& path,
                                                                  unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );
/// Loads 3D first volumetric data from DICOM files in a folder
MRVOXELS_API Expected<DicomVolume> loadDicomFolder( const std::filesystem::path& path,
                                                    unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// Loads every subfolder with DICOM volume as new object
MRVOXELS_API std::vector<Expected<DicomVolumeAsVdb>> loadDicomsFolderTreeAsVdb( const std::filesystem::path& path,
                                                                     unsigned maxNumThreads = 4, const ProgressCallback& cb = {} );

/// converts DicomVolumeAsVdb in ObjectVoxels
MRVOXELS_API Expected<std::shared_ptr<ObjectVoxels>> createObjectVoxels( const DicomVolumeAsVdb & dcm, const ProgressCallback & cb = {} );

/// Loads 3D volumetric data from a single DICOM file
MRVOXELS_API Expected<DicomVolume> loadDicomFile( const std::filesystem::path& path, const ProgressCallback& cb = {} );
MRVOXELS_API Expected<DicomVolumeAsVdb> loadDicomFileAsVdb( const std::filesystem::path& path, const ProgressCallback& cb = {} );

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