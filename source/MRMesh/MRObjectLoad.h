#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include "MRUniqueTemporaryFolder.h"
#include "MRLoadedObjects.h"

#include <filesystem>

namespace MR
{

/// \ingroup DataModelGroup
/// \{

/// information about loading process and mesh construction from primitives
struct MeshLoadInfo
{
    std::string* warnings = nullptr; ///< any warnings during loading will be appended here
    ProgressCallback callback;       ///< callback for set progress and stop process
};

/// loads mesh from given file in new object
MRMESH_API Expected<ObjectMesh> makeObjectMeshFromFile( const std::filesystem::path& file, const MeshLoadInfo& info = {} );

/// loads mesh from given file and makes either ObjectMesh or ObjectPoints (if the file has points but not faces)
MRMESH_API Expected<std::shared_ptr<Object>> makeObjectFromMeshFile( const std::filesystem::path& file, const MeshLoadInfo& info = {},
    bool returnOnlyMesh = false ); ///< if true the function can return only ObjectMesh and never ObjectPoints

/// loads lines from given file in new object
MRMESH_API Expected<ObjectLines> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads points from given file in new object
MRMESH_API Expected<ObjectPoints> makeObjectPointsFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads distance map from given file in new object
MRMESH_API Expected<ObjectDistanceMap> makeObjectDistanceMapFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/// loads gcode from given file in new object
MRMESH_API Expected<ObjectGcode> makeObjectGcodeFromFile( const std::filesystem::path& file, ProgressCallback callback = {} );

/**
 * \brief load all objects (or any type: mesh, lines, points, voxels or scene) from file
 * \param callback - callback function to set progress (for progress bar)
 */
MRMESH_API Expected<LoadedObjects> loadObjectFromFile( const std::filesystem::path& filename, const ProgressCallback& callback = {} );

// check if there are any supported files folder and subfolders
MRMESH_API bool isSupportedFileInSubfolders( const std::filesystem::path& folder );

/// loads meshes from given folder in new container object
MRMESH_API Expected<Object> makeObjectTreeFromFolder( const std::filesystem::path& folder, std::string* loadWarn = nullptr, ProgressCallback callback = {} );

//tries to load scene from every format listed in SceneFormatFilters
MRMESH_API Expected<std::shared_ptr<Object>> loadSceneFromAnySupportedFormat( const std::filesystem::path& path, 
    std::string* loadWarn = nullptr, ProgressCallback callback = {} );

/**
 * \brief loads objects tree from given scene file (zip/mru)
 * \details format specification:
 *  children are saved under folder with name of their parent object
 *  all objects parameters are saved in one JSON file in the root folder
 *
 * if postDecompress is set, it is called after decompression
 * loading is controlled with Object::deserializeModel_ and Object::deserializeFields_
 */
MRMESH_API Expected<std::shared_ptr<Object>> deserializeObjectTree( const std::filesystem::path& path,
                                                                    FolderCallback postDecompress = {},
                                                                    ProgressCallback progressCb = {} );

/**
 * \brief loads objects tree from given scene folder
 * \details format specification:
 *  children are saved under folder with name of their parent object
 *  all objects parameters are saved in one JSON file in the root folder
 *
 * loading is controlled with Object::deserializeModel_ and Object::deserializeFields_
 */
MRMESH_API Expected<std::shared_ptr<Object>> deserializeObjectTreeFromFolder( const std::filesystem::path& folder,
                                                                              ProgressCallback progressCb = {} );

/// \}

} // namespace MR
