#pragma once

#include "exports.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRProgressCallback.h"
#include <filesystem>

namespace MR
{
class Object;

struct SaveObjectSettings
{
    /// if true then before saving, original files is renamed, and renamed back if saving fails
    bool backupOriginalFile = false;

    /// callback function to set progress (for progress bar)
    ProgressCallback callback;
};


/**
 * \brief save visual object (mesh, lines, points or voxels) to file
 * \return empty string if no error or error text
 */
MRVIEWER_API Expected<void> saveObjectToFile( const Object& obj, const std::filesystem::path& filename,
    const SaveObjectSettings & settings = {} );

} //namespace MR
