#pragma once

#include "config.h"
#ifndef MRIOEXTRAS_NO_E57
#include "exports.h"

#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRColor.h>
#include <MRMesh/MRExpected.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRPointsLoadSettings.h>
#include <MRMesh/MRLoadedObjects.h>

#include <filesystem>
#include <string>

namespace MR::PointsLoad
{

struct E57LoadSettings
{
    /// true => if input file has more than one cloud, they all will be combined in one
    bool combineAllObjects = false;

    /// true => return only identity transforms, applying them to points
    bool identityXf = false;

    /// progress report and cancellation
    ProgressCallback progress;
};

/// loads scene from e57 file
struct NamedCloud
{
    std::string name;
    PointCloud cloud;
    AffineXf3f xf;
    VertColors colors;
};

MRIOEXTRAS_API Expected<std::vector<NamedCloud>> fromSceneE57File( const std::filesystem::path& file,
                                                                   const E57LoadSettings & settings = {} );

/// loads from .e57 file
MRIOEXTRAS_API Expected<PointCloud> fromE57( const std::filesystem::path& file,
                                             const PointsLoadSettings& settings = {} );
MRIOEXTRAS_API Expected<PointCloud> fromE57( std::istream& in, const PointsLoadSettings& settings = {} );

MRIOEXTRAS_API Expected<LoadedObjects> loadObjectFromE57( const std::filesystem::path& path, const ProgressCallback& cb = {} );

} // namespace MR::PointsLoad
#endif
