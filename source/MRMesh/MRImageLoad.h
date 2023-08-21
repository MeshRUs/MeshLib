#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRExpected.h"
#include <filesystem>

namespace MR
{

struct Image;

namespace ImageLoad
{

/// \defgroup ImageLoadGroup Image Load
/// \ingroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

#ifndef MRMESH_NO_PNG
/// loads from .png format
MRMESH_API Expected<Image, std::string> fromPng( const std::filesystem::path& path );
MRMESH_API Expected<Image, std::string> fromPng( std::istream& in );
#endif

#ifndef __EMSCRIPTEN__

#ifndef MRMESH_NO_JPEG
/// loads from .jpg format
MRMESH_API Expected<Image, std::string> fromJpeg( const std::filesystem::path& path );
MRMESH_API Expected<Image, std::string> fromJpeg( std::istream& in );
#endif

#endif

/// detects the format from file extension and loads image from it
MRMESH_API Expected<Image, std::string> fromAnySupportedFormat( const std::filesystem::path& path );

/// \}

}

}
