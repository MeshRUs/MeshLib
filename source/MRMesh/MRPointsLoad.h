#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

namespace PointsLoad
{

/// \defgroup PointsLoadGroup Points Load
/// \addtogroup IOGroup
/// \{

MRMESH_API extern const IOFilters Filters;

/// loads from .csv, .xyz, .txt file
MRMESH_API Expected<PointCloud, std::string> fromText( const std::filesystem::path& file, AffineXf3f* outXf = nullptr, ProgressCallback callback = {} );
MRMESH_API Expected<PointCloud, std::string> fromText( std::istream& in, AffineXf3f* outXf = nullptr, ProgressCallback callback = {} );


#ifndef MRMESH_NO_OPENCTM
/// loads from .ctm file
MRMESH_API Expected<PointCloud, std::string> fromCtm( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                          ProgressCallback callback = {} );
MRMESH_API Expected<PointCloud, std::string> fromCtm( std::istream& in, VertColors* colors = nullptr,
                                                          ProgressCallback callback = {} );
#endif

/// loads from .ply file
MRMESH_API Expected<PointCloud, std::string> fromPly( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                          ProgressCallback callback = {} );
MRMESH_API Expected<PointCloud, std::string> fromPly( std::istream& in, VertColors* colors = nullptr,
                                                          ProgressCallback callback = {} );

/// loads from .obj file
MRMESH_API Expected<PointCloud, std::string> fromObj( const std::filesystem::path& file, ProgressCallback callback = {} );
MRMESH_API Expected<PointCloud, std::string> fromObj( std::istream& in, ProgressCallback callback = {} );

/// loads from .asc file
MRMESH_API Expected<PointCloud, std::string> fromAsc( const std::filesystem::path& file, ProgressCallback callback = {} );
MRMESH_API Expected<PointCloud, std::string> fromAsc( std::istream& in, ProgressCallback callback = {} );

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
/// loads from .e57 file
MRMESH_API Expected<PointCloud, std::string> fromE57( const std::filesystem::path& file, VertColors* colors = nullptr, ProgressCallback callback = {} );
// no support for reading e57 from arbitrary stream yet
#endif

/// detects the format from file extension and loads points from it
MRMESH_API Expected<PointCloud, std::string> fromAnySupportedFormat( const std::filesystem::path& file, VertColors* colors = nullptr,
                                                                         ProgressCallback callback = {} );
/// extension in `*.ext` format
MRMESH_API Expected<PointCloud, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, VertColors* colors = nullptr,
                                                                         ProgressCallback callback = {} );

/// \}

} // namespace PointsLoad

} // namespace MR
