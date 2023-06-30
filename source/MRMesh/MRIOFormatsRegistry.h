#pragma once
#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <filesystem>

namespace MR
{

namespace MeshLoad
{

/// \defgroup IOFormatsRegistryGroup IO Formats Registry
/// \ingroup IOGroup
/// \{

using MeshLoader = Expected<MR::Mesh, std::string>( * )( const std::filesystem::path&, VertColors*, ProgressCallback );
using MeshStreamLoader = Expected<MR::Mesh, std::string>( * )( std::istream&, VertColors*, ProgressCallback );

struct NamedMeshLoader
{
    IOFilter filter;
    MeshLoader loader{ nullptr };
    MeshStreamLoader streamLoader{ nullptr };
};

/// Finds expected loader from registry
MRMESH_API MeshLoader getMeshLoader( IOFilter filter );
/// Finds expected loader from registry
MRMESH_API MeshStreamLoader getMeshStreamLoader( IOFilter filter );
/// Gets all registered filters
MRMESH_API IOFilters getFilters();

/** 
 * \brief Register filter with loader function
 * \details loader function signature: Expected<Mesh, std::string> fromFormat( const std::filesystem::path& path, std::vector<Color>* colors );
 * example:
 * ADD_MESH_LOADER( IOFilter("Name of filter (.ext)","*.ext"), fromFormat)
 */
#define MR_ADD_MESH_LOADER( filter, loader ) \
MR::MeshLoad::MeshLoaderAdder __meshLoaderAdder_##loader(MR::MeshLoad::NamedMeshLoader{filter,static_cast<MR::MeshLoad::MeshLoader>(loader),static_cast<MR::MeshLoad::MeshStreamLoader>(loader)});\

class MeshLoaderAdder
{
public:
    MRMESH_API MeshLoaderAdder( const NamedMeshLoader& loader );
};

/// \}

}

}
