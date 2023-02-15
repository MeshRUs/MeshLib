#pragma once
#include "MRMeshFwd.h"
#include "MRObject.h"

namespace MR
{
//loads scene from glTF file in a new container object
MRMESH_API tl::expected<std::shared_ptr<Object>, std::string> deserializeObjectTreeFromGltf( const std::filesystem::path& file, ProgressCallback callback = {} );

MRMESH_API tl::expected<void, std::string> serializeObjectTreeToGltf( const std::filesystem::path& file, std::shared_ptr<Object> objectTree, ProgressCallback callback = {} );

}
