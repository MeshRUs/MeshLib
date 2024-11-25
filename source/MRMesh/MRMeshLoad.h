#pragma once

#include "MRMeshFwd.h"
#include "MRIOFilters.h"
#include "MRId.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

// new simpler names

/// loads mesh from file in internal MeshLib format
MRMESH_API Expected<Mesh> loadMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in internal MeshLib format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> loadMrmesh( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .OFF format
MRMESH_API Expected<Mesh> loadOff( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .OFF format
MRMESH_API Expected<Mesh> loadOff( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .OBJ format
MRMESH_API Expected<Mesh> loadObj( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .OBJ format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> loadObj( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in any .STL format: both binary and ASCII
MRMESH_API Expected<Mesh> loadStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in any .STL format: both binary and ASCII;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> loadStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in binary .STL format
MRMESH_API Expected<Mesh> loadBinaryStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in binary .STL format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> loadBinaryStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in textual .STL format
MRMESH_API Expected<Mesh> loadASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in textual .STL format
MRMESH_API Expected<Mesh> loadASCIIStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .PLY format;
MRMESH_API Expected<Mesh> loadPly( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .PLY format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> loadPly( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .DXF format;
MRMESH_API Expected<Mesh> loadDxf( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .DXF format;
MRMESH_API Expected<Mesh> loadDxf( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in the format detected from file extension
MRMESH_API Expected<Mesh> loadMesh( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in the format detected from given extension-string (`*.ext`);
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> loadMesh( std::istream& in, const std::string& extension, const MeshLoadSettings& settings = {} );

// compatibility names
namespace MeshLoad
{

/// \defgroup MeshLoadGroup Mesh Load
/// \ingroup IOGroup
/// \{

/// loads mesh from file in internal MeshLib format
MRMESH_API Expected<Mesh> fromMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in internal MeshLib format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromMrmesh( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .OFF format
MRMESH_API Expected<Mesh> fromOff( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .OFF format
MRMESH_API Expected<Mesh> fromOff( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .OBJ format
MRMESH_API Expected<Mesh> fromObj( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .OBJ format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromObj( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in any .STL format: both binary and ASCII
MRMESH_API Expected<Mesh> fromAnyStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in any .STL format: both binary and ASCII;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromAnyStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in binary .STL format
MRMESH_API Expected<Mesh> fromBinaryStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in binary .STL format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromBinaryStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in textual .STL format
MRMESH_API Expected<Mesh> fromASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in textual .STL format
MRMESH_API Expected<Mesh> fromASCIIStl( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .PLY format;
MRMESH_API Expected<Mesh> fromPly( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .PLY format;
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromPly( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in .DXF format;
MRMESH_API Expected<Mesh> fromDxf( const std::filesystem::path& path, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in .DXF format;
MRMESH_API Expected<Mesh> fromDxf( std::istream& in, const MeshLoadSettings& settings = {} );

/// loads mesh from file in the format detected from file extension
MRMESH_API Expected<Mesh> fromAnySupportedFormat( const std::filesystem::path& file, const MeshLoadSettings& settings = {} );

/// loads mesh from stream in the format detected from given extension-string (`*.ext`);
/// important on Windows: in stream must be open in binary mode
MRMESH_API Expected<Mesh> fromAnySupportedFormat( std::istream& in, const std::string& extension, const MeshLoadSettings& settings = {} );

/// \}

} // namespace MeshLoad

} // namespace MR
