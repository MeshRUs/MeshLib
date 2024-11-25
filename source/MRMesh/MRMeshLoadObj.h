#pragma once

#include "MRMesh.h"
#include "MRMeshTexture.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRMeshLoadSettings.h"
#include "MRAffineXf3.h"
#include "MRLoadedObjects.h"
#include <filesystem>
#include <istream>
#include <string>

namespace MR
{

namespace MeshLoad
{

/// \defgroup MeshLoadObjGroup Mesh Load Obj
/// \ingroup IOGroup
/// \{

struct ObjLoadSettings
{
    /// if true then vertices will be returned relative to some transformation to avoid precision loss
    bool customXf = false;

    /// if true, the number of skipped faces (faces than can't be created) will be counted
    bool countSkippedFaces = false;

    /// callback for set progress and stop process
    ProgressCallback callback;
};

struct NamedMesh
{
    std::string name;
    Mesh mesh;
    VertUVCoords uvCoords;
    VertColors colors;
    Vector<std::filesystem::path, TextureId> textureFiles;
    Vector<TextureId, FaceId> texturePerFace;
    std::optional<Color> diffuseColor;

    /// transform of the loaded mesh, not identity only if ObjLoadSettings.customXf
    AffineXf3f xf;

    /// counter of skipped faces (faces than can't be created), not zero only if ObjLoadSettings.countSkippedFaces
    int skippedFaceCount = 0;

    /// counter of duplicated vertices (that created for resolve non-manifold geometry)
    int duplicatedVertexCount = 0;
};

/// loads meshes from .obj file
MRMESH_API Expected<std::vector<NamedMesh>> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                           const ObjLoadSettings& settings = {} );

/// loads meshes from a stream with .obj file contents
/// important on Windows: in stream must be open in binary mode
/// \param dir working directory where materials and textures are located
MRMESH_API Expected<std::vector<NamedMesh>> fromSceneObjFile( std::istream& in, bool combineAllObjects, const std::filesystem::path& dir,
                                                                           const ObjLoadSettings& settings = {} );

/// loads meshes from memory array with .obj file contents
/// \param dir working directory where materials and textures are located
MRMESH_API Expected<std::vector<NamedMesh>> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects, const std::filesystem::path& dir,
                                                                           const ObjLoadSettings& settings = {} );

/// reads all objects from .OBJ file
MRMESH_API Expected<LoadedObjects> loadObjectFromObj( const std::filesystem::path& file, const ProgressCallback& cb = {} );

} // namespace MeshLoad

} // namespace MR
