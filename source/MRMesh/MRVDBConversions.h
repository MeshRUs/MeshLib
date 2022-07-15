#pragma once

#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRProgressCallback.h"
#include <tl/expected.hpp>
#include <string>

namespace MR
{
struct SimpleVolume;
// closed surface is required
// surfaceOffset - number voxels around surface to calculate distance in (should be positive)
MRMESH_API FloatGrid meshToLevelSet( const MeshPart& mp, const AffineXf3f& xf,
                                     const Vector3f& voxelSize, float surfaceOffset = 3,
                                     const ProgressCallback& cb = {} );

// does not require closed surface, resulting grid cannot be used for boolean operations,
// surfaceOffset - the number voxels around surface to calculate distance in (should be positive)
MRMESH_API FloatGrid meshToDistanceField( const MeshPart& mp, const AffineXf3f& xf,
                                          const Vector3f& voxelSize, float surfaceOffset = 3,
                                          const ProgressCallback& cb = {} );

// make FloatGrid from SimpleVolume
// make copy of data
// grid can be used to make iso-surface later with gridToMesh function
MRMESH_API FloatGrid simpleVolumeToDenseGrid( const SimpleVolume& simpleVolue,
                                              const ProgressCallback& cb = {} );

// isoValue - layer of grid with this value would be converted in mesh
// isoValue can be negative only in level set grids
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
MRMESH_API Mesh gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
                            float isoValue = 0.0f, float adaptivity = 0.0f,
                            const ProgressCallback& cb = {} );

// isoValue - layer of grid with this value would be converted in mesh
// isoValue can be negative only in level set grids
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
// maxFaces if mesh faces exceed this value error returns
MRMESH_API tl::expected<Mesh, std::string> gridToMesh( const FloatGrid& grid, const Vector3f& voxelSize,
    int maxFaces,
    float isoValue = 0.0f, float adaptivity = 0.0f, const ProgressCallback& cb = {} );

// performs convention from mesh to levelSet and back with offsetA, and than same with offsetB
// allowed only for closed meshes
// adaptivity - [0.0;1.0] ratio of combining small triangles into bigger ones 
//                       (curvature can be lost on high values)
MRMESH_API Mesh levelSetDoubleConvertion( const MeshPart& mp, const AffineXf3f& xf,
    float voxelSize, float offsetA, float offsetB, float adaptivity = 0.0f,
    const ProgressCallback& cb = {} );

}
#endif
