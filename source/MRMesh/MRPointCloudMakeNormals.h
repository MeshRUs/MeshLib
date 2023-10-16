#pragma once

#include "MRMeshFwd.h"
#include <optional>

namespace MR
{

/// \brief Makes normals for valid points of given point cloud by directing them along the normal (in one of two sides arbitrary) of best plane through the neighbours
/// \param radius of neighborhood to consider
/// \return nullopt if progress returned false
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud,
    float radius, const ProgressCallback & progress = {} );

/// \brief Makes normals for valid points of given point cloud by directing them along the normal (in one of two sides arbitrary) of best plane through the neighbours
/// \param closeVert a buffer where for every valid point #i its neighbours are stored at indices [i*numNei; (i+1)*numNei)
/// \return nullopt if progress returned false
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertNormals> makeUnorientedNormals( const PointCloud& pointCloud,
    const Buffer<VertId> & closeVerts, int numNei, const ProgressCallback & progress = {} );

/// \brief Select consistent orientation of given normals to make directions of close points consistent;
/// \param radius of neighborhood to consider
/// \return false if progress returned false
/// \ingroup PointCloudGroup
MRMESH_API bool orientNormals( const PointCloud& pointCloud, VertNormals& normals, float radius,
    const ProgressCallback & progress = {} );

/// \brief Select consistent orientation of given normals to make directions of close points consistent;
/// \param closeVert a buffer where for every valid point #i its neighbours are stored at indices [i*numNei; (i+1)*numNei)
/// \return false if progress returned false
/// \ingroup PointCloudGroup
MRMESH_API bool orientNormals( const PointCloud& pointCloud, VertNormals& normals, const Buffer<VertId> & closeVerts, int numNei,
    const ProgressCallback & progress = {} );

/// \brief Makes normals for valid points of given point cloud; directions of close points are selected to be consistent;
/// \param numNei the number of closest points to consider around each point
/// \return nullopt if progress returned false
/// \ingroup PointCloudGroup
[[nodiscard]] MRMESH_API std::optional<VertNormals> makeOrientedNormals( const PointCloud& pointCloud,
    int numNei, const ProgressCallback & progress = {} );

} //namespace MR
