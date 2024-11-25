#pragma once
#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

///  Mesh to PointCloud
MRMESHC_API MRPointCloud* mrMeshToPointCloud( const MRMesh* mesh, bool saveNormals, const MRVertBitSet* verts );

MR_EXTERN_C_END