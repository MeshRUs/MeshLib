#include "MRMeshToPointCloud.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRMeshToPointCloud.h"

using namespace MR;

REGISTER_AUTO_CAST( VertBitSet );
REGISTER_AUTO_CAST( Mesh );
REGISTER_AUTO_CAST( PointCloud );

MRPointCloud* mrMeshToPointCloud( const MRMesh* mesh_, bool saveNormals, const MRVertBitSet* verts_ )
{
    ARG( mesh ); ARG_PTR( verts );
    RETURN_NEW( MR::meshToPointCloud( mesh, saveNormals, verts ) );
}
