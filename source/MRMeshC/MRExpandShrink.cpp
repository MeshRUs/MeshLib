#include "MRExpandShrink.h"
#include "detail/TypeCast.h"

#include "MRMesh/MRId.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRExpandShrink.h"

using namespace MR;

REGISTER_AUTO_CAST( FaceId )
REGISTER_AUTO_CAST( VertId )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST( MeshTopology )

void mrExpandFaceRegion( const MRMeshTopology* top_, MRFaceBitSet* region_, int hops )
{
    ARG( top ); ARG( region );
    expand( top, region, hops );
}

MRFaceBitSet* mrExpandFaceRegionFromFace( const MRMeshTopology* top_, MRFaceId face_, int hops )
{
    ARG( top ); ARG_VAL( face );
    RETURN_NEW( expand( top, face, hops ) );
}

void mrExpandVertRegion( const MRMeshTopology* top_, MRVertBitSet* region_, int hops )
{
    ARG( top ); ARG( region );
    expand( top, region, hops );
}

MRVertBitSet* mrExpandVertRegionFromVert( const MRMeshTopology* top_, MRVertId vert_, int hops )
{
    ARG( top ); ARG_VAL( vert );
    RETURN_NEW( expand( top, vert, hops ) );
}

void mrShrinkFaceRegion( const MRMeshTopology* top_, MRFaceBitSet* region_, int hops )
{
    ARG( top ); ARG( region );
    shrink( top, region, hops );
}

void mrShrinkVertRegion( const MRMeshTopology* top_, MRVertBitSet* region_, int hops )
{
    ARG( top ); ARG( region );
    shrink( top, region, hops );
}


