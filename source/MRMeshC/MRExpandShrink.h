#pragma once
#include "MRMeshFwd.h"
#include "MRId.h"

MR_EXTERN_C_BEGIN
/// adds to the region all faces within given number of hops (stars) from the initial region boundary
MRMESHC_API void mrExpandFaceRegion( const MRMeshTopology* top, MRFaceBitSet* region, int hops );
/// returns the region of all faces within given number of hops (stars) from the initial face
MRMESHC_API MRFaceBitSet* mrExpandFaceRegionFromFace( const MRMeshTopology* top, MRFaceId face, int hops );
// adds to the region all vertices within given number of hops (stars) from the initial region boundary
MRMESHC_API void mrExpandVertRegion( const MRMeshTopology* top, MRVertBitSet* region, int hops );
/// returns the region of all vertices within given number of hops (stars) from the initial vertex
MRMESHC_API MRVertBitSet* mrExpandVertRegionFromVert( const MRMeshTopology* top, MRVertId vert, int hops );
/// removes from the region all faces within given number of hops (stars) from the initial region boundary
MRMESHC_API void mrShrinkFaceRegion( const MRMeshTopology* top, MRFaceBitSet* region, int hops );
/// removes from the region all vertices within given number of hops (stars) from the initial region boundary
MRMESHC_API void mrShrinkVertRegion( const MRMeshTopology* top, MRVertBitSet* region, int hops );

MR_EXTERN_C_END
