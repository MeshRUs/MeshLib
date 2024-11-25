#pragma once

#include "MRMeshFwd.h"
#include "MRMeshPart.h"
#include "MRId.h"
MR_EXTERN_C_BEGIN

typedef struct MRMultipleEdge
{
    MRVertId v0;
    MRVertId v1;
} MRMultipleEdge;

/// returns all faces that complicate one of mesh holes;
/// hole is complicated if it passes via one vertex more than once;
/// deleting such faces simplifies the holes and makes them easier to fill
MRMESHC_API MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh );
/// finds faces having aspect ratio >= criticalAspectRatio
MRMESHC_API MRFaceBitSet* mrFindDegenerateFaces( const MRMeshPart* mp, float criticalAspectRatio, MRProgressCallback cb, MRString** errorString );
/// finds edges having length <= criticalLength
MRMESHC_API MRUndirectedEdgeBitSet* mrFindShortEdges( const MRMeshPart* mp, float criticalLength, MRProgressCallback cb, MRString** errorString );

/// resolves given multiple edges, but splitting all but one edge in each group
MRMESHC_API void fixMultipleEdges( MRMesh* mesh, const MRMultipleEdge* multipleEdges, size_t multipleEdgesNum );
/// finds and resolves multiple edges
MRMESHC_API void findAndFixMultipleEdges( MRMesh* mesh );

MR_EXTERN_C_END
