#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRMeshTopology.h"

MR_EXTERN_C_BEGIN

/// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
/// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
MRMESHC_API MREdgeLoop* mrTrackRightBoundaryLoop( const MRMeshTopology* topology, MREdgeId e0, const MRFaceBitSet* region );

typedef struct MREdgeLoops MREdgeLoops;

MRMESHC_API const MREdgeLoop mrEdgeLoopsGet( const MREdgeLoops* loops, size_t index );

MRMESHC_API size_t mrEdgeLoopsSize( const MREdgeLoops* loops );

MRMESHC_API void mrEdgeLoopsFree( MREdgeLoops* loops );

/// returns all region boundary loops;
/// every loop has region faces on the right, and not-region faces or holes on the left
MRMESHC_API MREdgeLoops* mrFindRightBoundary( const MRMeshTopology* topology, const MRFaceBitSet* region );
/// composes the set of all faces incident to given vertices
MRMESHC_API MRFaceBitSet* mrGetIncidentFacesFromVerts( const MRMeshTopology* topology, const MRVertBitSet* region );
/// composes the set of all faces incident to given edges
MRMESHC_API MRFaceBitSet* mrGetIncidentFacesFromEdges( const MRMeshTopology* topology, const MRUndirectedEdgeBitSet* region );

/// composes the set of all vertices incident to given faces
MRMESHC_API MRVertBitSet* mrGetIncidentVertsFromFaces( const MRMeshTopology* topology, const MRFaceBitSet* faces );

/// composes the set of all vertices incident to given edges
MRMESHC_API MRVertBitSet* mrGetIncidentVertsFromEdges( const MRMeshTopology* topology, const MRUndirectedEdgeBitSet* edges );

/// composes the set of all vertices not on the boundary of a hole and with all their adjacent faces in given set
MRMESHC_API MRVertBitSet* mrGetInnerVertsFromFaces( const MRMeshTopology* topology, const MRFaceBitSet* region );

/// composes the set of all vertices with all their edges in given set
MRMESHC_API MRVertBitSet* mrGetInnerVertsFromEdges( const MRMeshTopology* topology, const MRUndirectedEdgeBitSet* edges );

/// composes the set of all faces with all their vertices in given set
MRMESHC_API MRFaceBitSet* mrGetInnerFacesFromVerts( const MRMeshTopology* topology, const MRVertBitSet* verts );

/// composes the set of all undirected edges, having a face from given set from one of two sides
MRMESHC_API MRUndirectedEdgeBitSet* mrGetIncidentEdgesFromFaces( const MRMeshTopology* topology, const MRFaceBitSet* faces );

/// composes the set of all undirected edges, having at least one common vertex with an edge from given set
MRMESHC_API MRUndirectedEdgeBitSet* mrGetIncidentEdgesFromEdges( const MRMeshTopology* topology, const MRUndirectedEdgeBitSet* edges );

/// composes the set of all edges with all their vertices in given set
MRMESHC_API MRUndirectedEdgeBitSet* mrGetInnerEdgesFromVerts( const MRMeshTopology* topology, const MRVertBitSet* verts );

/// composes the set of all edges having both left and right in given region
MRMESHC_API MRUndirectedEdgeBitSet* mrGetInnerEdgesFromFaces( const MRMeshTopology* topology, const MRFaceBitSet* region );




MR_EXTERN_C_END
