#pragma once

#include "MRId.h"
#include "MREdgeMetric.h"
#include "MRProgressCallback.h"
#include <cfloat>
#include <vector>

namespace MR
{

/// \ingroup SurfacePathGroup
/// \{

/// returns true if every next edge starts where previous edge ends
[[nodiscard]] MRMESH_API bool isEdgePath( const MeshTopology & topology, const std::vector<EdgeId> & edges );

/// returns true if every next edge starts where previous edge ends, and start vertex coincides with finish vertex
[[nodiscard]] MRMESH_API bool isEdgeLoop( const MeshTopology & topology, const std::vector<EdgeId> & edges );

/// given a number of edge loops, splits every loop that passes via a vertex more than once on smaller loops without self-intersections
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> splitOnSimpleLoops( const MeshTopology & topology, std::vector<EdgeLoop> && loops );

/// reverses the order of edges and flips each edge orientation, thus
/// making the opposite directed edge path
MRMESH_API void reverse( EdgePath & path );
/// reverse every path in the vector
MRMESH_API void reverse( std::vector<EdgePath> & paths );

/// computes summed metric of all edges in the path
[[nodiscard]] MRMESH_API double calcPathMetric( const EdgePath & path, EdgeMetric metric );
[[nodiscard]] inline double calcPathLength( const EdgePath & path, const Mesh & mesh ) { return calcPathMetric( path, edgeLengthMetric( mesh ) ); }

/// returns the vector with the magnitude equal to the area surrounded by the loop (if the loop is planar),
/// and directed to see the loop in ccw order from the vector tip
[[nodiscard]] MRMESH_API Vector3d calcOrientedArea( const EdgeLoop & loop, const Mesh & mesh );

/// sorts given paths in ascending order of their metrics
MRMESH_API void sortPathsByMetric( std::vector<EdgePath> & paths, EdgeMetric metric );
inline void sortPathsByLength( std::vector<EdgePath> & paths, const Mesh & mesh ) { sortPathsByMetric( paths, edgeLengthMetric( mesh ) ); }

/// adds all faces incident to loop vertices and located to the left from the loop to given FaceBitSet
MRMESH_API void addLeftBand( const MeshTopology & topology, const EdgeLoop & loop, FaceBitSet & addHere );

/// a vertex with associated penalty metric
/// to designate a possible start or end of edge path
struct TerminalVertex
{
    VertId v;
    float metric = 0;
};

/// finds the shortest path in euclidean metric from start to finish vertices using Dijkstra algorithm;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPath( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );
/// finds the shortest path in euclidean metric from start to finish vertices using bidirectional modification of Dijkstra algorithm,
/// constructing the path simultaneously from both start and finish, which is faster for long paths;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPathBiDir( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );
/// finds the path from a vertex in start-triangle to a vertex in finish-triangle,
/// so that the length start-first_vertex-...-last_vertex-finish is shortest in euclidean metric;
/// using bidirectional modification of Dijkstra algorithm, constructing the path simultaneously from both start and finish, which is faster for long paths;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPathBiDir( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & finish,
    VertId * outPathStart = nullptr,  // if the path is found then its start vertex will be written here (even if start vertex is the same as finish vertex and the path is empty)
    VertId * outPathFinish = nullptr, // if the path is found then its finish vertex will be written here (even if start vertex is the same as finish vertex and the path is empty)
    float maxPathLen = FLT_MAX );
/// finds the shortest path in euclidean metric from start to finish vertices using A* modification of Dijkstra algorithm,
/// which is faster for near linear path;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPathAStar( const Mesh & mesh, VertId start, VertId finish, float maxPathLen = FLT_MAX );
/// finds the path from a vertex in start-triangle to a vertex in finish-triangle,
/// so that the length start-first_vertex-...-last_vertex-finish is shortest in euclidean metric;
/// using A* modification of Dijkstra algorithm, which is faster for near linear path;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPathAStar( const Mesh & mesh,
    const MeshTriPoint & start, const MeshTriPoint & finish,
    VertId * outPathStart = nullptr,  // if the path is found then its start vertex will be written here (even if start vertex is the same as finish vertex and the path is empty)
    VertId * outPathFinish = nullptr, // if the path is found then its finish vertex will be written here (even if start vertex is the same as finish vertex and the path is empty)
    float maxPathLen = FLT_MAX );

/// builds shortest path in euclidean metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildShortestPath( const Mesh& mesh, VertId start, const VertBitSet& finish, float maxPathLen = FLT_MAX );

/// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPath( const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric = FLT_MAX );
/// finds the smallest metric path from start vertex to finish vertex,
/// using bidirectional modification of Dijkstra algorithm, constructing the path simultaneously from both start and finish, which is faster for long paths;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPathBiDir( const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric = FLT_MAX );
/// finds the smallest metric path from one of start vertices to one of the finish vertices,
/// using bidirectional modification of Dijkstra algorithm, constructing the path simultaneously from both starts and finishes, which is faster for long paths;
/// if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPathBiDir( const MeshTopology & topology, const EdgeMetric & metric,
    const TerminalVertex * starts, int numStarts,
    const TerminalVertex * finishes, int numFinishes,
    VertId * outPathStart = nullptr,  // if the path is found then its start vertex will be written here (even if start vertex is the same as finish vertex and the path is empty)
    VertId * outPathFinish = nullptr, // if the path is found then its finish vertex will be written here (even if start vertex is the same as finish vertex and the path is empty)
    float maxPathMetric = FLT_MAX );

/// builds shortest path in given metric from start to finish vertices; if no path can be found then empty path is returned
[[nodiscard]] MRMESH_API EdgePath buildSmallestMetricPath( const MeshTopology& topology, const EdgeMetric& metric,
    VertId start, const VertBitSet& finish, float maxPathMetric = FLT_MAX );

/// returns all vertices from given region ordered in each connected component in breadth-first way
[[nodiscard]] MRMESH_API std::vector<VertId> getVertexOrdering( const MeshTopology & topology, VertBitSet region );

/// finds all closed loops from given edges and removes them from edges
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, EdgeBitSet & edges );
[[nodiscard]] MRMESH_API std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, const std::vector<EdgeId> & inEdges,
    EdgeBitSet * outNotLoopEdges = nullptr );
[[nodiscard]] MRMESH_API EdgeLoop extractLongestClosedLoop( const Mesh & mesh, const std::vector<EdgeId> & inEdges );

/// expands the region (of faces or vertices) on given metric value. returns false if callback also returns false
MRMESH_API bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// shrinks the region (of faces or vertices) on given metric value. returns false if callback also returns false
MRMESH_API bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// expands the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
MRMESH_API bool dilateRegion( const Mesh& mesh, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegion( const Mesh& mesh, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool dilateRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// shrinks the region (of faces or vertices) on given value (in meters). returns false if callback also returns false
MRMESH_API bool erodeRegion( const Mesh& mesh, FaceBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegion( const Mesh& mesh, VertBitSet& region, float dilation, ProgressCallback callback = {} );
MRMESH_API bool erodeRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback = {} );

/// finds all intersection points between given path and plane, adds them in outIntersections and returns their number
MRMESH_API int getPathPlaneIntersections( const Mesh & mesh, const EdgePath & path, const Plane3f & plane,
    std::vector<MeshEdgePoint> * outIntersections = nullptr );

/// finds all path edges located in given plane with given tolerance, adds them in outInPlaneEdges and returns their number
MRMESH_API int getPathEdgesInPlane( const Mesh & mesh, const EdgePath & path, const Plane3f & plane, float tolerance = 0.0f,
    std::vector<EdgeId> * outInPlaneEdges = nullptr );

/// \}

} // namespace MR
