#pragma once

#include "MRId.h"
#include "MRVector.h"
#include "MRBitSet.h"
#include "MRPartMapping.h"
#include "MRMeshTriPoint.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include <fstream>

namespace MR
{

/// Mesh Topology
/// \ingroup MeshGroup
class MeshTopology
{
public:
    /// creates an edge not associated with any vertex or face
    [[nodiscard]] MRMESH_API EdgeId makeEdge();

    /// checks whether the edge is disconnected from all other edges and disassociated from all vertices and faces (as if after makeEdge)
    [[nodiscard]] MRMESH_API bool isLoneEdge( EdgeId a ) const;

    /// returns last not lone edge id, or invalid id if no such edge exists
    [[nodiscard]] MRMESH_API EdgeId lastNotLoneEdge() const;

    /// remove all lone edges from given set
    MRMESH_API void excludeLoneEdges( UndirectedEdgeBitSet & edges ) const;

    /// allows to copy edges data when necessary
    [[nodiscard]] const auto& edges() const { return edges_; }

    /// returns the number of half-edge records including lone ones
    [[nodiscard]] size_t edgeSize() const { return edges_.size(); }

    /// returns the number of allocated edge records
    [[nodiscard]] size_t edgeCapacity() const { return edges_.capacity(); }

    /// returns the number of undirected edges (pairs of half-edges) including lone ones
    [[nodiscard]] size_t undirectedEdgeSize() const { return edges_.size() >> 1; }

    /// returns the number of allocated undirected edges (pairs of half-edges)
    [[nodiscard]] size_t undirectedEdgeCapacity() const { return edges_.capacity() >> 1; }

    /// computes the number of not-lone (valid) undirected edges
    [[nodiscard]] MRMESH_API size_t computeNotLoneUndirectedEdges() const;

    /// finds and returns all not-lone (valid) undirected edges
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet findNotLoneUndirectedEdges() const;

    /// sets the capacity of half-edges vector
    void edgeReserve( size_t newCapacity ) { edges_.reserve( newCapacity ); }

    /// returns true if given edge is within valid range and not-lone
    [[nodiscard]] bool hasEdge( EdgeId e ) const { assert( e.valid() ); return e < (int)edgeSize() && !isLoneEdge( e ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

    /// requests the removal of unused capacity
    MRMESH_API void shrinkToFit();


    /// given two half edges do either of two:
    /// 1) if a and b were from distinct rings, puts them in one ring;
    /// 2) if a and b were from the same ring, puts them in separate rings;
    /// the cut in rings in both cases is made after a and b
    MRMESH_API void splice( EdgeId a, EdgeId b );


    /// next (counter clock wise) half-edge in the origin ring
    [[nodiscard]] EdgeId next( EdgeId he ) const { assert(he.valid()); return edges_[he].next; }

    /// previous (clock wise) half-edge in the origin ring
    [[nodiscard]] EdgeId prev( EdgeId he ) const { assert(he.valid()); return edges_[he].prev; }

    /// returns origin vertex of half-edge
    [[nodiscard]] VertId org( EdgeId he ) const { assert(he.valid()); return edges_[he].org; }

    /// returns destination vertex of half-edge
    [[nodiscard]] VertId dest( EdgeId he ) const { assert(he.valid()); return edges_[he.sym()].org; }

    /// returns left face of half-edge
    [[nodiscard]] FaceId left( EdgeId he ) const { assert(he.valid()); return edges_[he].left; }

    /// returns right face of half-edge
    [[nodiscard]] FaceId right( EdgeId he ) const { assert(he.valid()); return edges_[he.sym()].left; }


    /// sets new origin to the full origin ring including this edge;
    /// edgePerVertex_ table is updated accordingly
    MRMESH_API void setOrg( EdgeId a, VertId v );

    /// sets new left face to the full left ring including this edge;
    /// edgePerFace_ table is updated accordingly
    MRMESH_API void setLeft( EdgeId a, FaceId f );
    

    /// returns true if a and b are both from the same origin ring
    [[nodiscard]] MRMESH_API bool fromSameOriginRing( EdgeId a, EdgeId b ) const;

    /// returns true if a and b are both from the same left face ring
    [[nodiscard]] MRMESH_API bool fromSameLeftRing( EdgeId a, EdgeId b ) const;


    /// returns the number of edges around the origin vertex, returns 1 for lone edges
    [[nodiscard]] MRMESH_API int getOrgDegree( EdgeId a ) const;

    /// returns the number of edges around the given vertex
    [[nodiscard]] int getVertDegree( VertId v ) const { return getOrgDegree( edgeWithOrg( v ) ); }

    /// returns the number of edges around the left face: 3 for triangular faces, ...
    [[nodiscard]] MRMESH_API int getLeftDegree( EdgeId a ) const;

    /// returns the number of edges around the given face: 3 for triangular faces, ...
    [[nodiscard]] int getFaceDegree( FaceId f ) const { return getLeftDegree( edgeWithLeft( f ) ); }

    /// returns true if the cell to the left of a is triangular
    [[nodiscard]] MRMESH_API bool isLeftTri( EdgeId a ) const;

    /// gets 3 vertices of given triangular face;
    /// the vertices are returned in counter-clockwise order if look from mesh outside
    void getTriVerts( FaceId f, VertId & v0, VertId & v1, VertId & v2 ) const { getLeftTriVerts( edgeWithLeft( f ), v0, v1, v2 ); }
    void getTriVerts( FaceId f, VertId (&v)[3] ) const { getLeftTriVerts( edgeWithLeft( f ), v ); }
    void getTriVerts( FaceId f, ThreeVertIds & v ) const { getLeftTriVerts( edgeWithLeft( f ), v ); }
    [[nodiscard]] ThreeVertIds getTriVerts( FaceId f ) const { return getLeftTriVerts( edgeWithLeft( f ) ); }

    /// returns three vertex ids for valid triangles, invalid triangles are skipped
    [[nodiscard]] MRMESH_API std::vector<ThreeVertIds> getAllTriVerts() const;

    /// returns three vertex ids for valid triangles (which can be accessed by FaceId),
    /// vertex ids for invalid triangles are undefined, and shall not be read
    [[nodiscard]] MRMESH_API Triangulation getTriangulation() const;

    /// gets 3 vertices of the left face ( face-id may not exist, but the shape must be triangular)
    /// the vertices are returned in counter-clockwise order if look from mesh outside
    MRMESH_API void getLeftTriVerts( EdgeId a, VertId & v0, VertId & v1, VertId & v2 ) const;
               void getLeftTriVerts( EdgeId a, VertId (&v)[3] ) const { getLeftTriVerts( a, v[0], v[1], v[2] ); }
               void getLeftTriVerts( EdgeId a, ThreeVertIds & v ) const { getLeftTriVerts( a, v[0], v[1], v[2] ); }
    [[nodiscard]] ThreeVertIds getLeftTriVerts( EdgeId a ) const { ThreeVertIds v; getLeftTriVerts( a, v[0], v[1], v[2] ); return v; }

    /// if given point is
    /// 1) in vertex, then invokes callback once with it;
    /// 2) on edge, then invokes callback twice with every vertex of the edge;
    /// 3) inside triangle, then invokes callback trice with every vertex of the triangle
    template <typename T>
    void forEachVertex( const MeshTriPoint & p, T && callback ) const;

    /// gets 3 edges of given triangular face, oriented to have it on the left;
    /// the edges are returned in counter-clockwise order if look from mesh outside
    MRMESH_API void getTriEdges( FaceId f, EdgeId & e0, EdgeId & e1, EdgeId & e2 ) const;
               void getTriEdges( FaceId f, EdgeId (&e)[3] ) const { getTriEdges( f, e[0], e[1], e[2] ); }

    /// returns true if the cell to the left of a is quadrangular
    [[nodiscard]] MRMESH_API bool isLeftQuad( EdgeId a ) const;

    /// for all valid vertices this vector contains an edge with the origin there
    [[nodiscard]] const Vector<EdgeId, VertId> & edgePerVertex() const { return edgePerVertex_; }

    /// returns valid edge if given vertex is present in the mesh
    [[nodiscard]] EdgeId edgeWithOrg( VertId a ) const { assert( a.valid() ); return a < int(edgePerVertex_.size()) ? edgePerVertex_[a] : EdgeId(); }

    /// returns true if given vertex is present in the mesh
    [[nodiscard]] bool hasVert( VertId a ) const { assert( updateValids_ ); return validVerts_.test( a ); }

    /// returns the number of valid vertices
    [[nodiscard]] int numValidVerts() const { assert( updateValids_ ); return numValidVerts_; }

    /// returns last valid vertex id, or invalid id if no single valid vertex exists
    [[nodiscard]] MRMESH_API VertId lastValidVert() const;

    /// creates new vert-id not associated with any edge yet
    [[nodiscard]] VertId addVertId() { edgePerVertex_.emplace_back(); if ( updateValids_ ) { validVerts_.push_back( false ); } return edgePerVertex_.backId(); }

    /// explicitly increases the size of vertices vector
    MRMESH_API void vertResize( size_t newSize );

    /// explicitly increases the size of vertices vector, doubling the current capacity if it was not enough
    MRMESH_API void vertResizeWithReserve( size_t newSize );

    /// sets the capacity of vertices vector
    void vertReserve( size_t newCapacity ) { edgePerVertex_.reserve( newCapacity ); if ( updateValids_ ) { validVerts_.reserve( newCapacity ); } }

    /// returns the number of vertex records including invalid ones
    [[nodiscard]] size_t vertSize() const { return edgePerVertex_.size(); }

    /// returns the number of allocated vert records
    [[nodiscard]] size_t vertCapacity() const { return edgePerVertex_.capacity(); }

    /// returns cached set of all valid vertices
    [[nodiscard]] const VertBitSet & getValidVerts() const { assert( updateValids_ ); return validVerts_; }

    /// sets in (vs) all valid vertices that were not selected before the call, and resets other bits
    void flip( VertBitSet & vs ) const { vs = getValidVerts() - vs; }

    /// if region pointer is not null then converts it in reference, otherwise returns all valid vertices in the mesh
    [[nodiscard]] const VertBitSet & getVertIds( const VertBitSet * region ) const
    {
        assert( region || updateValids_ ); // region shall be either given on input or maintained in validVerts_
        assert( !updateValids_ || !region || ( *region - validVerts_ ).none() ); // if region is given and all valid vertices are known, then region must be a subset of them
        return region ? *region : validVerts_;
    }


    /// for all valid faces this vector contains an edge with that face at left
    [[nodiscard]] const Vector<EdgeId, FaceId> & edgePerFace() const { return edgePerFace_; }

    /// returns valid edge if given vertex is present in the mesh
    [[nodiscard]] EdgeId edgeWithLeft( FaceId a ) const { assert( a.valid() ); return a < int(edgePerFace_.size()) ? edgePerFace_[a] : EdgeId(); }

    /// returns true if given face is present in the mesh
    [[nodiscard]] bool hasFace( FaceId a ) const { assert( updateValids_ ); return validFaces_.test( a ); }

    /// if two valid faces share the same edge then it is found and returned
    [[nodiscard]] MRMESH_API EdgeId sharedEdge( FaceId l, FaceId r ) const;

    /// if two valid edges share the same vertex then it is found and returned as Edge with this vertex in origin
    [[nodiscard]] MRMESH_API EdgeId sharedVertInOrg( EdgeId a, EdgeId b ) const;

    /// if two valid faces share the same vertex then it is found and returned as Edge with this vertex in origin
    [[nodiscard]] MRMESH_API EdgeId sharedVertInOrg( FaceId l, FaceId r ) const;

    /// if two valid edges belong to same valid face then it is found and returned
    [[nodiscard]] MRMESH_API FaceId sharedFace( EdgeId a, EdgeId b ) const;

    /// returns the number of valid faces
    [[nodiscard]] int numValidFaces() const { assert( updateValids_ ); return numValidFaces_; }

    /// returns last valid face id, or invalid id if no single valid face exists
    [[nodiscard]] MRMESH_API FaceId lastValidFace() const;

    /// creates new face-id not associated with any edge yet
    [[nodiscard]] FaceId addFaceId() { edgePerFace_.emplace_back(); if ( updateValids_ ) { validFaces_.push_back( false ); } return edgePerFace_.backId(); }

    /// deletes the face, also deletes its edges and vertices if they were not shared by other faces ant not in \param keepFaces
    MRMESH_API void deleteFace( FaceId f, const UndirectedEdgeBitSet * keepEdges = nullptr );

    /// deletes multiple given faces by calling \ref deleteFace for each
    MRMESH_API void deleteFaces( const FaceBitSet & fs, const UndirectedEdgeBitSet * keepEdges = nullptr );

    /// explicitly increases the size of faces vector
    MRMESH_API void faceResize( size_t newSize );

    /// explicitly increases the size of faces vector, doubling the current capacity if it was not enough
    MRMESH_API void faceResizeWithReserve( size_t newSize );

    /// sets the capacity of faces vector
    void faceReserve( size_t newCapacity ) { edgePerFace_.reserve( newCapacity ); if ( updateValids_ ) { validFaces_.reserve( newCapacity ); } }

    /// returns the number of face records including invalid ones
    [[nodiscard]] size_t faceSize() const { return edgePerFace_.size(); }

    /// returns the number of allocated face records
    [[nodiscard]] size_t faceCapacity() const { return edgePerFace_.capacity(); }

    /// returns cached set of all valid faces
    [[nodiscard]] const FaceBitSet & getValidFaces() const { assert( updateValids_ ); return validFaces_; }

    /// sets in (fs) all valid faces that were not selected before the call, and resets other bits
    void flip( FaceBitSet & fs ) const { fs = getValidFaces() - fs; }

    /// if region pointer is not null then converts it in reference, otherwise returns all valid faces in the mesh
    [[nodiscard]] const FaceBitSet & getFaceIds( const FaceBitSet * region ) const
    { 
        assert( region || updateValids_ ); // region shall be either given on input or maintained in validFaces_
        assert( !updateValids_ || !region || ( *region - validFaces_ ).none() ); // if region is given and all valid faces are known, then region must be a subset of them
        return region ? *region : validFaces_;
    }

    /// returns the first boundary edge (for given region or for whole mesh if region is nullptr) in counter-clockwise order starting from given edge with the same left;
    /// returns invalid edge if no boundary edge is found
    [[nodiscard]] MRMESH_API EdgeId bdEdgeSameLeft( EdgeId e, const FaceBitSet * region = nullptr ) const;

    /// returns true if edge's left is on (region) boundary
    [[nodiscard]] bool isBdVertexInLeft( EdgeId e, const FaceBitSet * region = nullptr ) const { return bdEdgeSameLeft( e, region ).valid(); }

    /// returns a boundary edge with given left face considering boundary of given region (or for whole mesh if region is nullptr);
    /// returns invalid edge if no boundary edge is found
    [[nodiscard]] EdgeId bdEdgeWithLeft( FaceId f, const FaceBitSet * region = nullptr ) const { return bdEdgeSameLeft( edgeWithLeft( f ), region ); }

    /// returns true if given face is on (region) boundary
    [[nodiscard]] bool isBdFace( FaceId f, const FaceBitSet * region = nullptr ) const { return isBdVertexInLeft( edgeWithLeft( f ), region ); }


    /// return true if left face of given edge belongs to region (or just have valid id if region is nullptr)
    [[nodiscard]] bool isLeftInRegion( EdgeId e, const FaceBitSet * region = nullptr ) const { return contains( region, left( e ) ); }

    /// return true if given edge is inner for given region (or for whole mesh if region is nullptr)
    [[nodiscard]] bool isInnerEdge( EdgeId e, const FaceBitSet * region = nullptr ) const { return isLeftInRegion( e, region ) && isLeftInRegion( e.sym(), region ); }

    /// return true if given edge is boundary for given region (or for whole mesh if region is nullptr)
    [[nodiscard]] bool isBdEdge( EdgeId e, const FaceBitSet * region = nullptr ) const { return isLeftInRegion( e, region ) != isLeftInRegion( e.sym(), region ); }

    /// returns the first boundary edge (for given region or for whole mesh if region is nullptr) in counter-clockwise order starting from given edge with the same origin;
    /// returns invalid edge if no boundary edge is found
    [[nodiscard]] MRMESH_API EdgeId bdEdgeSameOrigin( EdgeId e, const FaceBitSet * region = nullptr ) const;

    /// returns true if edge's origin is on (region) boundary
    [[nodiscard]] bool isBdVertexInOrg( EdgeId e, const FaceBitSet * region = nullptr ) const { return bdEdgeSameOrigin( e, region ).valid(); }

    /// returns a boundary edge with given vertex in origin considering boundary of given region (or for whole mesh if region is nullptr);
    /// returns invalid edge if no boundary edge is found
    [[nodiscard]] EdgeId bdEdgeWithOrigin( VertId v, const FaceBitSet * region = nullptr ) const { return bdEdgeSameOrigin( edgeWithOrg( v ), region ); }

    /// returns true if given vertex is on (region) boundary
    [[nodiscard]] bool isBdVertex( VertId v, const FaceBitSet * region = nullptr ) const { return isBdVertexInOrg( edgeWithOrg( v ), region ); }

    /// returns true if one of incident faces of given vertex pertain to given region (or any such face exists if region is nullptr)
    [[nodiscard]] MRMESH_API bool isInnerOrBdVertex( VertId v, const FaceBitSet * region = nullptr ) const;

    /// returns true if left face of given edge belongs to given region (if provided) and right face either does not exist or does not belong
    [[nodiscard]] bool isLeftBdEdge( EdgeId e, const FaceBitSet * region = nullptr ) const { return region ? ( isLeftInRegion( e, region ) && !isLeftInRegion( e.sym(), region ) ) : !right( e ); }

    /// return true if given edge is inner or boundary for given region (or for whole mesh if region is nullptr), returns false for lone edges
    [[nodiscard]] bool isInnerOrBdEdge( EdgeId e, const FaceBitSet * region = nullptr ) const { return isLeftInRegion( e, region ) || isLeftInRegion( e.sym(), region ); }

    /// given a (region) boundary edge with no right face in given region, returns next boundary edge for the same region: dest(e)==org(res)
    [[nodiscard]] MRMESH_API EdgeId nextLeftBd( EdgeId e, const FaceBitSet * region = nullptr ) const;

    /// given a (region) boundary edge with no right face in given region, returns previous boundary edge for the same region; dest(res)==org(e)
    [[nodiscard]] MRMESH_API EdgeId prevLeftBd( EdgeId e, const FaceBitSet * region = nullptr ) const;


    /// finds and returns edge from o to d in the mesh; returns invalid edge otherwise
    [[nodiscard]] MRMESH_API EdgeId findEdge( VertId o, VertId d ) const;

    /// returns true if the mesh does not have any holes
    [[nodiscard]] MRMESH_API bool isClosed() const;

    /// returns true if the mesh region does not have any neighboring holes
    [[nodiscard]] MRMESH_API bool isClosed( const FaceBitSet * region ) const;

    /// returns one edge with no valid left face for every boundary in the mesh
    [[nodiscard]] MRMESH_API std::vector<EdgeId> findHoleRepresentiveEdges() const;

    /// returns the number of hole loops in the mesh;
    /// \param holeRepresentativeEdges optional output of the smallest edge id with no valid left face in every hole
    [[nodiscard]] MRMESH_API int findNumHoles( EdgeBitSet * holeRepresentativeEdges = nullptr ) const;

    /// returns full edge-loop of left face from (e) starting from (e) itself
    [[nodiscard]] MRMESH_API EdgeLoop getLeftRing( EdgeId e ) const;

    /// returns full edge-loops of left faces from every edge in (es);
    /// each edge-loop will be returned only once even if some faces are represented by more than one edge in (es)
    [[nodiscard]] MRMESH_API std::vector<EdgeLoop> getLeftRings( const std::vector<EdgeId> & es ) const;

    /// returns all boundary edges, where each edge does not have valid left face
    [[nodiscard]] MRMESH_API EdgeBitSet findBoundaryEdges() const;

    /// returns all boundary faces, having at least one boundary edge
    [[nodiscard]] MRMESH_API FaceBitSet findBoundaryFaces() const;

    /// returns all boundary vertices, incident to at least one boundary edge
    [[nodiscard]] MRMESH_API VertBitSet findBoundaryVerts() const;


    /// returns all vertices incident to path edges
    [[nodiscard]] MRMESH_API VertBitSet getPathVertices( const EdgePath & path ) const;

    /// returns all valid left faces of path edges
    [[nodiscard]] MRMESH_API FaceBitSet getPathLeftFaces( const EdgePath & path ) const;

    /// returns all valid right faces of path edges
    [[nodiscard]] MRMESH_API FaceBitSet getPathRightFaces( const EdgePath & path ) const;


    /// given the edge with left and right triangular faces, which form together a quadrangle,
    /// rotates the edge counter-clockwise inside the quadrangle
    MRMESH_API void flipEdge( EdgeId e );

    /// tests all edges e having valid left and right faces and org(e0) == dest(next(e));
    /// if the test has passed, then flips the edge so increasing the degree of org(e0)
    template<typename T>
    void flipEdgesIn( EdgeId e0, T && flipNeeded );

    /// tests all edges e having valid left and right faces and v == dest(next(e));
    /// if the test has passed, then flips the edge so increasing the degree of vertex v
    template<typename T>
    void flipEdgesIn( VertId v, T && flipNeeded ) { flipEdgesIn( edgeWithOrg( v ), std::forward<T>( flipNeeded ) ); }

    /// tests all edges e having valid left and right faces and org(e0) == org(e);
    /// if the test has passed, then flips the edge so decreasing the degree of org(e0)
    template<typename T>
    void flipEdgesOut( EdgeId e0, T && flipNeeded );

    /// tests all edges e having valid left and right faces and v == org(e);
    /// if the test has passed, then flips the edge so decreasing the degree of vertex v
    template<typename T>
    void flipEdgesOut( VertId v, T && flipNeeded ) { flipEdgesOut( edgeWithOrg( v ), std::forward<T>( flipNeeded ) ); }

    /// split given edge on two parts:
    /// dest(returned-edge) = org(e) - newly created vertex,
    /// org(returned-edge) = org(e-before-split),
    /// dest(e) = dest(e-before-split)
    /// \details left and right faces of given edge if valid are also subdivided on two parts each;
    /// if left or right faces of the original edge were in the region, then include new parts of these faces in the region
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    MRMESH_API EdgeId splitEdge( EdgeId e, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr );

    /// split given triangle on three triangles, introducing new vertex (which is returned) inside original triangle and connecting it to its vertices
    /// \details if region is given, then it must include (f) and new faces will be added there as well
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    MRMESH_API VertId splitFace( FaceId f, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr );

    /// flip orientation (normals) of
    /// * all mesh elements if \param fullComponents is nullptr, or
    /// * given mesh components in \param fullComponents.
    /// The behavior is undefined if fullComponents is given but there are connected components with some edges included and not-included there.
    MRMESH_API void flipOrientation( const UndirectedEdgeBitSet * fullComponents = nullptr );


    /// appends mesh topology (from) in addition to the current topology: creates new edges, faces, verts;
    /// \param rearrangeTriangles if true then the order of triangles is selected according to the order of their vertices,
    /// please call rotateTriangles() first
    /// \param outFmap,outVmap,outEmap (optionally) returns mappings: from.id -> this.id
    MRMESH_API void addPart( const MeshTopology & from,
        FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );

    /// the same but copies only portion of (from) specified by fromFaces,
    MRMESH_API void addPartByMask( const MeshTopology & from, const FaceBitSet & fromFaces, const PartMapping & map );

    /// this version has more parameters
    /// \param flipOrientation if true then every from triangle is inverted before adding
    /// \param thisContours contours on this mesh (no left face) that have to be stitched with
    /// \param fromContours contours on from mesh during addition (no left face if flipOrientation otherwise no right face)
    MRMESH_API void addPartByMask( const MeshTopology & from, const FaceBitSet & fromFaces, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {}, const std::vector<EdgePath> & fromContours = {},
        const PartMapping & map = {} );

    /// fromFaces contains mapping from this-topology (considering it is empty) to from-topology
    MRMESH_API void addPartByFaceMap( const MeshTopology & from, const FaceMap & fromFaces, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {}, const std::vector<EdgePath> & fromContours = {},
        const PartMapping & map = {} );

    /// both addPartByMask and addPartByFaceMap call this general implementation
    template<typename I>
    MRMESH_API void addPartBy( const MeshTopology & from, I fbegin, I fend, size_t fcount, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {},
        const std::vector<EdgePath> & fromContours = {},
        const PartMapping & map = {} );


    /// for each triangle selects edgeWithLeft with minimal origin vertex
    MRMESH_API void rotateTriangles();

    /// tightly packs all arrays eliminating lone edges and invalid faces and vertices
    /// \param outFmap,outVmap,outEmap if given returns mappings: old.id -> new.id;
    /// \param rearrangeTriangles if true then calls rotateTriangles() 
    /// and selects the order of triangles according to the order of their vertices
    MRMESH_API void pack( FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );

    /// tightly packs all arrays eliminating lone edges and invalid faces and vertices;
    /// reorder all faces, vertices and edges according to given maps, each containing old id -> new id mapping
    MRMESH_API void pack( const PackMapping & map );

    /// tightly packs all arrays eliminating lone edges and invalid faces and vertices;
    /// reorder all faces, vertices and edges according to given maps, each containing old id -> new id mapping;
    /// unlike \ref pack method, this method allocates minimal amount of memory for its operation but works much slower
    MRMESH_API void packMinMem( const PackMapping & map );


    /// saves in binary stream
    MRMESH_API void write( std::ostream & s ) const;

    /// loads from binary stream
    /// \return text of error if any
    MRMESH_API VoidOrErrStr read( std::istream& s, ProgressCallback callback = {} );

    /// compare that two topologies are exactly the same
    [[nodiscard]] MRMESH_API bool operator ==( const MeshTopology & b ) const;

    /// These function are for parallel mesh creation from different threads. If you are not sure, do not use them.
    /// \details resizes all internal vectors and sets the numbers of valid elements in preparation for addPackedPart;
    /// edges are resized without initialization (so the user must initialize them using addPackedPart)
    MRMESH_API void resizeBeforeParallelAdd( size_t edgeSize, size_t vertSize, size_t faceSize );

    /// copies topology (from) into this;
    /// \param from edges must be tightly packes without any lone edges, and they are mapped [0, from.edges.size()) -> [toEdgeId, toEdgeId + from.edges.size());
    /// \param fmap,vmap mapping of vertices and faces if it is given ( from.id -> this.id )
    MRMESH_API void addPackedPart( const MeshTopology & from, EdgeId toEdgeId,
        const FaceMap & fmap, const VertMap & vmap );

    /// compute
    /// 1) numValidVerts_ and validVerts_ from edgePerVertex_
    /// 2) numValidFaces_ and validFaces_ from edgePerFace_
    /// and activates their auto-update
    MRMESH_API bool computeValidsFromEdges( ProgressCallback cb = {} );

    /// stops updating validVerts(), validFaces(), numValidVerts(), numValidFaces() for parallel processing of mesh parts
    MRMESH_API void stopUpdatingValids();

    /// returns whether the methods validVerts(), validFaces(), numValidVerts(), numValidFaces() can be called
    [[nodiscard]] bool updatingValids() const { return updateValids_; }

    /// for incident vertices and faces of given edges, remember one of them as edgeWithOrg and edgeWithLeft;
    /// this is important in parallel algorithms where other edges may change but stable ones will survive
    MRMESH_API void preferEdges( const UndirectedEdgeBitSet & stableEdges );

    // constructs triangular grid mesh topology in parallel
    MRMESH_API bool buildGridMesh( const GridSettings& settings, ProgressCallback cb = {} );

    /// verifies that all internal data structures are valid
    MRMESH_API bool checkValidity( ProgressCallback cb = {} ) const;

private:
    friend class MeshDiff;
    /// computes from edges_ all remaining fields: \n
    /// 1) numValidVerts_, 2) validVerts_, 3) edgePerVertex_, 
    /// 4) numValidFaces_, 5) validFaces_, 6) edgePerFace_
    MRMESH_API void computeAllFromEdges_();

private:
    /// sets new origin to the full origin ring including this edge, without updating edgePerVertex_ table
    void setOrg_( EdgeId a, VertId v );

    /// sets new left face to the full left ring including this edge, without updating edgePerFace_ table
    void setLeft_( EdgeId a, FaceId f );

    /// data of every half-edge
    struct HalfEdgeRecord
    {
        EdgeId next; ///< next counter clock wise half-edge in the origin ring
        EdgeId prev; ///< next clock wise half-edge in the origin ring
        VertId org;  ///< vertex at the origin of the edge
        FaceId left; ///< face at the left of the edge

        bool operator ==( const HalfEdgeRecord& b ) const
        {
            return next == b.next && prev == b.prev && org == b.org && left == b.left;
        }
        HalfEdgeRecord() noexcept = default;
        explicit HalfEdgeRecord( NoInit ) noexcept : next( noInit ), prev( noInit ), org( noInit ), left( noInit ) {}
    };
    /// translates all fields in the record for this edge given maps
    void translateNoFlip_( HalfEdgeRecord & r,
        const FaceMap & fmap, const VertMap & vmap, const WholeEdgeMap & emap ) const;
    void translate_( HalfEdgeRecord & r, HalfEdgeRecord & rsym,
        const FaceMap & fmap, const VertMap & vmap, const WholeEdgeMap & emap, bool flipOrientation ) const;
    void translateNoFlip_( HalfEdgeRecord & r,
        const FaceHashMap & fmap, const VertHashMap & vmap, const WholeEdgeHashMap & emap ) const;
    void translate_( HalfEdgeRecord & r, HalfEdgeRecord & rsym,
        const FaceHashMap & fmap, const VertHashMap & vmap, const WholeEdgeHashMap & emap, bool flipOrientation ) const;

    /// edges_: EdgeId -> edge data
    Vector<HalfEdgeRecord, EdgeId> edges_;

    /// edgePerVertex_: VertId -> one edge id of one of edges with origin there
    Vector<EdgeId, VertId> edgePerVertex_;
    VertBitSet validVerts_; ///< each true bit here corresponds to valid element in edgePerVertex_

    /// edgePerFace_: FaceId -> one edge id with this face at left
    Vector<EdgeId, FaceId> edgePerFace_;
    FaceBitSet validFaces_; ///< each true bit here corresponds to valid element in edgePerFace_

    int numValidVerts_ = 0; ///< the number of valid elements in edgePerVertex_ or set bits in validVerts_
    int numValidFaces_ = 0; ///< the number of valid elements in edgePerFace_ or set bits in validFaces_

    bool updateValids_ = true; ///< if false, validVerts_, validFaces_, numValidVerts_, numValidFaces_ are not updated
};

template <typename T>
void MeshTopology::forEachVertex( const MeshTriPoint & p, T && callback ) const
{
    if ( auto v = p.inVertex( *this ) )
    {
        callback( v );
        return;
    }
    if ( auto e = p.onEdge( *this ) )
    {
        callback( org( e.e ) );
        callback( dest( e.e ) );
        return;
    }

    VertId v[3];
    getLeftTriVerts( p.e, v );
    for ( int i = 0; i < 3; ++i )
        callback( v[i] );
}

template<typename T>
void MeshTopology::flipEdgesIn( const EdgeId e0, T && flipNeeded )
{
    EdgeId e = e0;
    for (;;)
    {
        auto testEdge = prev( e.sym() );
        if ( left( testEdge ) && right( testEdge ) && flipNeeded( testEdge ) )
            flipEdge( testEdge );
        else
        {
            e = next( e );
            if ( e == e0 )
                break; // full ring has been inspected
        }
    } 
}

template<typename T>
void MeshTopology::flipEdgesOut( EdgeId e0, T && flipNeeded )
{
    EdgeId e = e0;
    for (;;)
    {
        if ( left( e ) && right( e ) && flipNeeded( e ) )
        {
            e0 = next( e );
            flipEdge( e );
            e = e0;
        }
        else
        {
            e = next( e );
            if ( e == e0 )
                break; // full ring has been inspected
        }
    } 
}

// rearrange vector values by map (old.id -> new.id)
template<typename T, typename I>
[[nodiscard]] Vector<T, I> rearrangeVectorByMap( const Vector<T, I>& oldVector, const BMap<I, I>& map )
{
    Vector<T, I> newVector;
    newVector.resize( map.tsize );

    const auto& mData = map.b.data();
    const auto sz = std::min( oldVector.size(), map.b.size() );
    for ( I i = I(0); i < sz; ++i)
    {
        I newV = mData[i];
        if ( newV.valid() )
            newVector[newV] = oldVector[i];
    }
    return newVector;
}

MRMESH_API void loadMeshDll();

} // namespace MR
