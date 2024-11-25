#pragma once

#include "MRMeshBuilderTypes.h"
#include "MRMeshTopology.h"
#include "MRMeshProject.h"
#include "MREdgePoint.h"
#include "MRLineSegm.h"
#include "MRUniqueThreadSafeOwner.h"
#include "MRWriter.h"
#include "MRConstants.h"
#include "MRProgressCallback.h"
#include <cfloat>

namespace MR
{

/// \defgroup MeshGroup Mesh

/// This class represents a mesh, including topology (connectivity) information and point coordinates,
/// as well as some caches to accelerate search algorithms
/// \ingroup MeshGroup
struct [[nodiscard]] Mesh
{
    MeshTopology topology;
    VertCoords points;

    /// construct mesh from vertex coordinates and a set of triangles with given ids
    [[nodiscard]] MRMESH_API static Mesh fromTriangles(
        VertCoords vertexCoordinates,
        const Triangulation& t, const MeshBuilder::BuildSettings& settings = {}, ProgressCallback cb = {} );

    /// construct mesh from TriMesh representation
    [[nodiscard]] MRMESH_API static Mesh fromTriMesh(
        TriMesh && triMesh, ///< points of triMesh will be moves in the result
        const MeshBuilder::BuildSettings& settings = {}, ProgressCallback cb = {} );

    /// construct mesh from vertex coordinates and a set of triangles with given ids;
    /// unlike simple fromTriangles() it tries to resolve non-manifold vertices by creating duplicate vertices
    [[nodiscard]] MRMESH_API static Mesh fromTrianglesDuplicatingNonManifoldVertices(
        VertCoords vertexCoordinates,
        Triangulation & t,
        std::vector<MeshBuilder::VertDuplication> * dups = nullptr,
        const MeshBuilder::BuildSettings & settings = {} );

    /// construct mesh from vertex coordinates and construct mesh topology from face soup,
    /// where each face can have arbitrary degree (not only triangles);
    /// all non-triangular faces will be automatically subdivided on triangles
    [[nodiscard]] MRMESH_API static Mesh fromFaceSoup(
        VertCoords vertexCoordinates,
        const std::vector<VertId> & verts, const Vector<MeshBuilder::VertSpan, FaceId> & faces,
        const MeshBuilder::BuildSettings& settings = {}, ProgressCallback cb = {} );

    /// construct mesh from point triples;
    /// \param duplicateNonManifoldVertices = false, all coinciding points are given the same VertId in the result;
    /// \param duplicateNonManifoldVertices = true, it tries to avoid non-manifold vertices by creating duplicate vertices with same coordinates 
    [[nodiscard]] MRMESH_API static Mesh fromPointTriples( const std::vector<Triangle3f> & posTriples, bool duplicateNonManifoldVertices );

    /// compare that two meshes are exactly the same
    [[nodiscard]] MRMESH_API bool operator ==( const Mesh & b ) const;

    /// returns coordinates of the edge origin
    [[nodiscard]] Vector3f orgPnt( EdgeId e ) const { return points[ topology.org( e ) ]; }

    /// returns coordinates of the edge destination
    [[nodiscard]] Vector3f destPnt( EdgeId e ) const { return points[ topology.dest( e ) ]; }

    /// returns vector equal to edge destination point minus edge origin point
    [[nodiscard]] Vector3f edgeVector( EdgeId e ) const { return destPnt( e ) - orgPnt( e ); }

    /// returns line segment of given edge
    [[nodiscard]] LineSegm3f edgeSegment( EdgeId e ) const { return { orgPnt( e ), destPnt( e ) }; }

    /// returns a point on the edge: origin point for f=0 and destination point for f=1
    [[nodiscard]] Vector3f edgePoint( EdgeId e, float f ) const { return f * destPnt( e ) + ( 1 - f ) * orgPnt( e ); }

    /// computes coordinates of point given as edge and relative position on it
    [[nodiscard]] Vector3f edgePoint( const MeshEdgePoint & ep ) const { return edgePoint( ep.e, ep.a ); }

    /// computes the center of given edge
    [[nodiscard]] Vector3f edgeCenter( UndirectedEdgeId e ) const { return edgePoint( e, 0.5f ); }

    /// returns three points of left face of e
    MRMESH_API void getLeftTriPoints( EdgeId e, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const;

    /// returns three points of left face of e
    void getLeftTriPoints( EdgeId e, Vector3f (&v)[3] ) const { getLeftTriPoints( e, v[0], v[1], v[2] ); }

    /// returns three points of left face of e
    [[nodiscard]] Triangle3f getLeftTriPoints( EdgeId e ) const { Triangle3f res; getLeftTriPoints( e, res[0], res[1], res[2] ); return res; }

    /// returns three points of given face
    void getTriPoints( FaceId f, Vector3f & v0, Vector3f & v1, Vector3f & v2 ) const { getLeftTriPoints( topology.edgeWithLeft( f ), v0, v1, v2 ); }

    /// returns three points of given face
    void getTriPoints( FaceId f, Vector3f (&v)[3] ) const { getTriPoints( f, v[0], v[1], v[2] ); }

    /// returns three points of given face
    [[nodiscard]] Triangle3f getTriPoints( FaceId f ) const { Triangle3f res; getTriPoints( f, res[0], res[1], res[2] ); return res; }

    /// computes coordinates of point given as face and barycentric representation
    [[nodiscard]] MRMESH_API Vector3f triPoint( const MeshTriPoint & p ) const;

    /// returns the centroid of given triangle
    [[nodiscard]] MRMESH_API Vector3f triCenter( FaceId f ) const;

    /// returns aspect ratio of given mesh triangle equal to the ratio of the circum-radius to twice its in-radius
    [[nodiscard]] MRMESH_API float triangleAspectRatio( FaceId f ) const;
    
    /// returns squared circumcircle diameter of given mesh triangle
    [[nodiscard]] MRMESH_API float circumcircleDiameterSq( FaceId f ) const;

    /// returns circumcircle diameter of given mesh triangle
    [[nodiscard]] MRMESH_API float circumcircleDiameter( FaceId f ) const;

    /// converts vertex into barycentric representation
    [[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( VertId v ) const;

    /// converts face id and 3d point into barycentric representation
    [[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( FaceId f, const Vector3f & p ) const;

    /// converts face id and 3d point into barycentric representation
    [[nodiscard]] MRMESH_API MeshTriPoint toTriPoint( const PointOnFace& p ) const;

    /// converts vertex into edge-point representation
    [[nodiscard]] MRMESH_API MeshEdgePoint toEdgePoint( VertId v ) const;

    /// converts edge and 3d point into edge-point representation
    [[nodiscard]] MRMESH_API MeshEdgePoint toEdgePoint( EdgeId e, const Vector3f & p ) const;

    /// returns one of three face vertices, closest to given point
    [[nodiscard]] MRMESH_API VertId getClosestVertex( const PointOnFace & p ) const;

    /// returns one of three face vertices, closest to given point
    [[nodiscard]] VertId getClosestVertex( const MeshTriPoint & p ) const { return getClosestVertex( PointOnFace{ topology.left( p.e ), triPoint( p ) } ); }

    /// returns one of three face edges, closest to given point
    [[nodiscard]] MRMESH_API UndirectedEdgeId getClosestEdge( const PointOnFace & p ) const;

    /// returns one of three face edges, closest to given point
    [[nodiscard]] UndirectedEdgeId getClosestEdge( const MeshTriPoint & p ) const { return getClosestEdge( PointOnFace{ topology.left( p.e ), triPoint( p ) } ); }

    /// returns Euclidean length of the edge
    [[nodiscard]] float edgeLength( UndirectedEdgeId e ) const { return edgeVector( e ).length(); }

    /// returns squared Euclidean length of the edge (faster to compute than length)
    [[nodiscard]] float edgeLengthSq( UndirectedEdgeId e ) const { return edgeVector( e ).lengthSq(); }

    /// computes directed double area of left triangular face of given edge
    [[nodiscard]] MRMESH_API Vector3f leftDirDblArea( EdgeId e ) const;

    /// computes directed double area for a triangular face from its vertices
    [[nodiscard]] Vector3f dirDblArea( FaceId f ) const { return leftDirDblArea( topology.edgeWithLeft( f ) ); }

    /// returns twice the area of given face
    [[nodiscard]] float dblArea( FaceId f ) const { return dirDblArea( f ).length(); }

    /// returns the area of given face
    [[nodiscard]] float area( FaceId f ) const { return 0.5f * dblArea( f ); }

    /// computes the area of given face-region
    [[nodiscard]] MRMESH_API double area( const FaceBitSet & fs ) const;

    /// computes the area of given face-region (or whole mesh)
    [[nodiscard]] double area( const FaceBitSet * fs = nullptr ) const { return area( topology.getFaceIds( fs ) ); }

    /// computes the sum of directed areas for faces from given region
    [[nodiscard]] MRMESH_API Vector3d dirArea( const FaceBitSet & fs ) const;

    /// computes the sum of directed areas for faces from given region (or whole mesh)
    [[nodiscard]] Vector3d dirArea( const FaceBitSet * fs = nullptr ) const { return dirArea( topology.getFaceIds( fs ) ); }

    /// computes the sum of absolute projected area of faces from given region as visible if look from given direction
    [[nodiscard]] MRMESH_API double projArea( const Vector3f & dir, const FaceBitSet & fs ) const;

    /// computes the sum of absolute projected area of faces from given region (or whole mesh) as visible if look from given direction
    [[nodiscard]] double projArea( const Vector3f & dir, const FaceBitSet * fs = nullptr ) const { return projArea( dir, topology.getFaceIds( fs ) ); }

    /// returns volume of closed mesh region, if region is not closed DBL_MAX is returned
    /// if region is nullptr - whole mesh is region
    [[nodiscard]] MRMESH_API double volume( const FaceBitSet* region = nullptr ) const;

    /// computes the perimeter of the hole specified by one of its edges with no valid left face (left is hole)
    [[nodiscard]] MRMESH_API double holePerimiter( EdgeId e ) const;

    /// computes directed area of the hole specified by one of its edges with no valid left face (left is hole);
    /// if the hole is planar then returned vector is orthogonal to the plane pointing outside and its magnitude is equal to hole area
    [[nodiscard]] MRMESH_API Vector3d holeDirArea( EdgeId e ) const;

    /// computes triangular face normal from its vertices
    [[nodiscard]] Vector3f leftNormal( EdgeId e ) const { return leftDirDblArea( e ).normalized(); }

    /// computes triangular face normal from its vertices
    [[nodiscard]] Vector3f normal( FaceId f ) const { return dirDblArea( f ).normalized(); }

    /// computes sum of directed double areas of all triangles around given vertex
    [[nodiscard]] MRMESH_API Vector3f dirDblArea( VertId v ) const;

    /// computes the length of summed directed double areas of all triangles around given vertex
    [[nodiscard]] float dblArea( VertId v ) const { return dirDblArea( v ).length(); }

    /// computes normal in a vertex using sum of directed areas of neighboring triangles
    [[nodiscard]] Vector3f normal( VertId v ) const { return dirDblArea( v ).normalized(); }

    /// computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates and normalizes again;
    /// this is the same normal as in rendering with smooth shading
    [[nodiscard]] MRMESH_API Vector3f normal( const MeshTriPoint & p ) const;

    /// computes angle-weighted sum of normals of incident faces of given vertex (only (region) faces will be considered);
    /// the sum is normalized before returning
    [[nodiscard]] MRMESH_API Vector3f pseudonormal( VertId v, const FaceBitSet * region = nullptr ) const;

    /// computes normalized half sum of face normals sharing given edge (only (region) faces will be considered);
    [[nodiscard]] MRMESH_API Vector3f pseudonormal( UndirectedEdgeId e, const FaceBitSet * region = nullptr ) const;

    /// returns pseudonormal in corresponding face/edge/vertex for signed distance calculation;
    /// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.107.9173&rep=rep1&type=pdf
    /// unlike normal( const MeshTriPoint & p ), this is not a smooth function
    [[nodiscard]] MRMESH_API Vector3f pseudonormal( const MeshTriPoint & p, const FaceBitSet * region = nullptr ) const;

    /// given a point (pt) in 3D and the closest point to in on mesh (proj),
    /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API float signedDistance( const Vector3f & pt, const MeshProjectionResult & proj, const FaceBitSet * region = nullptr ) const;
    [[deprecated]] MRMESH_API float signedDistance( const Vector3f & pt, const MeshTriPoint & proj, const FaceBitSet * region = nullptr ) const;

    /// given a point (pt) in 3D, computes the closest point on mesh, and
    /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API float signedDistance( const Vector3f & pt ) const;

    /// given a point (pt) in 3D, computes the closest point on mesh, and
    /// \return signed distance from pt to mesh: positive value - outside mesh, negative - inside mesh;
    ///   or std::nullopt if the projection point is not within maxDist;
    /// this method can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API std::optional<float> signedDistance( const Vector3f & pt, float maxDistSq, const FaceBitSet * region = nullptr ) const;

    /// computes generalized winding number in a point (pt), which is
    /// * for closed mesh with normals outside: 1 inside, 0 outside;
    /// * for planar mesh: 0.5 inside, -0.5 outside;
    /// and in general is equal to (portion of solid angle where inside part of mesh is observable) minus (portion of solid angle where outside part of mesh is observable)
    /// \param beta determines the precision of fast approximation: the more the better, recommended value 2 or more
    [[nodiscard]] MRMESH_API float calcFastWindingNumber( const Vector3f & pt, float beta = 2 ) const;

    /// computes whether a point (pt) is located outside the object surrounded by this mesh using generalized winding number
    /// \param beta determines the precision of winding number computation: the more the better, recommended value 2 or more
    [[nodiscard]] bool isOutside( const Vector3f & pt, float windingNumberThreshold = 0.5f, float beta = 2 ) const { return calcFastWindingNumber( pt, beta ) <= windingNumberThreshold; }

    /// computes whether a point (pt) is located outside the object surrounded by this mesh
    /// using pseudonormal at the closest point to in on mesh (proj);
    /// this method works much faster than \ref isOutside but can return wrong sign if the closest point is located on self-intersecting part of the mesh
    [[nodiscard]] MRMESH_API bool isOutsideByProjNorm( const Vector3f & pt, const MeshProjectionResult & proj, const FaceBitSet * region = nullptr ) const;

    /// computes the sum of triangle angles at given vertex; optionally returns whether the vertex is on boundary
    [[nodiscard]] MRMESH_API float sumAngles( VertId v, bool * outBoundaryVert = nullptr ) const;

    /// returns vertices where the sum of triangle angles is below given threshold
    [[nodiscard]] MRMESH_API Expected<VertBitSet> findSpikeVertices( float minSumAngle, const VertBitSet* region = nullptr, ProgressCallback cb = {} ) const;

    /// given an edge between two triangular faces, computes sine of dihedral angle between them:
    /// 0 if both faces are in the same plane,
    /// positive if the faces form convex surface,
    /// negative if the faces form concave surface
    [[nodiscard]] MRMESH_API float dihedralAngleSin( UndirectedEdgeId e ) const;

    /// given an edge between two triangular faces, computes cosine of dihedral angle between them:
    /// 1 if both faces are in the same plane,
    /// 0 if the surface makes right angle turn at the edge,
    /// -1 if the faces overlap one another
    [[nodiscard]] MRMESH_API float dihedralAngleCos( UndirectedEdgeId e ) const;

    /// given an edge between two triangular faces, computes the dihedral angle between them:
    /// 0 if both faces are in the same plane,
    /// positive if the faces form convex surface,
    /// negative if the faces form concave surface;
    /// please consider the usage of faster dihedralAngleSin(e) and dihedralAngleCos(e)
    [[nodiscard]] MRMESH_API float dihedralAngle( UndirectedEdgeId e ) const;

    /// computes discrete mean curvature in given vertex, measures in length^-1;
    /// 0 for planar regions, positive for convex surface, negative for concave surface
    [[nodiscard]] MRMESH_API float discreteMeanCurvature( VertId v ) const;

    /// computes discrete mean curvature in given edge, measures in length^-1;
    /// 0 for planar regions, positive for convex surface, negative for concave surface
    [[nodiscard]] MRMESH_API float discreteMeanCurvature( UndirectedEdgeId e ) const;

    /// computes discrete Gaussian curvature (or angle defect) at given vertex,
    /// which 0 in inner vertices on planar mesh parts and reaches 2*pi on needle's tip, see http://math.uchicago.edu/~may/REU2015/REUPapers/Upadhyay.pdf
    /// optionally returns whether the vertex is on boundary
    [[nodiscard]] float discreteGaussianCurvature( VertId v, bool * outBoundaryVert = nullptr ) const { return 2 * PI_F - sumAngles( v, outBoundaryVert ); }

    /// finds all mesh edges where dihedral angle is distinct from planar PI angle on at least given value
    [[nodiscard]] MRMESH_API UndirectedEdgeBitSet findCreaseEdges( float angleFromPlanar ) const;

    /// computes cotangent of the angle in the left( e ) triangle opposite to e,
    /// and returns 0 if left face does not exist
    [[nodiscard]] MRMESH_API float leftCotan( EdgeId e ) const;

    /// computes sum of cotangents of the angle in the left and right triangles opposite to given edge,
    /// and returns 0 if left face does not exist
    [[nodiscard]] float cotan( UndirectedEdgeId ue ) const { EdgeId e{ ue }; return leftCotan( e ) + leftCotan( e.sym() ); }

    /// computes quadratic form in the vertex as the sum of squared distances from
    /// 1) planes of adjacent triangles
    /// 2) lines of adjacent boundary and crease edges
    [[nodiscard]] MRMESH_API QuadraticForm3f quadraticForm( VertId v, const FaceBitSet * region = nullptr, const UndirectedEdgeBitSet * creases = nullptr ) const;

    /// passes through all valid vertices and finds the minimal bounding box containing all of them;
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const AffineXf3f * toWorld = nullptr ) const;

    /// returns the bounding box containing all valid vertices (implemented via getAABBTree())
    /// this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision
    [[nodiscard]] MRMESH_API Box3f getBoundingBox() const;

    /// passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them
    /// if toWorld transformation is given then returns minimal bounding box in world space
    [[nodiscard]] MRMESH_API Box3f computeBoundingBox( const FaceBitSet* region, const AffineXf3f* toWorld = nullptr ) const;

    /// computes average length of an edge in this mesh
    [[nodiscard]] MRMESH_API float averageEdgeLength() const;

    /// computes average position of all valid mesh vertices
    [[nodiscard]] MRMESH_API Vector3f findCenterFromPoints() const;

    /// computes center of mass considering that density of all triangles is the same
    [[nodiscard]] MRMESH_API Vector3f findCenterFromFaces() const;

    /// computes bounding box and returns its center
    [[nodiscard]] MRMESH_API Vector3f findCenterFromBBox() const;

    /// for all points not in topology.getValidVerts() sets coordinates to (0,0,0)
    MRMESH_API void zeroUnusedPoints();

    /// applies given transformation to specified vertices
    /// if region is nullptr, all valid mesh vertices are used
    MRMESH_API void transform( const AffineXf3f& xf, const VertBitSet* region = nullptr );

    /// creates new point and assigns given position to it
    MRMESH_API VertId addPoint( const Vector3f & pos );

    /// append points to mesh and connect them as closed edge loop
    /// returns first EdgeId of new edges
    MRMESH_API EdgeId addSeparateEdgeLoop(const std::vector<Vector3f>& contourPoints);

    /// append points to mesh and connect them
    /// returns first EdgeId of new edges
    MRMESH_API EdgeId addSeparateContours( const Contours3f& contours, const AffineXf3f* xf = nullptr );

    /// append points to mesh and connect them to given edges making edge loop
    /// first point connects with first edge dest
    /// last point connects with last edge org
    /// note that first and last edge should have no left face
    MRMESH_API void attachEdgeLoopPart( EdgeId first, EdgeId last, const std::vector<Vector3f>& contourPoints );

    /// split given edge on two parts:
    /// dest(returned-edge) = org(e) - newly created vertex,
    /// org(returned-edge) = org(e-before-split),
    /// dest(e) = dest(e-before-split)
    /// \details left and right faces of given edge if valid are also subdivided on two parts each;
    /// if left or right faces of the original edge were in the region, then include new parts of these faces in the region
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    MRMESH_API EdgeId splitEdge( EdgeId e, const Vector3f & newVertPos, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr );
    // same, but split given edge on two equal parts
    EdgeId splitEdge( EdgeId e, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr ) { return splitEdge( e, edgeCenter( e ), region, new2Old ); }

    /// split given triangle on three triangles, introducing new vertex with given coordinates and connecting it to original triangle vertices;
    /// if region is given, then it must include (f) and new faces will be added there as well
    /// \param new2Old receive mapping from newly appeared triangle to its original triangle (part to full)
    MRMESH_API VertId splitFace( FaceId f, const Vector3f & newVertPos, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr );
    // same, putting new vertex in the centroid of original triangle
    VertId splitFace( FaceId f, FaceBitSet * region = nullptr, FaceHashMap * new2Old = nullptr ) { return splitFace( f, triCenter( f ), region, new2Old ); }

    /// appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points
    MRMESH_API void addPart( const Mesh & from,
        // optionally returns mappings: from.id -> this.id
        FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );

    /// the same but copies only portion of (from) specified by fromFaces
    MRMESH_API void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, const PartMapping & map );

    /// this version has more parameters:
    ///   if flipOrientation then every from triangle is inverted before adding
    MRMESH_API void addPartByMask( const Mesh & from, const FaceBitSet & fromFaces, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {}, // contours on this mesh that have to be stitched with
        const std::vector<EdgePath> & fromContours = {}, // contours on from mesh during addition
        // optionally returns mappings: from.id -> this.id
        const PartMapping & map = {} );

    /// fromFaces contains mapping from this-mesh (considering it is empty) to from-mesh
    MRMESH_API void addPartByFaceMap( const Mesh & from, const FaceMap & fromFaces, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {}, // contours on this mesh that have to be stitched with
        const std::vector<EdgePath> & fromContours = {}, // contours on from mesh during addition
        // optionally returns mappings: from.id -> this.id
    
        const PartMapping & map = {} );
    /// both addPartByMask and addPartByFaceMap call this general implementation
    template<typename I>
    MRMESH_API void addPartBy( const Mesh & from, I fbegin, I fend, size_t fcount, bool flipOrientation = false,
        const std::vector<EdgePath> & thisContours = {},
        const std::vector<EdgePath> & fromContours = {},
        PartMapping map = {} );

    /// creates new mesh from given triangles of this mesh
    MRMESH_API Mesh cloneRegion( const FaceBitSet & region, bool flipOrientation = false, const PartMapping & map = {} ) const;

    /// tightly packs all arrays eliminating lone edges and invalid face, verts and points,
    /// optionally returns mappings: old.id -> new.id
    MRMESH_API void pack( FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr, bool rearrangeTriangles = false );

    /// packs tightly and rearranges vertices, triangles and edges to put close in space elements in close indices
    /// \param preserveAABBTree whether to keep valid mesh's AABB tree after return (it will take longer to compute and it will occupy more memory)
    MRMESH_API PackMapping packOptimally( bool preserveAABBTree = true );
    MRMESH_API Expected<PackMapping> packOptimally( bool preserveAABBTree, ProgressCallback cb );

    /// deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces ant not in \param keepFaces
    MRMESH_API void deleteFaces( const FaceBitSet & fs, const UndirectedEdgeBitSet * keepEdges = nullptr );

    /// finds the closest mesh point on this mesh (or its region) to given point;
    /// \param point source location to look the closest to
    /// \param res found closest point including Euclidean coordinates and FaceId
    /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
    /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
    /// \return false if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
    [[nodiscard]] MRMESH_API bool projectPoint( const Vector3f& point, PointOnFace& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const;

    /// finds the closest mesh point on this mesh (or its region) to given point;
    /// \param point source location to look the closest to
    /// \param res found closest point including Euclidean coordinates, barycentric coordinates, FaceId and squared distance to point
    /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
    /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
    /// \return false if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
    [[nodiscard]] MRMESH_API bool projectPoint( const Vector3f& point, MeshProjectionResult& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const;
    [[nodiscard]] bool findClosestPoint( const Vector3f& point, MeshProjectionResult& res, float maxDistSq = FLT_MAX, const FaceBitSet* region = nullptr, const AffineXf3f * xf = nullptr ) const { return projectPoint( point, res, maxDistSq, region, xf ); }

    /// finds the closest mesh point on this mesh (or its region) to given point;
    /// \param point source location to look the closest to
    /// \param maxDistSq search only in the ball with sqrt(maxDistSq) radius around given point, smaller value here increases performance
    /// \param xf is mesh-to-point transformation, if not specified then identity transformation is assumed and works much faster;
    /// \return found closest point including Euclidean coordinates, barycentric coordinates, FaceId and squared distance to point
    ///         or std::nullopt if no mesh point is found in the ball with sqrt(maxDistSq) radius around given point
    [[nodiscard]] MRMESH_API std::optional<MeshProjectionResult> projectPoint( const Vector3f& point, float maxDistSq = FLT_MAX, const FaceBitSet * region = nullptr, const AffineXf3f * xf = nullptr ) const;
    [[nodiscard]] std::optional<MeshProjectionResult> findClosestPoint( const Vector3f& point, float maxDistSq = FLT_MAX, const FaceBitSet * region = nullptr, const AffineXf3f * xf = nullptr ) const { return projectPoint( point, maxDistSq, region, xf ); }

    /// returns cached aabb-tree for this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTree & getAABBTree() const;

    /// returns cached aabb-tree for this mesh, but does not create it if it did not exist
    [[nodiscard]] const AABBTree * getAABBTreeNotCreate() const { return AABBTreeOwner_.get(); }

    /// returns cached aabb-tree for points of this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const AABBTreePoints & getAABBTreePoints() const;

    /// returns cached aabb-tree for points of this mesh, but does not create it if it did not exist
    [[nodiscard]] const AABBTreePoints * getAABBTreePointsNotCreate() const { return AABBTreePointsOwner_.get(); }

    /// returns cached dipoles of aabb-tree nodes for this mesh, creating it if it did not exist in a thread-safe manner
    MRMESH_API const Dipoles & getDipoles() const;

    /// returns cached dipoles of aabb-tree nodes for this mesh, but does not create it if it did not exist
    [[nodiscard]] const Dipoles * getDipolesNotCreate() const { return dipolesOwner_.get(); }

    /// invalidates caches (aabb-trees) after any change in mesh geometry or topology
    /// \param pointsChanged specifies whether points have changed (otherwise only topology has changed)
    MRMESH_API void invalidateCaches( bool pointsChanged = true );

    /// updates existing caches in case of few vertices were changed insignificantly,
    /// and topology remained unchanged;
    /// it shall be considered as a faster alternative to invalidateCaches() and following rebuild of trees
    MRMESH_API void updateCaches( const VertBitSet & changedVerts );

    // returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

    /// requests the removal of unused capacity
    MRMESH_API void shrinkToFit();

    /// reflects the mesh from a given plane
    MRMESH_API void mirror( const Plane3f& plane );

private:
    mutable UniqueThreadSafeOwner<AABBTree> AABBTreeOwner_;
    mutable UniqueThreadSafeOwner<AABBTreePoints> AABBTreePointsOwner_;
    mutable UniqueThreadSafeOwner<Dipoles> dipolesOwner_;
};

} //namespace MR
