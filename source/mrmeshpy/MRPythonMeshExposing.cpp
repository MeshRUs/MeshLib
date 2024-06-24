#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRCube.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRBoolean.h"
#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRMeshBuilder.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRCylinder.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRPartMapping.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRMeshExtrude.h"
#include "MRMesh/MRMeshBoundary.h"
#include <pybind11/functional.h>

using namespace MR;

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, MeshTopology, MR::MeshTopology )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshTopology, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( MeshTopology ).
        def( pybind11::init<>() ).
        def( "numValidFaces", &MeshTopology::numValidFaces, "Returns the number of valid faces." ).
        def( "getAllTriVerts", &MeshTopology::getAllTriVerts, "Returns three vertex ids for valid triangles, invalid triangles are skipped.").
        def( "numValidVerts", &MeshTopology::numValidVerts, "Returns the number of valid vertices." ).
        def( "getValidFaces", &MeshTopology::getValidFaces, pybind11::return_value_policy::copy, "returns cached set of all valid faces" ).
        def( "getValidVerts", &MeshTopology::getValidVerts, pybind11::return_value_policy::copy, "returns cached set of all valid vertices" ).
        def( "flip", (void (MeshTopology::*)(FaceBitSet&)const)&MeshTopology::flip, pybind11::arg( "fs" ), "sets in (fs) all valid faces that were not selected before the call, and resets other bits" ).
        def( "flip", (void (MeshTopology::*)(VertBitSet&)const)&MeshTopology::flip, pybind11::arg( "vs" ), "sets in (vs) all valid vertices that were not selected before the call, and resets other bits" ).
        def( "flipOrientation", &MeshTopology::flipOrientation, pybind11::arg( "fullComponents" ) = nullptr, "flip orientation (normals) of all faces" ).
        def( "hasEdge", &MeshTopology::hasEdge, pybind11::arg( "he" ), "Returns true if given edge is within valid range and not-lone" ).
        def( "next", &MeshTopology::next, pybind11::arg( "he" ), "Next (counter clock wise) half-edge in the origin ring" ).
        def( "prev", &MeshTopology::prev, pybind11::arg( "he" ), "Previous (clock wise) half-edge in the origin ring" ).
        def( "org", &MeshTopology::org, pybind11::arg( "he" ), "Returns origin vertex of half-edge." ).
        def( "dest", &MeshTopology::dest, pybind11::arg( "he" ), "Returns destination vertex of half-edge." ).
        def( "left", &MeshTopology::left, pybind11::arg( "he" ), "Returns left face of half-edge." ).
        def( "right", &MeshTopology::right, pybind11::arg( "he" ), "Returns right face of half-edge." ).
        def( "getOrgDegree", &MeshTopology::getOrgDegree, pybind11::arg( "he" ), "Returns the number of edges around the origin vertex, returns 1 for lone edges." ).
        def( "getVertDegree", &MeshTopology::getVertDegree, pybind11::arg( "v" ), "Returns the number of edges around the given vertex." ).
        def( "getLeftDegree", &MeshTopology::getLeftDegree, pybind11::arg( "he" ), " Returns the number of edges around the left face: 3 for triangular faces, ..." ).
        def( "getFaceDegree", &MeshTopology::getFaceDegree, pybind11::arg( "f" ), "Returns the number of edges around the given face: 3 for triangular faces, ..." ).
        def( "findBoundaryFaces", &MeshTopology::findBoundaryFaces, "returns all boundary faces, having at least one boundary edge" ).
        def( "findBoundaryEdges", &MeshTopology::findBoundaryEdges, "returns all boundary edges, where each edge does not have valid left face" ).
        def( "findBoundaryVerts", &MeshTopology::findBoundaryVerts, "returns all boundary vertices, incident to at least one boundary edge" ).
        def( "deleteFaces", &MeshTopology::deleteFaces, pybind11::arg( "fs" ), pybind11::arg( "keepEdges" ) = nullptr,
            "deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces ant not in keepFaces" ).
        def( "findHoleRepresentiveEdges", &MeshTopology::findHoleRepresentiveEdges, "returns one edge with no valid left face for every boundary in the mesh" ).
        def( "getTriVerts", ( void( MeshTopology::* )( FaceId, VertId&, VertId&, VertId& )const )& MeshTopology::getTriVerts,
            pybind11::arg("f"), pybind11::arg( "v0" ), pybind11::arg( "v1" ), pybind11::arg( "v2" ),
            "gets 3 vertices of given triangular face;\n"
            "the vertices are returned in counter-clockwise order if look from mesh outside" ).
        def( "edgeSize", &MeshTopology::edgeSize, "returns the number of half-edge records including lone ones" ).
        def( "undirectedEdgeSize", &MeshTopology::undirectedEdgeSize, "returns the number of undirected edges (pairs of half-edges) including lone ones" ).
        def( "computeNotLoneUndirectedEdges", &MeshTopology::computeNotLoneUndirectedEdges, "computes the number of not-lone (valid) undirected edges" ).
        def( pybind11::self == pybind11::self, "compare that two topologies are exactly the same" );
} )

// these declarations fix "Invalid expression" errors in pybind11_stubgen
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VertColorMap, MR::VertColors )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VertCoords, MR::VertCoords )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, FaceMap, MR::FaceMap )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VertMap, MR::VertMap )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, WholeEdgeMap, MR::WholeEdgeMap )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, UndirectedEdgeMap, MR::UndirectedEdgeMap )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, EdgeMap, MR::EdgeMap )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VectorFloatByVert, MR::VertScalars )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, FaceNormals, MR::FaceNormals )
using VertCoords2 = Vector<Vector2f, VertId>;
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, VertCoords2, VertCoords2 )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Vector, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( VertCoords ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertCoords::vec_ );

    MR_PYTHON_CUSTOM_CLASS( FaceMap ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &FaceMap::vec_ );

    MR_PYTHON_CUSTOM_CLASS( VertMap ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertMap::vec_ );

    MR_PYTHON_CUSTOM_CLASS( WholeEdgeMap ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &WholeEdgeMap::vec_ );

    MR_PYTHON_CUSTOM_CLASS( UndirectedEdgeMap ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &UndirectedEdgeMap::vec_ );

    MR_PYTHON_CUSTOM_CLASS( EdgeMap ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &EdgeMap::vec_ );

    MR_PYTHON_CUSTOM_CLASS( VectorFloatByVert ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertScalars::vec_ );

    MR_PYTHON_CUSTOM_CLASS( FaceNormals ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &FaceNormals::vec_ );

    MR_PYTHON_CUSTOM_CLASS( VertCoords2 ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertCoords2::vec_ );

    MR_PYTHON_CUSTOM_CLASS( VertColorMap ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &VertColors::vec_ );
} )

MR_ADD_PYTHON_MAP( mrmeshpy, FaceHashMap, FaceHashMap )
MR_ADD_PYTHON_MAP( mrmeshpy, VertHashMap, VertHashMap )
MR_ADD_PYTHON_MAP( mrmeshpy, WholeEdgeHashMap, WholeEdgeHashMap )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PartMapping, MR::PartMapping )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PartMapping, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( PartMapping ).doc() =
        "mapping among elements of source mesh, from which a part is taken, and target (this) mesh";
    MR_PYTHON_CUSTOM_CLASS( PartMapping ).
        def( pybind11::init<>() ).
        def_readwrite( "src2tgtFaces", &PartMapping::src2tgtFaces, "from.id -> this.id" ).
        def_readwrite( "src2tgtVerts", &PartMapping::src2tgtVerts, "from.id -> this.id" ).
        def_readwrite( "src2tgtEdges", &PartMapping::src2tgtEdges, "from.id -> this.id" ).
        def_readwrite( "tgt2srcFaces", &PartMapping::tgt2srcFaces, "this.id -> from.id" ).
        def_readwrite( "tgt2srcVerts", &PartMapping::tgt2srcVerts, "this.id -> from.id" ).
        def_readwrite( "tgt2srcEdges", &PartMapping::tgt2srcEdges, "this.id -> from.id" );
} )


MeshTopology topologyFromTriangles( const Triangulation& t, const MeshBuilder::BuildSettings& s )
{
    return MeshBuilder::fromTriangles( t, s );
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, ThreeVertIds, MR::ThreeVertIds )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, MeshBuilderSettings, MeshBuilder::BuildSettings )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBuilderSettings, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( MeshBuilderSettings ).
        def( pybind11::init<>() ).
        def_readwrite( "region", &MeshBuilder::BuildSettings::region, "if region is given then on input it contains the faces to be added, and on output the faces failed to be added" ).
        def_readwrite( "shiftFaceId", &MeshBuilder::BuildSettings::shiftFaceId, "this value to be added to every faceId before its inclusion in the topology" ).
        def_readwrite( "allowNonManifoldEdge", &MeshBuilder::BuildSettings::allowNonManifoldEdge, "whether to permit non-manifold edges in the resulting topology" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBuilder, []( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( ThreeVertIds ).
        def( pybind11::init( [] ( VertId v0, VertId v1, VertId v2 ) -> ThreeVertIds
        {
            return { v0, v1, v2 };
        } ) ).
        def( "__getitem__", [] ( const ThreeVertIds& self, int key )->VertId
        {
            return self[key];
        } ).
        def( "__setitem__", [] ( ThreeVertIds& self, int key, VertId val )
        {
            self[key] = val;
        } ).
        def( "__len__", &ThreeVertIds::size ).
        def( "__iter__", [] ( ThreeVertIds& self )
        {
            return pybind11::make_iterator<pybind11::return_value_policy::reference_internal, ThreeVertIds::iterator, ThreeVertIds::iterator, ThreeVertIds::value_type&>(
                self.begin(), self.end() );
        }, pybind11::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */ );

    pybind11::class_<Triangulation>( m, "Triangulation" ).
        def( pybind11::init<>() ).
        def_readwrite( "vec", &Triangulation::vec_ );

    m.def( "topologyFromTriangles",
        ( MeshTopology( * )( const Triangulation&, const MeshBuilder::BuildSettings& ) )& topologyFromTriangles,
        pybind11::arg( "triangulation" ), pybind11::arg_v( "settings", MeshBuilder::BuildSettings(), "MeshBuilderSettings()" ),
        "construct mesh topology from a set of triangles with given ids;\n"
        "if skippedTris is given then it receives all input triangles not added in the resulting topology" );

    m.def( "uniteCloseVertices", &MR::MeshBuilder::uniteCloseVertices,
        pybind11::arg( "mesh" ), pybind11::arg( "closeDist" ), pybind11::arg( "uniteOnlyBd" ) = true, pybind11::arg( "optionalVertOldToNew" ) = nullptr,
        "the function finds groups of mesh vertices located closer to each other than closeDist, and unites such vertices in one;\n"
        "then the mesh is rebuilt from the remaining triangles\n"
        "\toptionalVertOldToNew is the mapping of vertices: before -> after\n"
        "\tuniteOnlyBd if true then only boundary vertices can be united, all internal vertices (even close ones) will remain\n"
        "returns the number of vertices united, 0 means no change in the mesh" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorThreeVertIds, ThreeVertIds )

Mesh pythonCopyMeshFunction( const Mesh& mesh )
{
    return mesh;
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, PackMapping, MR::PackMapping )
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, Mesh, MR::Mesh )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Mesh, [] ( pybind11::module_& m )
{
    MR_PYTHON_CUSTOM_CLASS( PackMapping ).doc() =
        "Not fully exposed, for now dummy class";

    MR_PYTHON_CUSTOM_CLASS( Mesh ).
        def( pybind11::init<>() ).
        def( "computeBoundingBox", ( Box3f( Mesh::* )( const FaceBitSet*, const AffineXf3f* ) const )& Mesh::computeBoundingBox,
            pybind11::arg( "region" ) = nullptr, pybind11::arg( "toWorld" ) = nullptr,
            "passes through all given faces (or whole mesh if region == null) and finds the minimal bounding box containing all of them\n"
            "if toWorld transformation is given then returns minimal bounding box in world space" ).
        def( "getBoundingBox", &Mesh::getBoundingBox,
            "returns the bounding box containing all valid vertices (implemented via getAABBTree())\n"
            "this bounding box is insignificantly bigger that minimal box due to AABB algorithms precision" ).
        def( "area", ( double( Mesh::* )( const FaceBitSet* fs )const )& Mesh::area, pybind11::arg( "fs" ) = nullptr, "this version returns the area of whole mesh if argument is nullptr" ).
        def( "volume", &Mesh::volume, pybind11::arg( "region" ) = nullptr,
            "returns volume of closed mesh region, if region is not closed DBL_MAX is returned\n"
            "if region is nullptr - whole mesh is region" ).
        def( "pack", &Mesh::pack, pybind11::arg( "outFmap" ) = nullptr, pybind11::arg( "outVmap" ) = nullptr, pybind11::arg( "outEmap" ) = nullptr, pybind11::arg( "rearrangeTriangles" ) = false,
            "tightly packs all arrays eliminating lone edges and invalid face, verts and points,\n"
            "optionally returns mappings: old.id -> new.id" ).
        def( "packOptimally", &Mesh::packOptimally, pybind11::arg( "preserveAABBTree" ) = true,
            "packs tightly and rearranges vertices, triangles and edges to put close in space elements in close indices\n"
            "\tpreserveAABBTree whether to keep valid mesh's AABB tree after return (it will take longer to compute and it will occupy more memory)" ).
        def( "deleteFaces", &Mesh::deleteFaces, pybind11::arg( "fs" ), pybind11::arg( "keepEdges" ) = nullptr,
            "deletes multiple given faces, also deletes adjacent edges and vertices if they were not shared by remaining faces ant not in keepFaces" ).
        def( "discreteMeanCurvature", ( float( Mesh::* )( VertId ) const ) &Mesh::discreteMeanCurvature, pybind11::arg( "v" ),
            "computes discrete mean curvature in given vertex measures in length^-1;\n"
            "0 for planar regions, positive for convex surface, negative for concave surface" ).
        def_readwrite( "topology", &Mesh::topology ).
        def_readwrite( "points", &Mesh::points ).
        def( "triPoint", ( Vector3f( Mesh::* )( const MeshTriPoint& )const )& Mesh::triPoint, pybind11::arg( "p" ), "returns interpolated coordinates of given point" ).
        def( "edgePoint", ( Vector3f( Mesh::* )( EdgeId, float )const )&Mesh::edgePoint, pybind11::arg( "e" ), pybind11::arg( "f" ), "Returns a point on the edge: origin point for f=0 and destination point for f=1." ).\
        def( "edgePoint", ( Vector3f( Mesh::* )( const MeshEdgePoint& )const )& Mesh::edgePoint, pybind11::arg( "ep" ), "returns a point on the edge: origin point for f=0 and destination point for f=1" ).
        def( "invalidateCaches", &Mesh::invalidateCaches, pybind11::arg( "pointsChanged" ) = true, "Invalidates caches (e.g. aabb-tree) after a change in mesh geometry or topology" ).
        def( "transform", ( void( Mesh::* ) ( const AffineXf3f&, const VertBitSet* ) )& Mesh::transform, pybind11::arg( "xf" ), pybind11::arg( "region" ) = nullptr,
             "applies given transformation to specified vertices\n"
             "if region is nullptr, all valid mesh vertices are used" ).

        def( "leftNormal", &Mesh::leftNormal, pybind11::arg( "e" ), "computes triangular face normal from its vertices" ).
        def( "normal", ( Vector3f( Mesh::* )( FaceId )const )&Mesh::normal, pybind11::arg( "f" ), "computes triangular face normal from its vertices" ).
        def( "normal", ( Vector3f( Mesh::* )( VertId )const )&Mesh::normal, pybind11::arg( "v" ), "computes normal in a vertex using sum of directed areas of neighboring triangles" ).
        def( "normal", ( Vector3f( Mesh::* )( const MeshTriPoint & )const )&Mesh::normal, pybind11::arg( "p" ), "computes normal in three vertices of p's triangle, then interpolates them using barycentric coordinates" ).

        def( "pseudonormal", ( Vector3f( Mesh::* )( VertId, const FaceBitSet* )const )&Mesh::pseudonormal,
            pybind11::arg( "v" ), pybind11::arg( "region" ) = nullptr, "computes pseudo-normals for signed distance calculation at vertex, only region faces will be considered" ).
        def( "pseudonormal", ( Vector3f( Mesh::* )( UndirectedEdgeId, const FaceBitSet* )const )&Mesh::pseudonormal,
            pybind11::arg( "e" ), pybind11::arg( "region" ) = nullptr, "computes pseudo-normals for signed distance calculation at edge (middle of two face normals), only region faces will be considered" ).
        def( "pseudonormal", ( Vector3f( Mesh::* )( const MeshTriPoint &, const FaceBitSet* )const )&Mesh::pseudonormal,
            pybind11::arg( "p" ), pybind11::arg( "region" ) = nullptr, "computes pseudo-normals for signed distance calculation in corresponding face/edge/vertex, only region faces will be considered; unlike normal( MeshTriPoint ), this is not a smooth function" ).

        def( "splitEdge", ( EdgeId( Mesh::* )( EdgeId, const Vector3f&, FaceBitSet*, FaceHashMap* ) )& Mesh::splitEdge,
            pybind11::arg( "e" ), pybind11::arg( "newVertPos" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given edge on two parts:\n"
            "dest(returned-edge) = org(e) - newly created vertex,\n"
            "org(returned-edge) = org(e-before-split),\n"
            "dest(e) = dest(e-before-split)\n"
            "\tleft and right faces of given edge if valid are also subdivided on two parts each;\n"
            "\tif left or right faces of the original edge were in the region, then include new parts of these faces in the region\n"
            "\tnew2Old - receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "splitEdge", ( EdgeId( Mesh::* )( EdgeId, FaceBitSet*, FaceHashMap* ) )& Mesh::splitEdge,
            pybind11::arg( "e" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given edge on two equal parts:\n"
            "dest(returned-edge) = org(e) - newly created vertex,\n"
            "org(returned-edge) = org(e-before-split),\n"
            "dest(e) = dest(e-before-split)\n"
            "\tleft and right faces of given edge if valid are also subdivided on two parts each;\n"
            "\tif left or right faces of the original edge were in the region, then include new parts of these faces in the region\n"
            "\tnew2Old - receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "splitFace", ( VertId( Mesh::* )( FaceId, const Vector3f&, FaceBitSet*, FaceHashMap* ) )& Mesh::splitFace,
            pybind11::arg( "f" ), pybind11::arg( "newVertPos" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given triangle on three triangles, introducing new vertex with given coordinates and connecting it to original triangle vertices;\n"
            "if region is given, then it must include (f) and new faces will be added there as well\n"
            "\tnew2Old receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "splitFace", ( VertId( Mesh::* )( FaceId, FaceBitSet*, FaceHashMap* ) )& Mesh::splitFace,
            pybind11::arg( "f" ), pybind11::arg( "region" ) = nullptr, pybind11::arg( "new2Old" ) = nullptr,
            "split given triangle on three triangles, introducing new vertex in the centroid of original triangle and connecting it to original triangle vertices;\n"
            "if region is given, then it must include (f) and new faces will be added there as well\n"
            "\tnew2Old receive mapping from newly appeared triangle to its original triangle (part to full)" ).

        def( "addPartByMask", ( void( Mesh::* )( const Mesh&, const FaceBitSet&, const PartMapping& ) )& Mesh::addPartByMask,
            pybind11::arg( "from" ), pybind11::arg( "fromFaces" ) = nullptr, pybind11::arg_v( "map", PartMapping(), "PartMapping()" ),
            "appends mesh (from) in addition to this mesh: creates new edges, faces, verts and points\n"
            "copies only portion of (from) specified by fromFaces" ).

        def( pybind11::self == pybind11::self, "compare that two meshes are exactly the same" );

    m.def( "copyMesh", &pythonCopyMeshFunction, pybind11::arg( "mesh" ), "returns copy of input mesh" );
} )

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, MeshPart, MR::MeshPart )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshPart, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( MeshPart ).doc() =
        "stores reference on whole mesh (if region is nullptr) or on its part (if region pointer is valid)";
    MR_PYTHON_CUSTOM_CLASS( MeshPart ).
        def( pybind11::init<const Mesh&, const FaceBitSet*>(), pybind11::arg( "mesh" ), pybind11::arg( "region" ) = nullptr ).
        def_readwrite( "region", &MeshPart::region, "nullptr here means whole mesh" );

    pybind11::implicitly_convertible<const Mesh&, MeshPart>();
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, RegionBoundary, [] ( pybind11::module_& m )
{
    m.def( "findLeftBoundary", ( std::vector<EdgeLoop>( * )( const MeshTopology&, const FaceBitSet* ) )& findLeftBoundary,
        pybind11::arg( "topology" ), pybind11::arg( "region" ) = nullptr,
        "returns all region boundary loops;\n"
        "every loop has region faces on the left, and not-region faces or holes on the right" );

    m.def( "findRightBoundary", ( std::vector<EdgeLoop>( * )( const MeshTopology&, const FaceBitSet* ) )& findRightBoundary,
        pybind11::arg( "topology" ), pybind11::arg( "region" ) = nullptr,
        "returns all region boundary loops;\n"
        "every loop has region faces on the right, and not-region faces or holes on the left" );

    m.def( "getIncidentVerts", ( VertBitSet( * )( const MeshTopology&, const FaceBitSet& ) )& getIncidentVerts,
        pybind11::arg( "topology" ), pybind11::arg( "faces" ), "composes the set of all vertices incident to given faces" );
    m.def( "getInnerVerts", ( VertBitSet( * )( const MeshTopology&, const FaceBitSet& ) )& getInnerVerts,
        pybind11::arg( "topology" ), pybind11::arg( "faces" ), "composes the set of all vertices with all their faces in given set" );

    m.def( "getIncidentFaces", ( FaceBitSet( * )( const MeshTopology&, const VertBitSet& ) )& getIncidentFaces,
        pybind11::arg( "topology" ), pybind11::arg( "verts" ), "composes the set of all faces incident to given vertices" );
    m.def( "getInnerFaces", ( FaceBitSet( * )( const MeshTopology&, const VertBitSet& ) )& getInnerFaces,
        pybind11::arg( "topology" ), pybind11::arg( "verts" ), "composes the set of all faces with all their vertices in given set" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshComponents, [] ( pybind11::module_& m )
{
    pybind11::enum_<MeshComponents::FaceIncidence>( m, "FaceIncidence", "Face incidence type" ).
        value( "PerEdge", MeshComponents::FaceIncidence::PerEdge, "face can have neighbor only via edge" ).
        value( "PerVertex", MeshComponents::FaceIncidence::PerVertex, "face can have neighbor via vertex" );

    m.def( "getAllComponentsVerts", &MeshComponents::getAllComponentsVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "gets all connected components of mesh part" );

    m.def( "getAllComponents", []( const MeshPart& meshPart, MeshComponents::FaceIncidence incidence )
        { return MeshComponents::getAllComponents( meshPart, incidence ); },
        pybind11::arg( "meshPart" ),
        pybind11::arg_v( "incidence", MeshComponents::FaceIncidence::PerEdge, "FaceIncidence.PerEdge" ),
        "gets all connected components of mesh part" );

     m.def( "getComponent", []( const MeshPart& meshPart, FaceId id, MeshComponents::FaceIncidence incidence )
        { return MeshComponents::getComponent( meshPart, id, incidence ); },
        pybind11::arg( "meshPart" ),
        pybind11::arg( "faceId" ),
        pybind11::arg_v( "incidence", MeshComponents::FaceIncidence::PerEdge, "FaceIncidence.PerEdge" ),
        "returns one connected component containing given face,\n"
        "not effective to call more than once, if several components are needed use getAllComponents" );

    m.def( "getComponentVerts", &MeshComponents::getComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "vertId" ),
        pybind11::arg( "region" ) = nullptr,
        "returns one connected component containing given vertex,\n"
        "not effective to call more than once, if several components are needed use getAllComponentsVerts" );

     m.def( "getLargestComponent", []( const MeshPart& meshPart, MeshComponents::FaceIncidence incidence )
        { return MeshComponents::getLargestComponent( meshPart, incidence ); },
        pybind11::arg( "meshPart" ),
        pybind11::arg_v( "incidence", MeshComponents::FaceIncidence::PerEdge, "FaceIncidence.PerEdge" ),
        "returns largest by surface area component" );

    m.def( "getLargestComponentVerts", &MeshComponents::getLargestComponentVerts,
        pybind11::arg( "mesh" ),
        pybind11::arg( "region" ) = nullptr,
        "returns largest by number of elements component" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorMesh, Mesh )
MR_ADD_PYTHON_VEC( mrmeshpy, vectorConstMeshPtr, const Mesh* )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FillHole, [] ( pybind11::module_& m )
{
    pybind11::enum_<FillHoleParams::MultipleEdgesResolveMode>( m, "FillHoleParamsMultipleEdgesResolveMode" ).
        value( "None", FillHoleParams::MultipleEdgesResolveMode::None, "do not avoid multiple edges" ).
        value( "Simple", FillHoleParams::MultipleEdgesResolveMode::Simple, "avoid creating edges that already exist in topology (default)" ).
        value( "Strong", FillHoleParams::MultipleEdgesResolveMode::Strong,
            "makes additional efforts to avoid creating multiple edges,\n"
            "in some rare cases it is not possible (cases with extremely bad topology),\n"
            "if you faced one try to use duplicateMultiHoleVertices before fillHole" );

    pybind11::class_<FillHoleMetric>( m, "FillHoleMetric", "This is struct used as optimization metric of fillHole and buildCylinderBetweenTwoHoles functions" ).
        def( pybind11::init<>() );

    m.def( "getPlaneFillMetric", &getPlaneFillMetric, pybind11::arg( "mesh" ), pybind11::arg( "e" ),
        "Same as getCircumscribedFillMetric, but with extra penalty for the triangles having\n"
        "normals looking in the opposite side of plane containing left of (e)." );

    m.def( "getEdgeLengthFillMetric", &getEdgeLengthFillMetric, pybind11::arg( "mesh" ),
        "Simple metric minimizing the sum of all edge lengths" );

    m.def( "getEdgeLengthStitchMetric", &getEdgeLengthStitchMetric, pybind11::arg( "mesh" ),
        "Forbids connecting vertices from the same hole\n"
        "Simple metric minimizing edge length" );

    m.def( "getCircumscribedMetric", &getCircumscribedMetric, pybind11::arg( "mesh" ),
        "This metric minimizes the sum of circumcircle radii for all triangles in the triangulation.\n"
        "It is rather fast to calculate, and it results in typically good triangulations." );

    m.def( "getUniversalMetric", &getUniversalMetric, pybind11::arg( "mesh" ),
        "This metric minimizes the maximal dihedral angle between the faces in the triangulation\n"
        "and on its boundary, and it avoids creating too degenerate triangles;\n"
        " for planar holes it is the same as getCircumscribedMetric" );

    m.def( "getComplexFillMetric", &getComplexFillMetric, pybind11::arg( "mesh" ), pybind11::arg( "e" ),
        "This metric minimizes the sum of triangleMetric for all triangles in the triangulation\n"
        "plus the sum edgeMetric for all edges inside and on the boundary of the triangulation.\n"
        "Where\n"
        "triangleMetric is proportional to weighted triangle area and triangle aspect ratio\n"
        "edgeMetric grows with angle between triangles as ( ( 1 - cos( x ) ) / ( 1 + cos( x ) ) ) ^ 4." );

    m.def( "getMinAreaMetric", &getMinAreaMetric, pybind11::arg( "mesh" ),
        "This metric is for triangulation construction with minimal summed area of triangles.\n"
        "Warning: this metric can produce degenerated triangles" );

    pybind11::class_<FillHoleParams>( m, "FillHoleParams", "Structure has some options to control fillHole" ).
        def( pybind11::init<>() ).
        def_readwrite( "metric", &FillHoleParams::metric, "Specifies triangulation metric\n""default for fillHole: getCircumscribedFillMetric" ).
        def_readwrite( "multipleEdgesResolveMode", &FillHoleParams::multipleEdgesResolveMode ).
        def_readwrite( "makeDegenerateBand", &FillHoleParams::makeDegenerateBand,
            "If true creates degenerate faces band around hole to have sharp angle visualization\n"
            "warning: This flag bad for result topology, most likely you do not need it" ).
        def_readwrite( "maxPolygonSubdivisions", &FillHoleParams::maxPolygonSubdivisions, "The maximum number of polygon subdivisions on a triangle and two smaller polygons,\n""must be 2 or larger" ).
        def_readwrite( "outNewFaces", &FillHoleParams::outNewFaces, "If not nullptr accumulate new faces" );

    pybind11::class_<StitchHolesParams>( m, "StitchHolesParams", "Structure has some options to control buildCylinderBetweenTwoHoles" ).
        def( pybind11::init<>() ).
        def_readwrite( "metric", &StitchHolesParams::metric,
            "Specifies triangulation metric\n"
            "default for buildCylinderBetweenTwoHoles: getComplexStitchMetric").
        def_readwrite( "outNewFaces", &StitchHolesParams::outNewFaces, "If not nullptr accumulate new faces" );

    m.def( "fillHole", &fillHole,
        pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg_v( "params", FillHoleParams(), "FillHoleParams()" ),
        "Fills given hole represented by one of its edges (having no valid left face),\n"
        "uses fillHoleTrivially if cannot fill hole without multiple edges,\n"
        "default metric: CircumscribedFillMetric\n"
        "\tmesh - mesh with hole\n"
        "\ta - EdgeId which represents hole\n"
        "\tparams - parameters of hole filling" );

    m.def( "buildCylinderBetweenTwoHoles", ( void ( * )( Mesh&, EdgeId, EdgeId, const StitchHolesParams& ) )& buildCylinderBetweenTwoHoles,
        pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg_v( "params", StitchHolesParams(), "StitchHolesParams()" ),
        "Build cylindrical patch to fill space between two holes represented by one of their edges each,\n"
        "default metric: ComplexStitchMetric\n"
        "\tmesh - mesh with hole\n"
        "\ta - EdgeId which represents 1st hole\n"
        "\tb - EdgeId which represents 2nd hole\n"
        "\tparams - parameters of holes stitching" );

    m.def( "buildCylinderBetweenTwoHoles", ( bool ( * )( Mesh&, const StitchHolesParams& ) )& buildCylinderBetweenTwoHoles,
       pybind11::arg( "mesh" ), pybind11::arg_v( "params", StitchHolesParams(), "StitchHolesParams()" ),
       "this version finds holes in the mesh by itself and returns false if they are not found" );

    m.def( "makeBridgeEdge", & makeBridgeEdge,
        pybind11::arg( "topology" ), pybind11::arg( "a" ), pybind11::arg( "b" ),
        "creates a new bridge edge between origins of two boundary edges a and b (both having no valid left face);\n"
        "Returns invalid id if bridge cannot be created because otherwise multiple edges appear \n"
        "\ttopology - mesh topology\n"
        "\ta - first EdgeId\n"
        "\tb - second EdgeId\n" );

    m.def( "makeBridge", & makeBridge,
        pybind11::arg( "topology" ), pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg_v( "outNewFaces", nullptr, "nullptr"),
        "creates a bridge between two boundary edges a and b (both having no valid left face);\n"
        "bridge consists of two triangles in general or of one triangle if a and b are neighboring edges on the boundary;\n"
        "return false if bridge cannot be created because otherwise multiple edges appear\n"
        "\ttopology - mesh topology\n"
        "\ta - first EdgeId\n"
        "\tb - second EdgeId\n"
        "\toutNewFaces - FaceBitSet to store new triangles\n");
} )

Mesh pythonMergeMeshes( const pybind11::list& meshes )
{
    Mesh res;
    for ( int i = 0; i < pybind11::len( meshes ); ++i )
        res.addPart( pybind11::cast<Mesh>( meshes[i] ) );
    return res;
}

FaceBitSet getFacesByMinEdgeLength( const Mesh& mesh, float minLength )
{
    using namespace MR;
    FaceBitSet resultFaces( mesh.topology.getValidFaces().size() );
    float minLengthSq = minLength * minLength;
    for ( auto ue : undirectedEdges( mesh.topology ) )
    {
        if ( mesh.edgeLengthSq( ue ) > minLengthSq )
        {
            auto l = mesh.topology.left( ue );
            auto r = mesh.topology.right( ue );
            if ( l )
                resultFaces.set( l );
            if ( r )
                resultFaces.set( r );
        }
    }
    return resultFaces;
}

MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, FaceFace, MR::FaceFace )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SimpleFunctions, [] ( pybind11::module_& m )
{
    m.def( "computePerVertNormals", &computePerVertNormals, pybind11::arg( "mesh" ), "returns a vector with vertex normals in every element for valid mesh vertices" );
    m.def( "computePerVertPseudoNormals", &computePerVertPseudoNormals, pybind11::arg( "mesh" ), "returns a vector with vertex pseudonormals in every element for valid mesh vertices" );
    m.def( "computePerFaceNormals", &computePerFaceNormals, pybind11::arg( "mesh" ), "returns a vector with face-normal in every element for valid mesh faces" );
    m.def( "mergeMeshes", &pythonMergeMeshes, pybind11::arg( "meshes" ), "merge python list of meshes to one mesh" );
    m.def( "getFacesByMinEdgeLength", &getFacesByMinEdgeLength, pybind11::arg( "mesh" ), pybind11::arg( "minLength" ), "return faces with at least one edge longer than min edge length" );
    m.def( "buildBottom", &buildBottom, pybind11::arg( "mesh" ), pybind11::arg( "a" ), pybind11::arg( "dir" ), pybind11::arg( "holeExtension" ), pybind11::arg( "outNewFaces" ) = nullptr,
        "adds cylindrical extension of given hole represented by one of its edges (having no valid left face)\n"
        "by adding new vertices located in lowest point of the hole -dir*holeExtension and 2 * number_of_hole_edge triangles;\n"
        "return: the edge of new hole opposite to input edge (a)" );

    m.def( "makeCube", &makeCube, pybind11::arg( "size" ) = Vector3f::diagonal( 1 ), pybind11::arg( "base" ) = Vector3f::diagonal( -0.5f ),
        "Base is \"lower\" corner of the cube coordinates" );

    pybind11::class_<SphereParams>( m, "SphereParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "radius", &SphereParams::radius ).
        def_readwrite( "numMeshVertices", &SphereParams::numMeshVertices );

    m.def( "makeSphere", &makeSphere, pybind11::arg( "params" ),
        "creates a mesh of sphere with irregular triangulation" );
    m.def( "makeUVSphere", &makeUVSphere,
        pybind11::arg( "radius" ) = 1.0f,
        pybind11::arg( "horisontalResolution" ) = 16, pybind11::arg( "verticalResolution" ) = 16,
        "Z is polar axis of this UVSphere" );

    m.def( "makeCylinder", &makeCylinder,
        pybind11::arg( "radius" ) = 0.1f,
        pybind11::arg( "length" ) = 1.0f,
        pybind11::arg( "resolution" ) = 16,
        "creates Z-looking cylinder with radius 'radius', height - 'length', its base have 'resolution' sides" );

    m.def( "makeCylinderAdvanced", &makeCylinderAdvanced,
        pybind11::arg( "radius0" ) = 0.1f,
        pybind11::arg( "radius1" ) = 0.1f,
        pybind11::arg( "start_angle" ) = 0.0f,
        pybind11::arg( "arc_size" ) = 2.0f * PI_F,
        pybind11::arg( "length" ) = 1.0f,
        pybind11::arg( "resolution" ) = 16 );

    m.def( "makeTorus", &makeTorus,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "Z is symmetry axis of this torus\n"
        "points - optional out points of main circle" );

    m.def( "makeOuterHalfTorus", &makeOuterHalfTorus,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus without inner half faces\n"
        "main application - testing fillHole and Stitch" );

    m.def( "makeTorusWithUndercut", &makeTorusWithUndercut,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadiusInner" ) = 0.1f, pybind11::arg( "secondaryRadiusOuter" ) = 0.2f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with inner protruding half as undercut\n"
        "main application - testing fixUndercuts" );

    m.def( "makeTorusWithSpikes", &makeTorusWithSpikes,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadiusInner" ) = 0.1f, pybind11::arg( "secondaryRadiusOuter" ) = 0.2f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with some handed-up points\n"
        "main application - testing fixSpikes and Relax" );

    m.def( "makeTorusWithComponents", &makeTorusWithComponents,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with empty sectors\n"
        "main application - testing Components" );

    m.def( "makeTorusWithSelfIntersections", &makeTorusWithSelfIntersections,
        pybind11::arg( "primaryRadius" ) = 1.0f, pybind11::arg( "secondaryRadius" ) = 0.1f,
        pybind11::arg( "primaryResolution" ) = 16, pybind11::arg( "secondaryResolution" ) = 16,
        pybind11::arg( "points" ) = nullptr,
        "creates torus with empty sectors\n"
        "main application - testing Components" );

    MR_PYTHON_CUSTOM_CLASS( FaceFace ).
        def( pybind11::init<>() ).
        def( pybind11::init<FaceId, FaceId>(), pybind11::arg( "a" ), pybind11::arg( "b" ) ).
        def_readwrite( "aFace", &FaceFace::aFace ).
        def_readwrite( "bFace", &FaceFace::bFace );

    m.def( "findSelfCollidingTriangles",
        decorateExpected( []( const MeshPart& mp, ProgressCallback cb ) { return findSelfCollidingTriangles( mp, cb ); } ),
        pybind11::arg( "mp" ), pybind11::arg( "cb" ) = ProgressCallback{}, "finds all pairs of colliding triangles from one mesh or a region" );

    m.def( "findSelfCollidingTrianglesBS",
        decorateExpected( []( const MeshPart& mp, ProgressCallback cb ) { return findSelfCollidingTrianglesBS( mp, cb ); } ),
        pybind11::arg( "mp" ), pybind11::arg( "cb" ) = ProgressCallback{}, "finds union of all self-intersecting faces" );

    m.def( "findCollidingTriangles", &findCollidingTriangles,
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "firstIntersectionOnly" ) = false,
        "finds all pairs of colliding triangles from two meshes or two mesh regions\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tfirstIntersectionOnly - if true then the function returns at most one pair of intersecting triangles and returns faster" );


    m.def( "makeDegenerateBandAroundRegion",
        [] ( Mesh& mesh, const FaceBitSet& faces ) { makeDegenerateBandAroundRegion( mesh, faces ); },
        pybind11::arg( "mesh" ), pybind11::arg( "region" ),
        "creates a band of degenerate faces along the border of the specified region and the rest of the mesh\n"
        "the function is useful for extruding the region without changing the existing faces and creating holes\n"
        "\tmesh - the target mesh\n"
        "\tregion - the region required to be separated by a band of degenerate faces" );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, vectorFaceFace, FaceFace )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshBoundary, [] ( pybind11::module_& m )
{
    m.def( "straightenBoundary", &straightenBoundary,
    pybind11::arg( "mesh" ), pybind11::arg( "bdEdge" ), pybind11::arg("minNeiNormalsDot"), pybind11::arg("maxTriAspectRatio"), pybind11::arg("newFaces") = nullptr,
    "Adds triangles along the boundary to straighten it.\n"
    "New triangle is added only if:\n"
        "1) aspect ratio of the new triangle is at most maxTriAspectRatio\n"
        "2) dot product of its normal with neighbor triangles is at least minNeiNormalsDot." );
} )
