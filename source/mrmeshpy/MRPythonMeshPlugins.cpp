#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBoolean.h"
#include "MRMesh/MRVDBConversions.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRMeshMetrics.h"
#include "MRMesh/MRTunnelDetector.h"
#include "MRMesh/MRSymbolMesh.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRFixUndercuts.h"
#include "MRMesh/MRMeshCollide.h"
#include "MRMesh/MRMeshDistance.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRSurfacePath.h"
#include "MRMesh/MRPlanarPath.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRMeshDelone.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRPositionVertsSmoothly.h"
#include "MRMesh/MRAlignTextToMesh.h"
#include "MRMesh/MRFaceFace.h"
#include "MRMesh/MRLaplacian.h"
#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MROffset.h"
#include "MRMesh/MRSurfaceDistance.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRExtractIsolines.h"
#include "MRMesh/MRContour.h"
#include "MRMesh/MRContoursStitch.h"
#include "MRMesh/MRMeshOverhangs.h"
#include "MRMesh/MRConvexHull.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRMovementBuildBody.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRFixSelfIntersections.h"
#include <pybind11/functional.h>
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

using namespace MR;

#ifndef MRMESH_NO_OPENVDB
// Fix self-intersections
void fixSelfIntersections( Mesh& mesh1, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    mesh1 = convert(gridA);
}

// Boolean
Mesh booleanSub( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA -= gridB;
    return convert(gridA);
}

Mesh booleanUnion( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA += gridB;
    return convert( gridA );
}

Mesh booleanIntersect( const Mesh& mesh1, const Mesh& mesh2, float voxelSize )
{
    MeshVoxelsConverter convert;
    convert.voxelSize = voxelSize;
    auto gridA = convert(mesh1);
    auto gridB = convert(mesh2);
    gridA *= gridB;
    return convert( gridA );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, VoxelBooleanBlock, [] ( pybind11::module_& m )
{
    m.def( "fixSelfIntersections", &fixSelfIntersections, pybind11::arg( "mesh" ), pybind11::arg( "voxelSize" ), "fix self-intersections by converting to voxels and back" );
    m.def( "voxelBooleanSubtract", &booleanSub, pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "voxelSize" ), "subtract mesh B from mesh A" );
    m.def( "voxelBooleanUnite", &booleanUnion, pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "voxelSize" ), "unite mesh A and mesh B" );
    m.def( "voxelBooleanIntersect", &booleanIntersect, pybind11::arg( "meshA" ), pybind11::arg( "meshB" ), pybind11::arg( "voxelSize" ), "intersect mesh A and mesh B" );

} )
#endif

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SelfIntersections, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::SelfIntersections::Settings::Method>( m, "FixSelfIntersectionMethod" ).
        value( "Relax", MR::SelfIntersections::Settings::Method::Relax, "Relax mesh around self-intersections" ).
        value( "CutAndFill", MR::SelfIntersections::Settings::Method::CutAndFill, "Cut and re-fill regions around self-intersections (may fall back to `Relax`)" );


    pybind11::class_<MR::SelfIntersections::Settings>( m, "FixSelfIntersectionSettings", "Setting set for mesh self-intersections fix" ).
        def( pybind11::init<>() ).
        def_readwrite( "method", &MR::SelfIntersections::Settings::method ).
        def_readwrite( "relaxIterations", &MR::SelfIntersections::Settings::relaxIterations, "Maximum relax iterations" ).
        def_readwrite( "maxExpand", &MR::SelfIntersections::Settings::maxExpand, "Maximum expand count (edge steps from self-intersecting faces), should be > 0" ).
        def_readwrite( "subdivideEdgeLen", &MR::SelfIntersections::Settings::subdivideEdgeLen,
            "Edge length for subdivision of holes covers (0.0f means auto)\n"
            "FLT_MAX to disable subdivision" );

    m.def( "localFixSelfIntersections", MR::decorateExpected( &MR::SelfIntersections::fix ),
        pybind11::arg( "mesh" ), pybind11::arg( "settings" ),
        "Finds and fixes self-intersections per component" );

    m.def( "localFindSelfIntersections", MR::decorateExpected( &MR::SelfIntersections::getFaces ),
        pybind11::arg( "mesh" ), pybind11::arg( "cb" ) = MR::ProgressCallback{},
        "Find all self-intersections faces component-wise" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, DegenerationsDetection, [] ( pybind11::module_& m )
{
    m.def( "detectTunnelFaces", []( const MeshPart & mp, float maxTunnelLength ) { return MR::detectTunnelFaces( mp, { .maxTunnelLength = maxTunnelLength } ).value(); },
        pybind11::arg( "mp" ), pybind11::arg( "maxTunnelLength" ), 
        "returns tunnels as a number of faces;\n"
        "if you remove these faces and patch every boundary with disk, then the surface will be topology equivalent to sphere" );

    m.def( "detectBasisTunnels", []( const MeshPart & mp ) { return MR::detectBasisTunnels( mp ).value(); },
        pybind11::arg( "mp" ), "detects all not-contractible-in-point and not-equivalent tunnel loops on the mesh" );

    m.def( "findDegenerateFaces", MR::decorateExpected( &MR::findDegenerateFaces ),
        pybind11::arg( "mp" ), pybind11::arg( "criticalAspectRatio" ) = FLT_MAX, pybind11::arg( "cb" ) = ProgressCallback{},
        "finds faces which aspect ratio >= criticalAspectRatio" );

    m.def( "fixMultipleEdges", ( void( * )( MR::Mesh& ) )& MR::fixMultipleEdges,
        pybind11::arg( "mesh" ), "finds and resolves multiple edges" );

    m.def( "hasMultipleEdges", &MR::hasMultipleEdges,
        pybind11::arg( "topology" ), "finds multiple edges in the mesh" );

    m.def( "removeSpikes", &MR::removeSpikes,
        pybind11::arg( "mesh" ), pybind11::arg( "maxIterations" ), pybind11::arg( "minSumAngle" ), pybind11::arg( "region" ) = nullptr, 
        "applies at most given number of relaxation iterations the spikes detected by given threshold" );
} )

#ifndef MRMESH_NO_OPENVDB
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, FixUndercuts, [] ( pybind11::module_& m )
{
    m.def( "fixUndercuts", ( void ( * )( Mesh&, const FaceBitSet&, const Vector3f&, float, float ) )& MR::FixUndercuts::fixUndercuts,
        pybind11::arg( "mesh" ), pybind11::arg( "selectedArea" ), pybind11::arg( "upDirection" ), pybind11::arg( "voxelSize" ) = 0.0f, pybind11::arg( "bottomExtension" ) = 0.0f,
        "aChanges mesh:\n"
        "Fills all holes first, then:\n"
        "fixes undercuts (in selected area) via prolonging widest points down\n"
        "Requires to update RenderObject after using\n"
        "upDirection is in mesh space\n"
        "voxelSize -  size of voxel in mesh rasterization, precision grows with lower voxelSize\n"
        "bottomExtension - this parameter specifies how long should bottom prolongation be, if (bottomExtension <= 0) bottomExtension = 2*voxelSize\n"
        "\tif mesh is not closed this is used to prolong hole and make bottom\n"
        "\nif voxelSize == 0.0f it will be counted automaticly" );

    m.def( "fixUndercuts", ( void ( * )( Mesh&, const Vector3f&, float, float ) )& MR::FixUndercuts::fixUndercuts,
    pybind11::arg( "mesh" ), pybind11::arg( "upDirection" ), pybind11::arg( "voxelSize" ) = 0.0f, pybind11::arg( "bottomExtension" ) = 0.0f,
        "aChanges mesh:\n"
        "Fills all holes first, then:\n"
        "fixes undercuts via prolonging widest points down\n"
        "Requires to update RenderObject after using\n"
        "upDirection is in mesh space\n"
        "voxelSize -  size of voxel in mesh rasterization, precision grows with lower voxelSize\n"
        "bottomExtension - this parameter specifies how long should bottom prolongation be, if (bottomExtension <= 0) bottomExtension = 2*voxelSize\n"
        "\tif mesh is not closed this is used to prolong hole and make bottom\n"
        "\nif voxelSize == 0.0f it will be counted automaticly" );

    m.def( "findUndercuts", ( void( * )( const Mesh&, const Vector3f&, FaceBitSet& ) )& MR::FixUndercuts::findUndercuts,
        pybind11::arg( "mesh" ), pybind11::arg( "upDirection" ), pybind11::arg( "outUndercuts" ),
        "Adds to outUndercuts undercut faces" );
    m.def( "findUndercuts", ( void( * )( const Mesh&, const Vector3f&, VertBitSet& ) )& MR::FixUndercuts::findUndercuts,
        pybind11::arg( "mesh" ), pybind11::arg( "upDirection" ), pybind11::arg( "outUndercuts" ),
        "Adds to outUndercuts undercut vertices" );
} )
#endif

#ifndef MRMESH_NO_LABEL
// Text Mesh
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SymbolMeshParams, [] ( pybind11::module_& m )
{
    pybind11::class_<SymbolMeshParams>( m, "SymbolMeshParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "text", &SymbolMeshParams::text, "Text that will be made mesh" ).
        def_readwrite( "fontDetalization", &SymbolMeshParams::fontDetalization, "Detailization of Bezier curves on font glyphs" ).
        def_readwrite( "symbolsDistanceAdditionalOffset", &SymbolMeshParams::symbolsDistanceAdditionalOffset,
            "Additional offset between symbols\n"
            "X: In symbol size: 1.0f adds one \"space\", 0.5 adds half \"space\". Should be >= 0.0f\n"
            "Y: In symbol size: 1.0f adds one base height, 0.5 adds half base height" ).
        def_readwrite( "pathToFontFile", &TextMeshAlignParams::pathToFontFile, "Path to font file" );

    m.def( "createSymbolsMesh", MR::decorateExpected( &MR::createSymbolsMesh ), pybind11::arg( "params" ), "converts text string into Z-facing symbol mesh" );

    pybind11::class_<TextMeshAlignParams, SymbolMeshParams>( m, "TextAlignParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "startPoint", &TextMeshAlignParams::startPoint, "Start coordinate on mesh, represent lowest left corner of text" ).
        def_readwrite( "pivotPoint", &TextMeshAlignParams::pivotPoint, "Position of the startPoint in a text bounding box" ).
        def_readwrite( "direction", &TextMeshAlignParams::direction, "Direction of text" ).
        def_readwrite( "fontHeight", &TextMeshAlignParams::fontHeight, "Font height, meters" ).
        def_readwrite( "surfaceOffset", &TextMeshAlignParams::surfaceOffset, "Text mesh inside and outside offset of input mesh" ).
        def_readwrite( "textMaximumMovement", &TextMeshAlignParams::textMaximumMovement, "Maximum possible movement of text mesh alignment, meters" ).
        def_readwrite( "fontBasedSizeCalc", &TextMeshAlignParams::fontBasedSizeCalc, "If true then size of each symbol will be calculated from font height, otherwise - on bounding box of the text" );

    m.def( "alignTextToMesh", MR::decorateExpected( &MR::alignTextToMesh ), pybind11::arg( "mesh" ), pybind11::arg( "params" ),
        "Creates symbol mesh and aligns it to given surface" );
} )
#endif

MR_ADD_PYTHON_VEC( mrmeshpy, SurfacePath, MR::EdgePoint )
MR_ADD_PYTHON_VEC( mrmeshpy, SurfacePaths, MR::SurfacePath )

// Signed Distance
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshSignedDistanceResult, [] ( pybind11::module_& m )
{
    pybind11::class_<MeshSignedDistanceResult>( m, "MeshSignedDistanceResult" ).
        def( pybind11::init<>() ).
        def_readwrite( "a", &MeshSignedDistanceResult::a, "two closest points: from meshes A and B respectively" ).
        def_readwrite( "b", &MeshSignedDistanceResult::b, "two closest points: from meshes A and B respectively" ).
        def_readwrite( "signedDist", &MeshSignedDistanceResult::signedDist, "signed distance between a and b, positive if meshes do not collide" );

    m.def( "findSignedDistance", ( MeshSignedDistanceResult( * )( const MeshPart&, const MeshPart&, const AffineXf3f*, float ) )& MR::findSignedDistance,
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        "computes minimal distance between two meshes\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tupDistLimitSq - upper limit on the positive distance in question, if the real distance is larger than the function exists returning upDistLimitSq and no valid points" );

    m.def("findMaxDistanceSqOneWay",&MR::findMaxDistanceSqOneWay, 
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        "returns the maximum of the squared distances from each B-mesh vertex to A-mesh\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tmaxDistanceSq - upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq" );

    m.def( "findMaxDistanceSq", &MR::findMaxDistanceSq,
        pybind11::arg( "a" ), pybind11::arg( "b" ), pybind11::arg( "rigidB2A" ) = nullptr, pybind11::arg( "upDistLimitSq" ) = FLT_MAX,
        "returns the squared Hausdorff distance between two meshes, that is the maximum of squared distances from each mesh vertex to the other mesh (in both directions)\n"
        "\trigidB2A - rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation\n"
        "\tmaxDistanceSq - upper limit on the positive distance in question, if the real distance is larger than the function exists returning maxDistanceSq" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, PlaneSections, [] ( pybind11::module_& m )
{
    m.def( "extractPlaneSections", &extractPlaneSections,
        pybind11::arg( "mp" ), pybind11::arg( "plane" ),
        "extracts all plane sections of given mesh" );

    m.def( "planeSectionsToContours2f", &planeSectionsToContours2f,
        pybind11::arg( "mesh" ), pybind11::arg( "sections" ), pybind11::arg( "meshToPlane" ),
        "converts PlaneSections in 2D contours by computing coordinate of each point, applying given xf to it, and retaining only x and y" );

    m.def( "calcOrientedArea",
        ( double( * )( const Contour2f& ) )& calcOrientedArea<float, double>,
        pybind11::arg( "contour" ),
        ">0 for clockwise loop, < 0 for CCW loop" );
    
    m.def( "calcOrientedArea",
        ( Vector3f( * )( const Contour3f& ) )& calcOrientedArea<float, float>,
        pybind11::arg( "contour" ),
        "returns the vector with the magnitude equal to contour area, and directed to see the contour\n"
        "in ccw order from the vector tip" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MovementBody, [] ( pybind11::module_& m )
{
    m.def( "makeMovementBuildBody", &makeMovementBuildBody,
        pybind11::arg( "body" ), pybind11::arg( "trajectory" ), pybind11::arg_v( "params", MovementBuildBodyParams(), "MovementBuildBodyParams()" ),
        "makes mesh by moving `body` along `trajectory`\n"
        "if allowRotation rotate it in corners" );
} )


// Relax Mesh
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Relax, [] ( pybind11::module_& m )
{
    pybind11::class_<RelaxParams>( m, "RelaxParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "force", &RelaxParams::force, "speed of relaxing, typical values (0.0, 0.5]" ).
        def_readwrite( "iterations", &RelaxParams::iterations, "number of iterations" ).
        def_readwrite( "region", &RelaxParams::region, "region to relax" ).
        def_readwrite( "limitNearInitial", &RelaxParams::limitNearInitial, "if true then maximal displacement of each point during denoising will be limited" ).
        def_readwrite( "maxInitialDist", &RelaxParams::maxInitialDist, "maximum distance between a point and its position before relaxation, ignored if limitNearInitial = false" );

    pybind11::class_<MeshRelaxParams, RelaxParams>( m, "MeshRelaxParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "hardSmoothTetrahedrons", &MeshRelaxParams::hardSmoothTetrahedrons, "smooth tetrahedron verts (with complete three edges ring) to base triangle (based on its edges destinations)" );

    pybind11::enum_<MR::RelaxApproxType>( m, "RelaxApproxType", "Approximation strategy to use during `relaxApprox`" ).
        value( "Planar", MR::RelaxApproxType::Planar, "Projects the new neighborhood points onto a best approximating plane." ).
        value( "Quadric", MR::RelaxApproxType::Quadric, "Projects the new neighborhood points onto a best quadratic approximating." );
    
    pybind11::class_<MeshApproxRelaxParams, MeshRelaxParams>( m, "MeshApproxRelaxParams" ).
        def( pybind11::init<>() ).
        def_readwrite( "surfaceDilateRadius", &MeshApproxRelaxParams::surfaceDilateRadius, "Radius to find neighbors by surface. `0.0f - default = 1e-3 * sqrt(surface area)`" ).
        def_readwrite( "type", &MeshApproxRelaxParams::type, "" );

    m.def( "relax", ( bool( * )( Mesh&, const MeshRelaxParams&, ProgressCallback ) )& relax,
        pybind11::arg( "mesh" ), pybind11::arg_v( "params", MeshRelaxParams(), "MeshRelaxParams()" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "Applies the given number of relaxation iterations to the whole mesh (or some region if it is specified in the params).\n"
        "\tReturns `True` if the operation completed succesfully, and `False` if it was interrupted by the progress callback." );

    m.def( "relaxKeepVolume", &relaxKeepVolume,
        pybind11::arg( "mesh" ), pybind11::arg_v( "params", MeshRelaxParams(), "MeshRelaxParams()" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "Applies the given number of relaxation iterations to the whole mesh (or some region if it is specified in the params).\n"
        "do not really keeps volume but tries hard \n"
        "\tReturns `True` if the operation completed succesfully, and `False` if it was interrupted by the progress callback." );

    m.def( "relaxApprox", &relaxApprox,
        pybind11::arg( "mesh" ), pybind11::arg_v( "params", MeshApproxRelaxParams(), "MeshApproxRelaxParams()" ), pybind11::arg( "cb" ) = ProgressCallback{},
        "Applies the given number of relaxation iterations to the whole mesh (or some region if it is specified through the params).\n"
        "The algorithm looks at approx neighborhoods to smooth the mesh\n"
        "\tReturns `True` if the operation completed successfully, and `False` if it was interrupted by the progress callback." );

    m.def( "smoothRegionBoundary", &smoothRegionBoundary,
        pybind11::arg( "mesh" ), pybind11::arg( "regionFaces" ), pybind11::arg( "numIterations" ) = 4,
        "Given a region of faces on the mesh, moves boundary vertices of the region\n"
        "to make the region contour much smoother with minor optimization of mesh topology near region boundary.\n"
        "\tnumIterations - number of smoothing iterations. An even number is recommended due to oscillation of the algorithm" );
} )

// Subdivider Plugin
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SubdivideSettings, [] ( pybind11::module_& m )
{
    pybind11::class_<SubdivideSettings>( m, "SubdivideSettings" ).
        def( pybind11::init<>() ).
        def_readwrite( "maxEdgeLen", &SubdivideSettings::maxEdgeLen, "Maximal possible edge length created during decimation" ).
        def_readwrite( "maxEdgeSplits", &SubdivideSettings::maxEdgeSplits, "Maximum number of edge splits allowed" ).
        def_readwrite( "maxDeviationAfterFlip", &SubdivideSettings::maxDeviationAfterFlip, "Improves local mesh triangulation by doing edge flips if it does not make too big surface deviation" ).
        def_readwrite( "maxAngleChangeAfterFlip", &SubdivideSettings::maxAngleChangeAfterFlip, "Improves local mesh triangulation by doing edge flips if it does not change dihedral angle more than on this value. Unit: rad" ).
        def_readwrite( "criticalAspectRatioFlip", &SubdivideSettings::criticalAspectRatioFlip, "If this value is less than FLT_MAX then edge flips will ignore dihedral angle check if one of triangles has aspect ratio more than this value" ).
        def_readwrite( "region", &SubdivideSettings::region, "Region on mesh to be subdivided, it is updated during the operation" ).
        def_readwrite( "newVerts", &SubdivideSettings::newVerts, "New vertices appeared during subdivision will be added here" ).
        def_readwrite( "subdivideBorder", &SubdivideSettings::subdivideBorder,
            "If false do not touch border edges (cannot subdivide lone faces)\n"
            "use findRegionOuterFaces to find boundary faces" ).
        def_readwrite( "maxTriAspectRatio", &SubdivideSettings::maxTriAspectRatio,
            "The subdivision stops as soon as all triangles (in the region) have aspect ratio below or equal to this value" ).
        def_readwrite( "maxSplittableTriAspectRatio", &SubdivideSettings::maxSplittableTriAspectRatio,
            "An edge is subdivided only if both its left and right triangles have aspect ratio below or equal to this value. "
            "So this is a maximum aspect ratio of a triangle that can be split on two before Delone optimization. "
            "Please set it to a smaller value only if subdivideBorder==false, otherwise many narrow triangles can appear near border" ).
        def_readwrite( "smoothMode", &SubdivideSettings::smoothMode,
            "Puts new vertices so that they form a smooth surface together with existing vertices.\n"
            "This option works best for natural surfaces without sharp edges in between triangles" ).
        def_readwrite( "minSharpDihedralAngle", &SubdivideSettings::minSharpDihedralAngle,
            "In case of activated smoothMode, the smoothness is locally deactivated at the edges having dihedral angle at least this value" ).
        def_readwrite( "projectOnOriginalMesh", &SubdivideSettings::projectOnOriginalMesh,
            "If true, then every new vertex will be projected on the original mesh (before smoothing)" );

    m.def( "subdivideMesh", &MR::subdivideMesh,
        pybind11::arg( "mesh" ), pybind11::arg_v( "settings", MR::SubdivideSettings(), "SubdivideSettings()" ),
        "Split edges in mesh region according to the settings;\n"
        "return The total number of edge splits performed" );

} )

// Overhangs
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, Overhangs, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::FindOverhangsSettings>( m, "FindOverhangsSettings", "parameters for findOverhangs" ).
        def( pybind11::init<>() ).
        def_readwrite( "axis", &MR::FindOverhangsSettings::axis, "base axis marking the up direction" ).
        def_readwrite( "layerHeight", &MR::FindOverhangsSettings::layerHeight, "height of a layer" ).
        def_readwrite( "maxOverhangDistance", &MR::FindOverhangsSettings::maxOverhangDistance, "maximum overhang distance within a layer" ).
        def_readwrite( "hops", &MR::FindOverhangsSettings::hops, "number of hops used to smooth out the overhang regions (0 - disable smoothing)" ).
        def_readwrite( "xf", &MR::FindOverhangsSettings::xf, "mesh transform" );

    m.def( "findOverhangs", MR::decorateExpected( &MR::findOverhangs ),
        pybind11::arg( "mesh" ), pybind11::arg( "settings" ),
        "Find face regions that might create overhangs\n"
        "\tmesh - source mesh\n"
        "\tsettings - parameters\n"
        "\treturn face regions" );
} )

// Position Verts Smooth
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LaplacianEdgeWeightsParam, [] ( pybind11::module_& m )
{
    pybind11::enum_<EdgeWeights>( m, "LaplacianEdgeWeightsParam" ).
        value( "Unit", EdgeWeights::Unit, "all edges have same weight=1" ).
        value( "Cotan", EdgeWeights::Cotan, "edge weight depends on local geometry and uses cotangent values" ).
        value( "CotanTimesLength", EdgeWeights::CotanTimesLength, "[deprecated] edge weight is equal to edge length times cotangent weight" ).
        value( "CotanWithAreaEqWeight", EdgeWeights::CotanWithAreaEqWeight, "cotangent edge weights and equation weights inversely proportional to square root of local area" );

    m.def( "positionVertsSmoothly", &MR::positionVertsSmoothly,
        pybind11::arg( "mesh" ), pybind11::arg( "verts" ), pybind11::arg_v( "edgeWeightsType", MR::EdgeWeights::Cotan, "LaplacianEdgeWeightsParam.Cotan" ),
        pybind11::arg( "fixedSharpVertices" ) = nullptr,
        "Puts given vertices in such positions to make smooth surface both inside verts-region and on its boundary" );

    m.def( "positionVertsSmoothlySharpBd", &MR::positionVertsSmoothlySharpBd,
        pybind11::arg( "mesh" ), pybind11::arg( "verts" ), pybind11::arg( "vertShifts" ) = nullptr, pybind11::arg( "vertStabilizers" ) = nullptr,
        "Puts given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary\n"
        "\tmesh - source mesh\n"
        "\tverts - vertices to reposition. Cannot be all vertices of a connected component of the source mesh unless vertStabilizers are given\n"
        "\tvertShifts (optional) = additional shifts of each vertex relative to smooth position\n"
        "\vertStabilizers (optional) = per-vertex stabilizers: the more the value, the bigger vertex attraction to its original position"
    );
} )

// Position Verts Smooth (Inflation)
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, InflateSettings, [] ( pybind11::module_& m )
{
    pybind11::class_<InflateSettings>( m, "InflateSettings", "Controllable options for the inflation of a mesh region." ).
        def( pybind11::init<>() ).
        def_readwrite( "pressure", &InflateSettings::pressure,
            "The amount of pressure applied to mesh region: \n"
            "Positive pressure moves the vertices outward, negative moves them inward. \n"
            "Provided value should be in range of the [-region_diagonal, +region_diagonal]."
        ).
        def_readwrite( "iterations", &InflateSettings::iterations, 
            "The number of internal iterations (>=1) \n"
            "A larger number of iterations makes the performance slower, but the quality better"
        ).
        def_readwrite( "preSmooth", &InflateSettings::preSmooth, 
            "Smooths the area before starting inflation. \n"
            "Set to false only if the region is known to be already smooth"
        ).
        def_readwrite( "gradualPressureGrowth", &InflateSettings::gradualPressureGrowth, "whether to increase the pressure gradually during the iterations (recommended for best quality)" );
    
    m.def( "inflate", &MR::inflate,
        pybind11::arg( "mesh" ), pybind11::arg( "verts" ), pybind11::arg_v( "settings", InflateSettings(), "InflateSettings()" ),
        "Inflates (in one of two sides) given mesh region by"
        "putting given vertices in such positions to make smooth surface inside verts-region, but sharp on its boundary. \n"
        "\t verts must not include all vertices of a mesh connected component"
    );
} )

#ifndef MRMESH_NO_OPENVDB
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, MeshOffset, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::SignDetectionMode>( m, "SignDetectionMode", "How to determine the sign of distances from a mesh" ).
        value( "Unsigned",         MR::SignDetectionMode::Unsigned,         "unsigned distance, useful for bidirectional `Shell` offset" ).
        value( "OpenVDB",          MR::SignDetectionMode::OpenVDB,          "sign detection from OpenVDB library, which is good and fast if input geometry is closed" ).
        value( "ProjectionNormal", MR::SignDetectionMode::ProjectionNormal, "the sign is determined based on pseudonormal in closest mesh point (unsafe in case of self-intersections" ).
        value( "WindingRule",      MR::SignDetectionMode::WindingRule,      "ray intersection counter, significantly slower than ProjectionNormal and does not support holes in mesh" ).
        value( "HoleWindingRule",  MR::SignDetectionMode::HoleWindingRule,  "computes winding number generalization with support of holes in mesh, slower than WindingRule" );

    pybind11::class_<MR::OffsetParameters>( m, "OffsetParameters", "This struct represents parameters for offsetting with voxels conversions" ).
        def( pybind11::init<>() ).
        def_readwrite( "voxelSize", &MR::OffsetParameters::voxelSize,
            "Size of voxel in grid conversions\n"
            "if value is not positive, it is calculated automatically (mesh bounding box is divided to 5e6 voxels)" ).
        def_readwrite( "signDetectionMode", &MR::OffsetParameters::signDetectionMode, "The method to compute distance sign" );

    pybind11::class_<MR::SharpOffsetParameters, MR::OffsetParameters>( m, "SharpOffsetParameters" ).
        def( pybind11::init<>() ).
        def_readwrite( "outSharpEdges", &MR::SharpOffsetParameters::outSharpEdges, "if non-null then created sharp edges will be saved here" ).
        def_readwrite( "minNewVertDev", &MR::SharpOffsetParameters::minNewVertDev, "minimal surface deviation to introduce new vertex in a voxel, measured in voxelSize" ).
        def_readwrite( "maxNewRank2VertDev", &MR::SharpOffsetParameters::maxNewRank2VertDev, "maximal surface deviation to introduce new rank 2 vertex (on intersection of 2 planes), measured in voxelSize" ).
        def_readwrite( "maxNewRank3VertDev", &MR::SharpOffsetParameters::maxNewRank3VertDev, "maximal surface deviation to introduce new rank 3 vertex (on intersection of 3 planes), measured in voxelSize" ).
        def_readwrite( "maxOldVertPosCorrection", &MR::SharpOffsetParameters::maxOldVertPosCorrection,
            "correct positions of the input vertices using reference mesh by not more than this distance, measured in voxelSize;\n"
            "big correction can be wrong and result from self-intersections in the reference mesh" );

    pybind11::enum_<MR::GeneralOffsetParameters::Mode>( m, "GeneralOffsetParametersMode" ).
        value( "Smooth", MR::GeneralOffsetParameters::Mode::Smooth, "create mesh using dual marching cubes from OpenVDB library" ).
        value( "Standard", MR::GeneralOffsetParameters::Mode::Standard, "create mesh using standard marching cubes implemented in MeshLib" ).
        value( "Sharpening", MR::GeneralOffsetParameters::Mode::Sharpening, "create mesh using standard marching cubes with additional sharpening implemented in MeshLib" );

    pybind11::class_<MR::GeneralOffsetParameters, MR::SharpOffsetParameters>( m, "GeneralOffsetParameters", "allows the user to select in the parameters which offset algorithm to call" ).
        def( pybind11::init<>() ).
        def_readwrite( "mode", &MR::GeneralOffsetParameters::mode );

    m.def( "suggestVoxelSize", &MR::suggestVoxelSize, pybind11::arg( "mp" ), pybind11::arg( "approxNumVoxels" ), "computes size of a cubical voxel to get approximately given number of voxels during rasterization" );

    m.def( "generalOffsetMesh",
        MR::decorateExpected( [] ( const MR::MeshPart& mp, float offset, MR::GeneralOffsetParameters params )
    {
        if ( params.voxelSize <= 0 )
            params.voxelSize = suggestVoxelSize( mp, 5e6f );
        return MR::generalOffsetMesh( mp, offset, params );
    } ),
        pybind11::arg( "mp" ), pybind11::arg( "offset" ), pybind11::arg_v( "params", MR::GeneralOffsetParameters(), "GeneralOffsetParameters()" ),
        "Offsets mesh by converting it to voxels and back using one of three modes specified in the parameters" );

    m.def( "offsetMesh",
        MR::decorateExpected( []( const MR::MeshPart & mp, float offset, MR::OffsetParameters params )
            {
                if ( params.voxelSize <= 0 )
                    params.voxelSize = suggestVoxelSize( mp, 5e6f );
                return MR::offsetMesh( mp, offset, params );
            } ),
        pybind11::arg( "mp" ), pybind11::arg( "offset" ), pybind11::arg_v( "params", MR::OffsetParameters(), "OffsetParameters()" ),
        "Offsets mesh by converting it to voxels and back\n"
        "use Shell type for non closed meshes\n"
        "so result mesh is always closed" );

    m.def( "thickenMesh",
         MR::decorateExpected( [] ( const MR::Mesh& mesh, float offset, const MR::OffsetParameters & params0 )
    {
        MR::GeneralOffsetParameters params = { { params0 } };
        if ( params.voxelSize <= 0 )
            params.voxelSize = suggestVoxelSize( mesh, 5e6f );
        return MR::thickenMesh( mesh, offset, params );
    } ),
        pybind11::arg( "mesh" ), pybind11::arg( "offset" ), pybind11::arg_v( "params", MR::OffsetParameters(), "OffsetParameters()" ),
        "in case of positive offset, returns the mesh consisting of offset mesh merged with inversed original mesh (thickening mode);\n"
        "in case of negative offset, returns the mesh consisting of inversed offset mesh merged with original mesh (hollowing mode);\n"
        "if your input mesh is closed then please specify params.type == Offset, and you will get closed mesh on output;\n"
        "if your input mesh is open then please specify params.type == Shell, and you will get open mesh on output" );


    m.def( "doubleOffsetMesh", 
        MR::decorateExpected( [] ( const MR::MeshPart& mp, float offsetA, float offsetB, MR::OffsetParameters params )
    {
        if ( params.voxelSize <= 0 )
            params.voxelSize = suggestVoxelSize( mp, 5e6f );
        return MR::doubleOffsetMesh( mp, offsetA, offsetB, params );
    } ), 
        pybind11::arg( "mp" ), pybind11::arg( "offsetA" ), pybind11::arg( "offsetB" ), pybind11::arg_v( "params", MR::OffsetParameters(), "OffsetParameters()" ),
        "Offsets mesh by converting it to voxels and back two times\n"
        "only closed meshes allowed (only Offset mode)\n"
        "typically offsetA and offsetB have distinct signs" );
})
#endif

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, GeodesicPath, [] ( pybind11::module_& m )
{
    pybind11::enum_<MR::GeodesicPathApprox>( m, "GeodesicPathApprox", "Method of approximation" ).
        value( "DijkstraBiDir", MR::GeodesicPathApprox::DijkstraBiDir, "Bidirectional Dijkstra algorithm" ).
        value( "DijkstraAStar", MR::GeodesicPathApprox::DijkstraAStar, "Dijkstra algorithm with A* modification" ).
        value( "FastMarching", MR::GeodesicPathApprox::FastMarching, "Fast marching algorithm" );

    m.def( "computeGeodesicPath",
        MR::decorateExpected( &MR::computeGeodesicPath ),
        pybind11::arg( "mesh" ), pybind11::arg( "start" ), pybind11::arg( "end" ), pybind11::arg( "atype" ), pybind11::arg( "maxGeodesicIters") = 100,
        "Returns intermediate points of the geodesic path from start to end, where it crosses mesh edges"
    );

    m.def( "surfacePathLength", &MR::surfacePathLength,
        pybind11::arg( "mesh" ), pybind11::arg( "surfacePath" ),
        "Computes the length of surface path"
    );

    m.def( "computeSurfaceDistances", (MR::Vector<float, MR::VertId>(*)(const MR::Mesh&, const MeshTriPoint&, float maxDist, const VertBitSet*, int ) )&MR::computeSurfaceDistances,
        pybind11::arg( "mesh" ), pybind11::arg( "start" ), pybind11::arg( "maxDist" ) = FLT_MAX, pybind11::arg( "region" ) = nullptr, pybind11::arg( "maxVertUpdates" ) = 3,
        "Computes path distances in mesh vertices from given start point, stopping when maxDist is reached;\n"
        "considered paths can go either along edges or straightly within triangles" 
    );

    m.def( "computeSurfaceDistances", 
        ( MR::Vector<float, MR::VertId>( * )( const MR::Mesh&, const VertBitSet&, const VertBitSet&, float maxDist, const VertBitSet*, int ) )& MR::computeSurfaceDistances,
        pybind11::arg( "mesh" ), pybind11::arg( "startVertices" ), pybind11::arg( "targetVertices" ), pybind11::arg( "maxDist" ) = FLT_MAX, pybind11::arg( "region" ) = nullptr, pybind11::arg( "maxVertUpdates" ) = 3,
        "Computes path distances in mesh vertices from given start vertices, stopping when all targetVertices or maxDist is reached;\n"
        "considered paths can go either along edges or straightly within triangles"
    );
})

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ConvexHull, [] ( pybind11::module_& m )
{
    m.def( "makeConvexHull",  ( Mesh ( * ) ( const VertCoords&, const VertBitSet& ) )&  makeConvexHull, 
        pybind11::arg( "points" ), pybind11::arg( "validPoints" ),
        "Computes the Mesh of convex hull from given input points" );

    m.def( "makeConvexHull",  ( Mesh ( * ) ( const Mesh& ) )&  makeConvexHull, 
        pybind11::arg( "mesh" ),
        "Computes the Mesh of convex hull from given input `Mesh`" );

    m.def( "makeConvexHull",  ( Mesh ( * ) ( const PointCloud& ) )&  makeConvexHull, 
        pybind11::arg( "pointCloud" ),
        "Computes the Mesh of convex hull from given input `PointCloud`" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, ContourStitch, [] ( pybind11::module_& m )
{
    m.def( "stitchContours", &MR::stitchContours,
        pybind11::arg( "topology" ), pybind11::arg( "c0" ), pybind11::arg( "c1" ),
        "Merges the surface along corresponding edges of two contours, and deletes all vertices and edges from c1"
        "Requires both contours to:\n"
        "\t1) have equal size\n"
        "\t2) All edges of c0 with no left faces\n"
        "\t3) All edges of c1 have no right faces"
    );

    m.def( "cutAlongEdgeLoop", ( MR::EdgeLoop ( * ) ( MR::MeshTopology&, const MR::EdgeLoop& ) ) & MR::cutAlongEdgeLoop,
        pybind11::arg( "topology" ), pybind11::arg( "c0" ),
        "Given a closed loop of edges, splits the surface along that loop such that after return:\n"
        "\t1) Returned loop has the same size as input, with corresponding edges in same indexed elements of both\n"
        "\t2) All edges of `edgeLoop` have no left faces\n"
        "\t3) All returned edges have no right faces"
    );

    m.def( "cutAlongEdgeLoop", ( MR::EdgeLoop ( * ) ( MR::Mesh&, const MR::EdgeLoop& ) ) & MR::cutAlongEdgeLoop,
        pybind11::arg( "mesh" ), pybind11::arg( "c0" ),
        "Given a closed loop of edges, splits the surface along that loop such that after return:\n"
        "\t1) Returned loop has the same size as input, with corresponding edges in same indexed elements of both\n"
        "\t2) All edges of `edgeLoop` have no left faces\n"
        "\t3) All returned edges have no right faces"
        "\t4) Vertices of the given mesh are updated"
    );
} )
