#include "MRMeshBoolean.h"
#include "MRBooleanOperation.h"
#include "MRMesh.h"
#include "MRMeshCollidePrecise.h"
#include "MRIntersectionContour.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRTorus.h"
#include "MRMatrix3.h"
#include "MRAffineXf3.h"
#include "MRLog.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include "MRFillContour.h"
#include "MRPrecisePredicates3.h"
#include "MRRegionBoundary.h"
#include "MRMeshComponents.h"
#include "MRMeshCollide.h"
#include "MRCube.h"
#include "MRMeshBuilder.h"
#include "MRParallelFor.h"

namespace
{

using namespace MR;

enum class EdgeTriComponent
{
    Edge,
    Tri,
};

Vector3f findEdgeTriIntersectionPoint( const Mesh& edgeMesh, EdgeId edge, const Mesh& triMesh, FaceId tri,
                                       const CoordinateConverters& converters,
                                       const AffineXf3f* rigidB2A = nullptr,
                                       EdgeTriComponent meshBComponent = EdgeTriComponent::Edge )
{
    Vector3f ev0, ev1;
    ev0 = edgeMesh.orgPnt( edge );
    ev1 = edgeMesh.destPnt( edge );

    Vector3f fv0, fv1, fv2;
    triMesh.getTriPoints( tri, fv0, fv1, fv2 );

    if ( rigidB2A )
    {
        const auto& xf = *rigidB2A;
        switch ( meshBComponent )
        {
        case EdgeTriComponent::Edge:
            ev0 = xf( ev0 );
            ev1 = xf( ev1 );
            break;
        case EdgeTriComponent::Tri:
            fv0 = xf( fv0 );
            fv1 = xf( fv1 );
            fv2 = xf( fv2 );
            break;
        }
    }

    return findTriangleSegmentIntersectionPrecise( fv0, fv1, fv2, ev0, ev1, converters );
}

void gatherEdgeInfo( const MeshTopology& topology, EdgeId e, FaceBitSet& faces, VertBitSet& orgs, VertBitSet& dests )
{
    faces.set( topology.left( e ) );
    faces.set( topology.right( e ) );
    orgs.set( topology.org( e ) );
    dests.set( topology.dest( e ) );
}

OneMeshContours getOtherMeshContoursByHint( const OneMeshContours& aContours, const ContinuousContours& contours, 
    const AffineXf3f* rigidB2A = nullptr )
{
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();
    OneMeshContours bMeshContours = aContours;
    for ( int j = 0; j < bMeshContours.size(); ++j )
    {
        const auto& inCont = contours[j];
        auto& outCont = bMeshContours[j].intersections;
        assert( inCont.size() == outCont.size() );
        ParallelFor( inCont, [&] ( size_t i )
        {
            const auto& inInter = inCont[i];
            auto& outInter = outCont[i];
            outInter.primitiveId = inInter.isEdgeATriB ?
                std::variant<FaceId, EdgeId, VertId>( inInter.tri ) :
                std::variant<FaceId, EdgeId, VertId>( inInter.edge );
            if ( rigidB2A )
                outInter.coordinate = inverseXf( outCont[i].coordinate );
        } );
    }
    return bMeshContours;
}

}

namespace MR
{

BooleanResult booleanImpl( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation, const BooleanParameters& params, BooleanInternalParameters intParams );

BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation,
                       const AffineXf3f* rigidB2A /*= nullptr */, BooleanResultMapper* mapper /*= nullptr */, ProgressCallback cb )
{
    return boolean( meshA, meshB, operation, { .rigidB2A = rigidB2A, .mapper = mapper, .cb = cb } );
}

BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation,
                       const AffineXf3f* rigidB2A /*= nullptr */, BooleanResultMapper* mapper /*= nullptr */, ProgressCallback cb )
{
    return boolean( meshA, meshB, operation, { .rigidB2A = rigidB2A, .mapper = mapper, .cb = cb } );
}

BooleanResult boolean( const Mesh& meshA, const Mesh& meshB, BooleanOperation operation, const BooleanParameters& params /*= {} */ )
{
    bool needCutMeshA = operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB;
    bool needCutMeshB = operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA;
    tbb::task_group taskGroup;
    if ( needCutMeshA )
    {
        // build tree for input mesh for the cloned mesh to copy the tree,
        // this is important for many calls to Boolean for the same mesh to avoid tree construction on every call
        taskGroup.run( [&] ()
        {
            meshA.getAABBTree();
        } );
    }
    if ( needCutMeshB )
    {
        // build tree for input mesh for the cloned mesh to copy the tree,
        // this is important for many calls to Boolean for the same mesh to avoid tree construction on every call
        meshB.getAABBTree();
    }
    taskGroup.wait();
    return booleanImpl( Mesh( meshA ), Mesh( meshB ), operation, params, { .originalMeshA = &meshA,.originalMeshB = &meshB } );
}

BooleanResult boolean( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation, const BooleanParameters& params /*= {} */ )
{
    return booleanImpl( std::move( meshA ), std::move( meshB ), operation, params, {} );
}

Contours3f findIntersectionContours( const Mesh& meshA, const Mesh& meshB, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    auto converters = getVectorConverters( meshA, meshB, rigidB2A );
    auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, rigidB2A );
    auto contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
    return extractIntersectionContours( meshA, meshB, contours, converters, rigidB2A );
}

BooleanResult booleanImpl( Mesh&& meshA, Mesh&& meshB, BooleanOperation operation, const BooleanParameters& params, BooleanInternalParameters intParams )
{
    MR_TIMER;
    BooleanResult result;
    CoordinateConverters converters;
    PreciseCollisionResult intersections;
    ContinuousContours contours;

    bool needCutMeshA = operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB;
    bool needCutMeshB = operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA;

    converters = getVectorConverters( meshA, meshB, params.rigidB2A );

    auto loneCb = subprogress( params.cb, 0.0f, 0.8f );

    FaceMap new2orgSubdivideMapA;
    FaceMap new2orgSubdivideMapB;
    std::vector<int> prevLoneContoursIds;
    int iters = 0;
    const int cMaxFixLoneIterations = 100;
    bool aSubdivided = false;
    bool bSubdivided = false;
    for ( ;; iters++ )
    {
        // find intersections
        intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, params.rigidB2A );
        // order intersections
        contours = orderIntersectionContours( meshA.topology, meshB.topology, intersections );
        // find lone
        auto loneContoursIds = detectLoneContours( contours );

        if ( loneCb && !loneCb( ( std::log10( float( iters + 1 ) * 0.1f ) + 2.0f ) / 3.0f ) )
            return { .errorString = stringOperationCanceled() };

        if ( !loneContoursIds.empty() && ( loneContoursIds == prevLoneContoursIds || iters == cMaxFixLoneIterations ) )
        {
            // in some rare cases there are lone contours with zero area that cannot be resolved
            // they lead to infinite loop, so just try to remove them
            removeLoneContours( contours );
            break;
        }

        prevLoneContoursIds = loneContoursIds;

        // separate A lone from B lone
        ContinuousContours loneA;
        ContinuousContours loneB;
        for ( int i = 0; i < loneContoursIds.size(); ++i )
        {
            const auto& contour = contours[loneContoursIds[i]];
            if ( contour[0].isEdgeATriB )
                loneB.push_back( contour );
            else
                loneA.push_back( contour );
        }
        if ( loneContoursIds.empty() ||
            ( loneA.empty() && !needCutMeshB ) ||
            ( loneB.empty() && !needCutMeshA ) )
            break;
        // subdivide owners of lone
        if ( !loneA.empty() && needCutMeshA )
        {
            aSubdivided = true;
            auto loneIntsA = getOneMeshIntersectionContours( meshA, meshB, loneA, true, converters, params.rigidB2A );
            auto loneIntsAonB = getOneMeshIntersectionContours( meshA, meshB, loneA, false, converters, params.rigidB2A );
            removeLoneDegeneratedContours( meshB.topology, loneIntsA, loneIntsAonB );
            FaceMap new2orgLocalMap;
            FaceMap* mapPointer = params.mapper ? &new2orgLocalMap : nullptr;
            subdivideLoneContours( meshA, loneIntsA, mapPointer );
            if ( new2orgSubdivideMapA.size() < new2orgLocalMap.size() )
                new2orgSubdivideMapA.resize( new2orgLocalMap.size() );
            ParallelFor( new2orgLocalMap, [&] ( FaceId i )
            {
                if ( !new2orgLocalMap[i] )
                    return;
                FaceId refFace = new2orgLocalMap[i];
                if ( new2orgSubdivideMapA[refFace] )
                    refFace = new2orgSubdivideMapA[refFace];
                new2orgSubdivideMapA[i] = refFace;
            } );
        }
        if ( !loneB.empty() && needCutMeshB )
        {
            bSubdivided = true;
            auto loneIntsB = getOneMeshIntersectionContours( meshA, meshB, loneB, false, converters, params.rigidB2A );
            auto loneIntsBonA = getOneMeshIntersectionContours( meshA, meshB, loneB, true, converters, params.rigidB2A );
            removeLoneDegeneratedContours( meshA.topology, loneIntsB, loneIntsBonA );
            FaceMap new2orgLocalMap;
            FaceMap* mapPointer = params.mapper ? &new2orgLocalMap : nullptr;
            subdivideLoneContours( meshB, loneIntsB, mapPointer );
            if ( new2orgSubdivideMapB.size() < new2orgLocalMap.size() )
                new2orgSubdivideMapB.resize( new2orgLocalMap.size() );
            ParallelFor( new2orgLocalMap, [&] ( FaceId i )
            {
                if ( !new2orgLocalMap[i] )
                    return;
                FaceId refFace = new2orgLocalMap[i];
                if ( new2orgSubdivideMapB[refFace] )
                    refFace = new2orgSubdivideMapB[refFace];
                new2orgSubdivideMapB[i] = refFace;
            } );
        }
    }
    if ( iters == cMaxFixLoneIterations )
    {
        spdlog::warn( "Boolean: fix lone contours iteration limit reached." );
        result.errorString = "Fix lone contours iteration limit reached.";
        return result;
    }
    // clear intersections
    intersections = {};


    auto mainCb = subprogress( params.cb, 0.8f, 1.0f );
    if ( mainCb && !mainCb( 0.0f ) )
        return { .errorString = stringOperationCanceled() };

    std::vector<EdgePath> cutA, cutB;
    OneMeshContours meshAContours;
    OneMeshContours meshBContours;
    // original mesh is needed to properly sort edges intersections while cutting meshes
    // as far as meshes is going to be change in the same time we need to take copies before start cutting
    Mesh meshACopyBuffer; // (only used if meshes were not copied before, otherwise old copy will be used)
    Mesh meshBCopyBuffer;// (only used if meshes were not copied before, otherwise old copy will be used)
    std::unique_ptr<SortIntersectionsData> dataForA;
    std::unique_ptr<SortIntersectionsData> dataForB;

    tbb::task_group taskGroup;
    if ( needCutMeshA )
    {
        taskGroup.run( [&] ()
        {
            if ( needCutMeshB )
            {
                if ( bSubdivided || !intParams.originalMeshB )
                {
                    meshBCopyBuffer = meshB;
                    if ( !intParams.originalMeshB )
                        intParams.originalMeshB = &meshBCopyBuffer;
                }
                assert( intParams.originalMeshB );
                const Mesh* sortMeshPtr = bSubdivided ? &meshBCopyBuffer : intParams.originalMeshB;
                dataForA = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ *sortMeshPtr, contours, converters.toInt, params.rigidB2A, meshA.topology.vertSize(), false } );
            }
            else
                // B is stable so no need copy
                dataForA = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ meshB, contours, converters.toInt, params.rigidB2A, meshA.topology.vertSize(), false } );
        } );
    }

    if ( needCutMeshB )
    {
        if ( needCutMeshA )
        {
            if ( aSubdivided || !intParams.originalMeshA )
            {
                meshACopyBuffer = meshA;
                if ( !intParams.originalMeshA )
                    intParams.originalMeshA = &meshACopyBuffer;
            }
            assert( intParams.originalMeshA );
            const Mesh* sortMeshPtr = aSubdivided ? &meshACopyBuffer : intParams.originalMeshA;
            dataForB = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ *sortMeshPtr, contours, converters.toInt, params.rigidB2A, meshA.topology.vertSize(), true } );
        }
        else
            // A is stable so no need to copy
            dataForB = std::make_unique<SortIntersectionsData>( SortIntersectionsData{ meshA, contours, converters.toInt, params.rigidB2A, meshA.topology.vertSize(), true } );
    }
    taskGroup.wait();

    if ( needCutMeshA )
        meshAContours = getOneMeshIntersectionContours( meshA, meshB, contours, true, converters, params.rigidB2A );
    if ( needCutMeshB )
    {
        if ( needCutMeshA )
            meshBContours = getOtherMeshContoursByHint( meshAContours, contours, params.rigidB2A );
        else
            meshBContours = getOneMeshIntersectionContours( meshA, meshB, contours, false, converters, params.rigidB2A );
    }

    if ( mainCb && !mainCb( 0.33f ) )
        return { .errorString = stringOperationCanceled() };

    if ( params.outPreCutA )
    {
        params.outPreCutA->contours = std::move( meshAContours );
        params.outPreCutA->mesh = std::move( meshA );
    }
    if ( needCutMeshA && !params.outPreCutA )
    {
        taskGroup.run( [&] ()
        {
            FaceMap* cut2oldAPtr = params.mapper ? &params.mapper->maps[int( BooleanResultMapper::MapObject::A )].cut2origin : nullptr;
            // cut meshes
            CutMeshParameters cmParams;
            cmParams.sortData = dataForA.get();
            cmParams.new2OldMap = cut2oldAPtr;
            auto res = cutMesh( meshA, meshAContours, cmParams );
            meshAContours.clear();
            meshAContours.shrink_to_fit(); // free memory
            if ( cut2oldAPtr && !new2orgSubdivideMapA.empty() )
            {
                ParallelFor( *cut2oldAPtr, [&] ( FaceId i )
                {
                    if ( !( *cut2oldAPtr )[i] )
                        return;
                    FaceId refFace = ( *cut2oldAPtr )[i];
                    if ( new2orgSubdivideMapA.size() > refFace && new2orgSubdivideMapA[refFace] )
                        ( *cut2oldAPtr )[i] = new2orgSubdivideMapA[refFace];

                } );
            }
            result.meshABadContourFaces = std::move( res.fbsWithCountourIntersections );
            cutA = std::move( res.resultCut );
        } );
    }

    if ( params.outPreCutB )
    {
        params.outPreCutB->contours = std::move( meshBContours );
        params.outPreCutB->mesh = std::move( meshB );
    }
    if ( needCutMeshB && !params.outPreCutB )
    {
        FaceMap* cut2oldBPtr = params.mapper ? &params.mapper->maps[int( BooleanResultMapper::MapObject::B )].cut2origin : nullptr;
        // cut meshes
        CutMeshParameters cmParams;
        cmParams.sortData = dataForB.get();
        cmParams.new2OldMap = cut2oldBPtr;
        auto res = cutMesh( meshB, meshBContours, cmParams );
        meshBContours.clear();
        meshBContours.shrink_to_fit(); // free memory
        if ( cut2oldBPtr && !new2orgSubdivideMapB.empty() )
        {
            ParallelFor( *cut2oldBPtr, [&] ( FaceId i )
            {
                if ( !( *cut2oldBPtr )[i] )
                    return;
                FaceId refFace = ( *cut2oldBPtr )[i];
                if ( new2orgSubdivideMapB.size() > refFace && new2orgSubdivideMapB[refFace] )
                    ( *cut2oldBPtr )[i] = new2orgSubdivideMapB[refFace];
            } );
        }
        result.meshBBadContourFaces = std::move( res.fbsWithCountourIntersections );
        cutB = std::move( res.resultCut );
    }
    taskGroup.wait();


    if ( result.meshABadContourFaces.any() )
    {
        result.errorString = "Bad contour on " + std::to_string( result.meshABadContourFaces.count() ) + " mesh A faces, " +
            "probably mesh B has self-intersections on contours lying on these faces.";
        return result;
    }
    else if ( result.meshBBadContourFaces.any() )
    {
        result.errorString = "Bad contour on " + std::to_string( result.meshBBadContourFaces.count() ) + " mesh B faces, " +
            "probably mesh A has self-intersections on contours lying on these faces.";
        return result;
    }


    if ( mainCb && !mainCb( 0.66f ) )
        return { .errorString = stringOperationCanceled() };

    if ( params.outPreCutA || params.outPreCutB )
        return {};

    intParams.optionalOutCut = params.outCutEdges;
    // do operation
    auto res = doBooleanOperation( std::move( meshA ), std::move( meshB ), cutA, cutB, operation, params.rigidB2A, params.mapper, params.mergeAllNonIntersectingComponents, intParams );

    if ( mainCb && !mainCb( 1.0f ) )
        return { .errorString = stringOperationCanceled() };

    if ( res.has_value() )
        result.mesh = std::move( res.value() );
    else
        result.errorString = res.error();
    return result;
}

Expected<BooleanResultPoints> getBooleanPoints( const Mesh& meshA, const Mesh& meshB, 
    BooleanOperation operation, const AffineXf3f* rigidB2A )
{
    MR_TIMER

    BooleanResultPoints result;
    result.meshAVerts.resize( meshA.topology.lastValidVert() + 1 );
    result.meshBVerts.resize( meshB.topology.lastValidVert() + 1 );

    const auto converters = getVectorConverters( meshA, meshB, rigidB2A );
    const auto intersections = findCollidingEdgeTrisPrecise( meshA, meshB, converters.toInt, rigidB2A );
    result.intersectionPoints.reserve( intersections.edgesAtrisB.size() + intersections.edgesBtrisA.size() );

    FaceBitSet collFacesA, collFacesB;
    VertBitSet destVertsA, destVertsB, orgVertsA, orgVertsB;
    collFacesA.resize( meshA.topology.lastValidFace() + 1 );
    collFacesB.resize( meshB.topology.lastValidFace() + 1 );
    orgVertsA.resize( meshA.topology.lastValidVert() + 1 );
    orgVertsB.resize( meshB.topology.lastValidVert() + 1 );
    destVertsA.resize( meshA.topology.lastValidVert() + 1 );
    destVertsB.resize( meshB.topology.lastValidVert() + 1 );
    for ( const auto& et : intersections.edgesAtrisB )
    {
        gatherEdgeInfo( meshA.topology, et.edge, collFacesA, orgVertsA, destVertsA );
        collFacesB.set( et.tri );

        const auto isect = findEdgeTriIntersectionPoint( meshA, et.edge, meshB, et.tri, converters, rigidB2A, EdgeTriComponent::Tri );
        result.intersectionPoints.emplace_back( isect );
    }
    for ( const auto& et : intersections.edgesBtrisA )
    {
        gatherEdgeInfo( meshB.topology, et.edge, collFacesB, orgVertsB, destVertsB );
        collFacesA.set( et.tri );

        const auto isect = findEdgeTriIntersectionPoint( meshB, et.edge, meshA, et.tri, converters, rigidB2A, EdgeTriComponent::Edge );
        result.intersectionPoints.emplace_back( isect );
    }

    if ( ( orgVertsA & destVertsA ).any() || ( orgVertsB & destVertsB ).any() )
    {
        // in this case we are not able to detect inside outside correctly
        BooleanResultMapper mapper;
        auto boolRes = MR::boolean( meshA, meshB, operation, rigidB2A, &mapper );
        if ( !boolRes.valid() )
            return unexpected( boolRes.errorString );

        if ( !mapper.maps[int( BooleanResultMapper::MapObject::A )].old2newVerts.empty() )
            for ( auto v : meshA.topology.getValidVerts() )
            {
                auto vn = mapper.maps[int( BooleanResultMapper::MapObject::A )].old2newVerts[v];
                if ( vn.valid() )
                    result.meshAVerts.set( v );
            }

        if ( !mapper.maps[int( BooleanResultMapper::MapObject::B )].old2newVerts.empty() )
            for ( auto v : meshB.topology.getValidVerts() )
            {
                auto vn = mapper.maps[int( BooleanResultMapper::MapObject::B )].old2newVerts[v];
                if ( vn.valid() )
                    result.meshBVerts.set( v );
            }

        return result;
    }

    auto collBordersA = findLeftBoundary( meshA.topology, collFacesA );
    auto collBordersB = findLeftBoundary( meshB.topology, collFacesB );

    const bool needInsidePartA = ( operation == BooleanOperation::Intersection || operation == BooleanOperation::InsideA || operation == BooleanOperation::DifferenceBA );
    const bool needInsidePartB = ( operation == BooleanOperation::Intersection || operation == BooleanOperation::InsideB || operation == BooleanOperation::DifferenceAB );

    if ( operation != BooleanOperation::InsideB && operation != BooleanOperation::OutsideB )
    {
        std::erase_if( collBordersA, [&] ( const EdgeLoop& edgeLoop )
        {
            return needInsidePartA != destVertsA.test( meshA.topology.dest( edgeLoop.front() ) );
        } );

        collFacesA = fillContourLeft( meshA.topology, collBordersA );
        result.meshAVerts = getInnerVerts( meshA.topology, collFacesA );

        // reset outer
        result.meshAVerts -= ( needInsidePartA ? destVertsA : orgVertsA );

        const auto aComponents = MeshComponents::getAllComponents(meshA);

        for ( const auto& aComponent : aComponents )
        {
            const auto aComponentVerts = getInnerVerts( meshA.topology, aComponent );
            if ( aComponentVerts.intersects( result.meshAVerts ) )
                continue;
            const bool inside = isInside( MeshPart( meshA, &aComponent ), MeshPart( meshB ), rigidB2A );
            if ( needInsidePartA == inside )
            {
                result.meshAVerts |= aComponentVerts;
            }
        }
    }

    if ( operation != BooleanOperation::InsideA && operation != BooleanOperation::OutsideA )
    {
        std::erase_if( collBordersB, [&] ( const EdgeLoop& edgeLoop )
        {
            return needInsidePartB != destVertsB.test( meshB.topology.dest( edgeLoop.front() ) );
        } );

        collFacesB = fillContourLeft(meshB.topology, collBordersB);
        result.meshBVerts = getInnerVerts( meshB.topology, collFacesB );

        // reset outer
        result.meshBVerts -= ( needInsidePartB ? destVertsB : orgVertsB );

        const auto bComponents = MeshComponents::getAllComponents(meshB);
        std::unique_ptr<AffineXf3f> rigidA2B;
        if ( rigidB2A )
            rigidA2B = std::make_unique<AffineXf3f>( rigidB2A->inverse() );

        for ( const auto& bComponent : bComponents )
        {
            const auto bComponentVerts = getInnerVerts( meshB.topology, bComponent );
            if ( bComponentVerts.intersects( result.meshBVerts ) )
                continue;

            const bool inside = isInside( MeshPart( meshB, &bComponent ), MeshPart( meshA ), rigidA2B.get() );

            if ( needInsidePartB == inside  )
            {
                result.meshBVerts |= bComponentVerts;
            }
        }
    }

    return result;
}

TEST( MRMesh, MeshBoolean )
{
    Mesh meshA = makeTorus( 1.1f, 0.5f, 8, 8 );
    Mesh meshB = makeTorus( 1.0f, 0.2f, 8, 8 );
    meshB.transform( AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), Vector3f::plusY() ) ) );

    const float shiftStep = 0.2f;
    const float angleStep = PI_F;/* *1.0f / 3.0f*/;
    const std::array<Vector3f, 3> baseAxis{Vector3f::plusX(),Vector3f::plusY(),Vector3f::plusZ()};
    for ( int maskTrans = 0; maskTrans < 8; ++maskTrans )
    {
        for ( int maskRot = 0; maskRot < 8; ++maskRot )
        {
            for ( float shift = 0.01f; shift < 0.2f; shift += shiftStep )
            {
                Vector3f shiftVec;
                for ( int i = 0; i < 3; ++i )
                    if ( maskTrans & ( 1 << i ) )
                        shiftVec += shift * baseAxis[i];
                for ( float angle = PI_F * 0.01f; angle < PI_F * 7.0f / 18.0f; angle += angleStep )
                {
                    Matrix3f rotation;
                    for ( int i = 0; i < 3; ++i )
                        if ( maskRot & ( 1 << i ) )
                            rotation = Matrix3f::rotation( baseAxis[i], angle ) * rotation;

                    AffineXf3f xf;
                    xf = AffineXf3f::translation( shiftVec ) * AffineXf3f::linear( rotation );

                    EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation::Union, &xf ).valid() );
                    EXPECT_TRUE( boolean( meshB, meshA, BooleanOperation::Intersection, &xf ).valid() );
                }
            }
        }
    }
}


TEST( MRMesh, BooleanMultipleEdgePropogationSort )
{
    Mesh meshA;
    meshA.points = std::vector<Vector3f>
    {
        {0.0f,0.0f,0.0f},
        {-0.5f,1.0f,0.0f},
        {0.5f,1.0f,0.0f},
        {0.0f,1.5f,0.5f},
        {-1.0f,1.5f,0.0f},
        {1.0f,1.5f,0.0f}
    };
    Triangulation tA =
    {
        { 0_v, 2_v, 1_v },
        { 1_v, 2_v, 3_v },
        { 3_v, 4_v, 1_v },
        { 2_v, 5_v, 3_v },
        { 3_v, 5_v, 4_v }
    };
    meshA.topology = MeshBuilder::fromTriangles( tA );
    {
        Mesh meshASup = meshA;
        meshASup.points[3_v] = { 0.0f,1.5f,-0.5f };


        auto border = trackRightBoundaryLoop( meshA.topology, meshA.topology.findHoleRepresentiveEdges()[0] );

        meshA.addPartByMask( meshASup, meshASup.topology.getValidFaces(), true, { border }, { border } );
    }

    auto meshB = makeCube( Vector3f::diagonal( 2.0f ) );
    meshB.transform( AffineXf3f::translation( Vector3f( -1.5f, -0.2f, -0.5f ) ) );


    for ( int i = 0; i<int( BooleanOperation::Count ); ++i )
    {
        EXPECT_TRUE( boolean( meshA, meshB, BooleanOperation( i ) ).valid() );
        EXPECT_TRUE( boolean( meshB, meshA, BooleanOperation( i ) ).valid() );
    }
}

} //namespace MR
