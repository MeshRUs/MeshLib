#include "MRLevelOfDetails.h"
#include "MRObjectMesh.h"
#include "MRMesh.h"
#include "MRMeshDecimate.h"
#include "MRBuffer.h"
#include "MRTimer.h"
#include "MRParallelFor.h"

namespace MR
{

std::shared_ptr<Object> makeLevelOfDetails( Mesh && mesh, int maxDepth )
{
    MR_TIMER
    assert( maxDepth > 0 );

    mesh.packOptimally( false );

    const int totalFaces = mesh.topology.numValidFaces();
    int numFacesPerObj = totalFaces;
    for ( int d = 0; d < maxDepth; ++d )
        numFacesPerObj /= 2;

    struct FaceSpan
    {
        FaceId beg, end;
    };

    std::vector<FaceSpan> spans;
    spans.push_back( { 0_f, FaceId( totalFaces ) } );
    for ( int d = 1; d < maxDepth; ++d )
    {
        std::vector<FaceSpan> nextSpans;
        for ( const auto & span : spans )
        {
            auto mid = span.beg + ( span.end - span.beg ) / 2;
            nextSpans.push_back( { span.beg, mid } );
            nextSpans.push_back( { mid, span.end } );
        }
        spans = std::move( nextSpans );
    }

    std::vector<FaceBitSet> meshParts( spans.size() );
    std::vector<std::shared_ptr<Object>> levelObjs( spans.size() );
    ParallelFor( levelObjs, [&]( size_t i )
    {
        auto objMesh = std::make_shared<ObjectMesh>();

        FaceBitSet & region = meshParts[i];
        region.resize( spans[i].end, false );
        region.set( spans[i].beg, spans[i].end - spans[i].beg, true );
        objMesh->setMesh( std::make_shared<Mesh>( mesh.cloneRegion( region ) ) );

        objMesh->setName( "mesh" + std::to_string( maxDepth ) + "_" + std::to_string( i ) );
        objMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Edges, ViewportMask::all() );
        objMesh->setVisible( false );
        levelObjs[i] = std::move( objMesh );
    } );

    int currDepth = maxDepth;
    while ( meshParts.size() > 1 )
    {
        --currDepth;
        DecimateSettings dsettings;
        dsettings.subdivideParts = int( meshParts.size() );
        dsettings.decimateBetweenParts = false;
        dsettings.partFaces = &meshParts;
        dsettings.maxError = FLT_MAX;
        dsettings.minFacesInPart = numFacesPerObj;
        decimateMesh( mesh, dsettings );

        ParallelFor( levelObjs, [&]( size_t i )
        {
            auto objMesh = std::make_shared<ObjectMesh>();

            const FaceBitSet & region = meshParts[i];
            objMesh->setMesh( std::make_shared<Mesh>( mesh.cloneRegion( region ) ) );
            objMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Edges, ViewportMask::all() );

            auto obj = std::make_shared<Object>();
            obj->setName( "lvl" + std::to_string( currDepth ) + "_" + std::to_string( i ) );

            objMesh->setName( "mesh" + std::to_string( currDepth ) + "_" + std::to_string( i ) );
            obj->addChild( objMesh );
            obj->addChild( levelObjs[i] );
            levelObjs[i]->setVisible( false );
            levelObjs[i] = std::move( obj );
        } );

        std::vector<FaceBitSet> nextMeshParts( meshParts.size() / 2 );
        std::vector<std::shared_ptr<Object>> nextLevelObjs( meshParts.size() / 2 );
        ParallelFor( nextLevelObjs, [&]( size_t i )
        {
            auto l = 2 * i;
            auto r = 2 * i + 1;
            nextMeshParts[i] = meshParts[l] | meshParts[r];
            auto obj = std::make_shared<Object>();
            obj->setName( "lvl" + std::to_string( currDepth ) );
            obj->addChild( levelObjs[l] );
            obj->addChild( levelObjs[r] );
            nextLevelObjs[i] = std::move( obj );
        } );
        meshParts = std::move( nextMeshParts );
        levelObjs = std::move( nextLevelObjs );
    }

    // final decimation of least precise mesh of whole object
    DecimateSettings dsettings;
    dsettings.maxError = FLT_MAX;
    dsettings.maxDeletedFaces = mesh.topology.numValidFaces() - numFacesPerObj;
    dsettings.packMesh = true;
    decimateMesh( mesh, dsettings );

    auto res = std::make_shared<Object>();
    res->setName( "Levels of Details" );
    auto objMesh = std::make_shared<ObjectMesh>();
    objMesh->setName( "mesh0" );
    objMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Edges, ViewportMask::all() );
    objMesh->setMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    res->addChild( objMesh );
    for ( auto & obj : levelObjs )
    {
        obj->setVisible( false );
        res->addChild( obj );
    }

    return res;
}

} //namespace MR
