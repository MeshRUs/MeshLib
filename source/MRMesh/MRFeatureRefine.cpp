#include "MRFeatureRefine.h"

#include "MRBitSetParallelFor.h"
#include "MRFeatureObjectImpls.h"
#include "MRFeatureHelpers.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRPointCloud.h"
#include "MRPointCloudRadius.h"
#include "MRPointsComponents.h"
#include "MRRegionBoundary.h"
#include "MRRingIterator.h"
#include "MRTimer.h"

#include <MRPch/MRFmt.h>

namespace
{

using namespace MR;

int getPointsMinimalCount( FeaturesObjectKind objectKind )
{
    switch ( objectKind )
    {
        case FeaturesObjectKind::Point:
            return 1;
        case FeaturesObjectKind::Line:
            return 2;
        case FeaturesObjectKind::Plane:
        case FeaturesObjectKind::Circle:
            return 3;
        case FeaturesObjectKind::Sphere:
            return 4;
        case FeaturesObjectKind::Cylinder:
            return 6;
        case FeaturesObjectKind::Cone:
            return 7;
        case FeaturesObjectKind::_count:
            MR_UNREACHABLE_NO_RETURN
    }
    MR_UNREACHABLE
}

VertBitSet getVertsForRefineFeature( const Mesh& mesh, const FeatureObject& feature, float distanceEps, float normalEps )
{
    VertBitSet detectedVerts;
    detectedVerts.resize( mesh.topology.lastValidVert() + 1, false );

    const auto distanceEpsSq = distanceEps * distanceEps;
    const auto cosNormalEps = std::cos( normalEps / 180.0f * PI_F );

    BitSetParallelForAll( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        const auto proj = feature.projectPoint( mesh.points[v] );
        const auto lenSq = ( proj.point - mesh.points[v] ).lengthSq();
        if ( lenSq >= distanceEpsSq )
            return;

        if ( !proj.normal )
        {
            detectedVerts.set( v, true );  // normals are not support for current feature type;
            return;
        }

        const auto& worldNormal = *proj.normal;

        // check vert normal.
        if ( std::abs( dot( worldNormal, mesh.normal( v ) ) ) >= cosNormalEps )
        {
            detectedVerts.set( v, true );
            return;
        }

        // Also check the face normals surrounding the current vertex.
        // If at least one face has a correct normal, add the vertex to the list.
        // It is required for working with meshes created by CAD (like STEP files or create primitives),
        // where almost all verts located at the sharp edged and have normals which is not correlated with surrounded faces.
        for ( const auto e : orgRing( mesh.topology, v ) )
        {
            const auto f = mesh.topology.left( e );
            if ( !f.valid() )
                continue;

            if ( std::abs( dot( worldNormal, mesh.normal( f ) ) ) >= cosNormalEps )
            {
                detectedVerts.set( v, true );
                return;
            }
        }
    } );

    return detectedVerts;
}

VertBitSet getPointsForRefineFeature( const PointCloud& pointCloud, const FeatureObject& feature, float distanceEps, float normalEps )
{
    VertBitSet detectedVerts;
    detectedVerts.resize( pointCloud.validPoints.size() + 1, false );

    const auto distanceEpsSq = distanceEps * distanceEps;
    const auto cosNormalEps = std::cos( normalEps / 180.0f * PI_F );

    BitSetParallelFor( pointCloud.validPoints, [&] ( VertId v )
    {
        const auto proj = feature.projectPoint( pointCloud.points[v] );
        const auto lenSq = ( proj.point - pointCloud.points[v] ).lengthSq();
        if ( lenSq >= distanceEpsSq )
            return;

        if ( pointCloud.hasNormals() && proj.normal )
        {
            const auto& worldNormal = *proj.normal;
            if ( std::abs( dot( worldNormal, pointCloud.normals[v] ) ) < cosNormalEps )
                return;
        }

        detectedVerts.set( v, true );
    } );

    return detectedVerts;
}

VertBitSet filterDisjointPoints( const Mesh& mesh, const VertBitSet& selectedPoints, const FaceBitSet& referenceFaces )
{
    MR_TIMER

    const auto selectedFaces = getIncidentFaces( mesh.topology, selectedPoints );
    const auto components = MeshComponents::getAllComponents( { mesh, &selectedFaces }, MeshComponents::FaceIncidence::PerVertex );

    FaceBitSet result( mesh.topology.lastValidFace() + 1, false );
    for ( const auto& component : components )
        if ( ( component & referenceFaces ).any() )
            result |= component;

    return getIncidentVerts( mesh.topology, result ) & selectedPoints;
}

VertBitSet filterDisjointPoints( const PointCloud& pointCloud, const VertBitSet& selectedPoints, const VertBitSet& referencePoints, float segmentTolerance )
{
    MR_TIMER

    const auto avgRadius = findAvgPointsRadius( pointCloud, 20 );
    const auto res = PointCloudComponents::getAllComponents( pointCloud, avgRadius );
    if ( !res )
        return selectedPoints;

    const auto& [components, groupSize] = *res;
    assert( groupSize == 1 );
    if ( components.size() <= 1 )
        return selectedPoints;

    Box3f refPointBox;
    for ( const auto& v : referencePoints )
        refPointBox.include( pointCloud.points[v] );
    refPointBox.include( refPointBox.min - Vector3f::diagonal( segmentTolerance ) );
    refPointBox.include( refPointBox.max + Vector3f::diagonal( segmentTolerance ) );

    VertBitSet result( pointCloud.points.size(), false );
    for ( const auto& component : components )
    {
        for ( const auto v : component )
        {
            const auto& p = pointCloud.points[v];
            if ( refPointBox.contains( p ) )
            {
                result |= component;
                break;
            }
        }
    }

    return result & selectedPoints;
}

AffineXf3f getRefinedXf( const Mesh& mesh, FeaturesObjectKind featureType, const VertBitSet& toRefine, const FaceBitSet* referenceFaces )
{
    auto selection = referenceFaces ? filterDisjointPoints( mesh, toRefine, *referenceFaces ) : toRefine;
    // ignore the result if it is too small
    if ( selection.count() < getPointsMinimalCount( featureType ) )
        selection = toRefine;

    std::vector<Vector3f> points;
    points.reserve( selection.count() );

    for ( const auto v : selection )
        points.emplace_back( mesh.points[v] );

    auto tempObject = makeObjectFromEnum( featureType, points );
    return tempObject->xf();
}

AffineXf3f getRefinedXf( const PointCloud& pointCloud, FeaturesObjectKind featureType, const VertBitSet& toRefine, const VertBitSet* referencePoints )
{
    const auto segmentTolerance = pointCloud.computeBoundingBox().diagonal() / 128.f;
    auto selection = referencePoints ? filterDisjointPoints( pointCloud, toRefine, *referencePoints, segmentTolerance ) : toRefine;
    // ignore the result if it is too small
    if ( selection.count() < getPointsMinimalCount( featureType ) )
        selection = toRefine;

    std::vector<Vector3f> points;
    points.reserve( selection.count() );

    for ( const auto v : selection )
        points.emplace_back( pointCloud.points[v] );

    auto tempObject = makeObjectFromEnum( featureType, points );
    return tempObject->xf();
}

void makeFeaturePseudoInfinite( std::shared_ptr<FeatureObject>& feature, const Box3f& boundingBox )
{
    constexpr float cMultiplierPseudoInfinite = 5.0f;
    const auto newDimension = boundingBox.diagonal() * cMultiplierPseudoInfinite;

    if ( auto plane = std::dynamic_pointer_cast<PlaneObject>( feature ) )
    {
        plane->setSize( newDimension );
    }
    else if ( auto cylinder = std::dynamic_pointer_cast<CylinderObject>( feature ) )
    {
        cylinder->setLength( newDimension );
    }
    else if ( auto cone = std::dynamic_pointer_cast<ConeObject>( feature ) )
    {
        cone->setHeight( newDimension );
    }
}

} // namespace

namespace MR
{

Expected<AffineXf3f> refineFeatureObject( const FeatureObject& featObj, const Mesh& mesh, const RefineParameters& params )
{
    MR_TIMER

    auto featureType = FeaturesObjectKind::_count;
    forEachObjectKind( [&] ( auto type )
    {
        if ( dynamic_cast<typename ObjKindTraits<type.value>::type const*>( &featObj ) )
        {
            featureType = type;
            return true;
        }
        return false;
    } );
    assert( featureType != FeaturesObjectKind::_count );

    auto refinedFeature = std::dynamic_pointer_cast<FeatureObject>( featObj.clone() );
    assert( refinedFeature );
    makeFeaturePseudoInfinite( refinedFeature, mesh.computeBoundingBox() );

    VertBitSet prevResults;
    for ( auto i = 0; i < params.maxIterations; ++i )
    {
        auto vbs = getVertsForRefineFeature( mesh, *refinedFeature, params.distanceLimit, params.normalTolerance );
        if ( vbs.count() < getPointsMinimalCount( featureType ) )
        {
            return unexpected( fmt::format(
                "Unable to refine. Number of selected verts ({}) less than minimal ({}) for this feature type",
                vbs.count(),
                getPointsMinimalCount( featureType )
            ) );
        }

        const auto refinedXf = getRefinedXf( mesh, featureType, vbs, params.faceRegion );
        refinedFeature->setXf( refinedXf );

        if ( vbs == prevResults )
            break;
        prevResults = std::move( vbs );

        if ( !reportProgress( params.callback, (float)i / (float)params.maxIterations ) )
            return unexpectedOperationCanceled();
    }

    return refinedFeature->xf();
}

Expected<AffineXf3f> refineFeatureObject( const FeatureObject& featObj, const PointCloud& pointCloud, const RefineParameters& params )
{
    MR_TIMER

    auto featureType = FeaturesObjectKind::_count;
    forEachObjectKind( [&] ( auto type )
    {
        if ( dynamic_cast<typename ObjKindTraits<type.value>::type const*>( &featObj ) )
        {
            featureType = type;
            return true;
        }
        return false;
    } );
    assert( featureType != FeaturesObjectKind::_count );

    auto refinedFeature = std::dynamic_pointer_cast<FeatureObject>( featObj.clone() );
    assert( refinedFeature );
    makeFeaturePseudoInfinite( refinedFeature, pointCloud.computeBoundingBox() );

    VertBitSet prevResults;
    for ( auto i = 0; i < params.maxIterations; ++i )
    {
        auto vs = getPointsForRefineFeature( pointCloud, *refinedFeature, params.distanceLimit, params.normalTolerance );
        if ( vs.count() < getPointsMinimalCount( featureType ) )
        {
            return unexpected( fmt::format(
                "Unable to refine. Number of selected verts ({}) less than minimal ({}) for this feature type",
                vs.count(),
                getPointsMinimalCount( featureType )
            ) );
        }

        const auto refinedXf = getRefinedXf( pointCloud, featureType, vs, params.vertRegion );
        refinedFeature->setXf( refinedXf );

        if ( vs == prevResults )
            break;
        prevResults = std::move( vs );

        if ( !reportProgress( params.callback, (float)i / (float)params.maxIterations ) )
            return unexpectedOperationCanceled();
    }

    return refinedFeature->xf();
}

} // namespace MR
