#include "MRMeshMetrics.h"
#include "MRMesh.h"
#include "MRMeshDelone.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"
#include "MRPlane3.h"
#include "MRBestFit.h"
#include "MRConstants.h"
#include "MRTimer.h"

namespace 
{
// This constant modifier was born empirically
constexpr double TriangleAreaModifier = 1e2;
}

namespace MR
{

const double BadTriangulationMetric = 1e10;

double calcCombinedFillMetric( const Mesh & mesh, const FaceBitSet & filledRegion, const FillHoleMetric & metric )
{
    MR_TIMER
    auto cm = metric.combineMetric;
    if ( !cm )
        cm = [] ( double a, double b ) { return a + b; };

    double res = 0;
    for ( auto t : filledRegion )
    {
        VertId vs[3];
        mesh.topology.getTriVerts( t, vs );
        res = cm( res, metric.triangleMetric( vs[0], vs[1], vs[2] ) );
        if ( !metric.edgeMetric )
            continue;
        EdgeId es[3];
        mesh.topology.getTriEdges( t, es );
        for ( int i = 0; i < 3; ++i )
        {
            const auto e = es[i];
            assert( t == mesh.topology.left( e ) );
            const auto rt = mesh.topology.right( e );
            if ( !rt || ( rt > t && filledRegion.test( rt ) ) )
                continue; // do not examine edge twice
            const auto a = mesh.topology.org( e );
            const auto b = mesh.topology.dest( e );
            const auto l = mesh.topology.dest( mesh.topology.next( e ) );
            const auto r = mesh.topology.dest( mesh.topology.prev( e ) );
            res = cm( res, metric.edgeMetric( a, b, l, r ) );
        }
    }
    return res;
}

FillHoleMetric getCircumscribedMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.triangleMetric = [&] ( VertId a, VertId b, VertId c )
    {
        return circumcircleDiameter( mesh.points[a], mesh.points[b], mesh.points[c] );
    };
    return metric;
}

FillHoleMetric getPlaneFillMetric( const Mesh& mesh, EdgeId e0 )
{
    auto norm = Vector3d();
    for ( auto e : leftRing( mesh.topology, e0 ) )
    {
        norm += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    }
    norm = norm.normalized();

    FillHoleMetric metric;
    metric.triangleMetric = [&mesh,norm] ( VertId a, VertId b, VertId c )
    {
        Vector3d aP = Vector3d( mesh.points[a] );
        Vector3d bP = Vector3d( mesh.points[b] );
        Vector3d cP = Vector3d( mesh.points[c] );
        if ( dot( norm, cross( bP - aP, cP - aP ) ) < 0.0 )
            return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to allow some bad meshes

        return circumcircleDiameter( aP, bP, cP );
    };
    return metric;
}

FillHoleMetric getPlaneNormalizedFillMetric( const Mesh& mesh, EdgeId e0 )
{
    auto norm = Vector3d();
    for ( auto e : leftRing( mesh.topology, e0 ) )
    {
        norm += cross( Vector3d( mesh.orgPnt( e ) ), Vector3d( mesh.destPnt( e ) ) );
    }
    norm = norm.normalized();

    FillHoleMetric metric;
    metric.triangleMetric = [&mesh, norm] ( VertId a, VertId b, VertId c )
    {
        Vector3d aP = Vector3d( mesh.points[a] );
        Vector3d bP = Vector3d( mesh.points[b] );
        Vector3d cP = Vector3d( mesh.points[c] );

        auto faceNorm = cross( bP - aP, cP - aP );
        auto faceDblAreaSq = faceNorm.lengthSq();
        if ( faceDblAreaSq == 0.0f ) // degenerated
            return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to be allow some bad meshes

        auto dotRes = dot( norm, faceNorm );
        if ( ( dotRes < 0.0f ) || ( sqr( dotRes ) * 4.0f < faceDblAreaSq ) )
            return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to be allow some bad meshes

        auto ar = triangleAspectRatio( aP, bP, cP );
        if ( ar > BadTriangulationMetric )
            return BadTriangulationMetric; // DBL_MAX break any triangulation, just return big value to be allow some bad meshes

        return circumcircleDiameter( aP, bP, cP ) * ar;
    };
    return metric;
}

FillHoleMetric getComplexStitchMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.triangleMetric = [&] ( VertId a, VertId b, VertId c )
    {
        return  ( triangleAspectRatio( mesh.points[a], mesh.points[b], mesh.points[c] ) - 1.0f ) * 1e-2f; // 1e-2 because aspect ratio grows infinitely
    };
    metric.edgeMetric = [&] ( VertId a, VertId b, VertId l, VertId r )
    {
        auto ab = ( mesh.points[b] - mesh.points[a] );
        auto normL = cross( mesh.points[l] - mesh.points[a], ab ).normalized();
        auto normR = cross( ab, mesh.points[r] - mesh.points[a] ).normalized();
        return ( 1.0 - dot( normL, normR ) ) * 1e4; // 1e4 because aspect ratio grows infinitely, and we need more affect from angles
    };
    return metric;
}

FillHoleMetric getEdgeLengthFillMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.edgeMetric = [&] ( VertId a, VertId b, VertId, VertId )
    {
        return ( mesh.points[b] - mesh.points[a] ).length();
    };
    return metric;
}

FillHoleMetric getEdgeLengthStitchMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    // this can be implemented via edgeMetric as in getEdgeLengthFillMetric,
    // but it is slower in stitchHoles than the implementation via triangleMetric
    metric.triangleMetric = [&] ( VertId a, VertId, VertId c )
    {
        return ( mesh.points[c] - mesh.points[a] ).length();
    };
    return metric;
}

FillHoleMetric getVerticalStitchMetric( const Mesh& mesh, const Vector3f& upDir )
{
    FillHoleMetric metric;
    metric.triangleMetric = [&mesh, up = upDir.normalized()]( VertId a, VertId b, VertId c )
    {
        auto ab = mesh.points[b] - mesh.points[a];
        auto ac = mesh.points[c] - mesh.points[a];

        auto norm = cross( ab, ac ); // dbl area
        auto parallelPenalty = std::abs( dot( up, norm ) );

        // sqr penalty and sides length to have valid m^4 power of each argument
        // norm.lengthSq - dbl area Sq - m^4
        // parallelPenaltySq ~ area cos(angle(updir,norm)) sq - m^4 
        // side length sq sq - m^4
        return
            norm.lengthSq() +
            100.0f * sqr( parallelPenalty ) + // this should be big
            sqr( ab.lengthSq() + ac.lengthSq() + ( mesh.points[c] - mesh.points[b] ).lengthSq() ) * 0.5f;
    };
    return metric;
}

FillHoleMetric getComplexFillMetric( const Mesh& mesh, EdgeId e0 )
{
    float maxEdgeLengthSq = 0.0f;
    for ( auto e : leftRing( mesh.topology, e0 ) )
        maxEdgeLengthSq = std::max( maxEdgeLengthSq, mesh.edgeLengthSq( e ) );

    assert( maxEdgeLengthSq > 0.0f );

    float reverseCharacteristicTriArea{ 0.0f };
    if ( maxEdgeLengthSq <= 0.0f )
        reverseCharacteristicTriArea = 1.0f;
    else
        reverseCharacteristicTriArea = 1.0f / maxEdgeLengthSq;

    FillHoleMetric metric;
    metric.triangleMetric = [&mesh, reverseCharacteristicTriArea] ( VertId a, VertId b, VertId c )
    {
        double aspectRatio = triangleAspectRatio( mesh.points[a], mesh.points[b], mesh.points[c] );
        if ( aspectRatio > BadTriangulationMetric )
            return BadTriangulationMetric;

        double normedArea = TriangleAreaModifier * cross( mesh.points[b] - mesh.points[a], mesh.points[c] - mesh.points[a] ).length() * reverseCharacteristicTriArea;

        return aspectRatio + normedArea;
    };
    metric.edgeMetric = [&mesh] ( VertId a, VertId b, VertId l, VertId r )
    {
        auto abVec = mesh.points[b] - mesh.points[a];
        auto bcVec = -abVec;
        auto normA = cross( mesh.points[r] - mesh.points[b], bcVec );
        auto normC = cross( mesh.points[l] - mesh.points[a], abVec );

        auto s_Abc_double = normA.length();
        auto s_abC_double = normC.length();
        auto denom = s_Abc_double * s_abC_double;
        if ( denom == 0.0f )
            return BadTriangulationMetric;
        auto cosAC = dot( normA, normC ) / ( s_Abc_double * s_abC_double );

        if ( cosAC <= -1.0f )
            return BadTriangulationMetric;

        return double( sqr( sqr( ( 1.0f - cosAC ) / ( 1.0f + cosAC ) ) ) );
    };
    return metric;
}

FillHoleMetric getParallelPlaneFillMetric( const Mesh& mesh, EdgeId e0, const Plane3f* plane /*= nullptr */ )
{
    auto normal = Vector3f();
    if ( plane )
        normal = plane->n.normalized();
    else
    {
        PointAccumulator accum;
        for ( auto e : leftRing( mesh.topology, e0 ) )
            accum.addPoint( mesh.orgPnt( e ) );

        normal = accum.getBestPlanef().n.normalized();
    }

    FillHoleMetric metric;
    metric.edgeMetric = [&mesh, normal] ( VertId a, VertId b, VertId, VertId )
    {
        return std::abs( dot( normal, mesh.points[b] - mesh.points[a] ) );
    };
    return metric;
}

FillHoleMetric getMaxDihedralAngleMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.edgeMetric = [&] ( VertId a, VertId b, VertId l, VertId r ) -> double
    {
        const auto& aP = mesh.points[a];
        const auto& bP = mesh.points[b];
        const auto& lP = mesh.points[l];
        const auto& rP = mesh.points[r];
        auto ab = bP - aP;
        auto normL = cross( lP - aP, ab ); //it is ok not to normalize for dihedralAngle call
        auto normR = cross( ab, rP - aP );
        return std::abs( dihedralAngle( normL, normR, ab ) );
    };
    metric.combineMetric = [] ( double a, double b )
    {
        return a > b ? a : b;
    };
    return metric;
}

FillHoleMetric getUniversalMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.triangleMetric = [&] ( VertId a, VertId b, VertId c )
    {
        return circumcircleDiameter( mesh.points[a], mesh.points[b], mesh.points[c] );
    };
    metric.edgeMetric = [&] ( VertId a, VertId b, VertId l, VertId r ) -> double
    {
        const auto& aP = mesh.points[a];
        const auto& bP = mesh.points[b];
        const auto& lP = mesh.points[l];
        const auto& rP = mesh.points[r];
        auto ab = bP - aP;
        auto normL = cross( lP - aP, ab ); //it is ok not to normalize for dihedralAngle call
        auto normR = cross( ab, rP - aP );
        // exp(10* angle) - was too big and broke even double precision
        return ab.length() * std::exp( 5 * ( std::abs( dihedralAngle( normL, normR, ab ) ) ) );
    };
    return metric;
}

FillHoleMetric getMinTriAngleMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.triangleMetric = [&] ( VertId a, VertId b, VertId c )
    {
        constexpr double maxSin = 0.86602540378443864676372317075294; //std::sqrt( 3. ) / 2;
        return std::exp( 25 * ( maxSin - minTriangleAngleSin( mesh.points[a], mesh.points[b], mesh.points[c] ) ) );
    };
    return metric;
}

FillHoleMetric getMinAreaMetric( const Mesh& mesh )
{
    FillHoleMetric metric;
    metric.triangleMetric = [&mesh] ( VertId a, VertId b, VertId c )
    {
        Vector3d aP = Vector3d( mesh.points[a] );
        Vector3d bP = Vector3d( mesh.points[b] );
        Vector3d cP = Vector3d( mesh.points[c] );

        return dblArea( aP, bP, cP );
    };
    return metric;
}

}
