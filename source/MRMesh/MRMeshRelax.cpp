#include "MRMeshRelax.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MREdgePaths.h"
#include "MRBestFitQuadric.h"
#include "MRVector4.h"
#include "MRMeshFixer.h"
#include "MRRegionBoundary.h"
#include "MRMeshComponents.h"
#include "MRLaplacian.h"
#include "MRLineSegm.h"
#include "MRGeodesicPath.h"
#include "MRSymMatrix2.h"

namespace MR
{

bool relax( Mesh& mesh, const MeshRelaxParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = mesh.points;
    MR_WRITER( mesh );

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newPoints = mesh.points;
        if ( !BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            Vector3d sum;
            int count = 0;
            for ( auto e : orgRing( mesh.topology, e0 ) )
            {
                sum += Vector3d( mesh.points[mesh.topology.dest( e )] );
                ++count;
            }
            auto np = newPoints[v];
            auto pushForce = params.force * ( Vector3f{sum / double( count )} - np );
            np += pushForce;
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb ) )
            return false;
        mesh.points.swap( newPoints );
    }
    if ( params.hardSmoothTetrahedrons )
        hardSmoothTetrahedrons( mesh, params.region );
    return true;
}

Vector3f vertexPosEqualNeiAreas( const Mesh& mesh, VertId v, bool noShrinkage )
{
    // computation in doubles improves quality of the result in case of degenerate input
    SymMatrix3d mat;
    Vector3d rhs;
    const EdgeId e0 = mesh.topology.edgeWithOrg( v );
    EdgeId ei = e0;
    EdgeId en = mesh.topology.next( ei );
    auto pi = Vector3d( mesh.destPnt( ei ) );
    auto pn = Vector3d( mesh.destPnt( en ) );
    for (;;)
    {
        if ( mesh.topology.left( ei ) )
        {
            const auto m = crossSquare( pn - pi );
            mat += m;
            rhs += m * pi;
        }
        if ( en == e0 )
            break;
        ei = en;
        pi = pn;
        en = mesh.topology.next( ei );
        pn = Vector3d( mesh.destPnt( en ) );
    } 

    if ( noShrinkage )
    {
        const auto norm = Vector3d( mesh.normal( v ) );
        const auto [x,y] = norm.perpendicular();
        SymMatrix2d mat2;
        const auto mx = mat * x;
        mat2.xx = dot( mx, x );
        mat2.xy = dot( mx, y );
        mat2.yy = dot( mat * y, y );

        const auto det = mat2.det();
        const auto tr = mat2.trace();
        if ( DBL_EPSILON * std::abs( tr * tr ) >= std::abs( det ) )
            return mesh.points[v]; // the linear system cannot be trusted

        const auto p0 = dot( norm, Vector3d( mesh.points[v] ) ) * norm;
        rhs -= mat * p0;
        Vector2d rhs2;
        rhs2.x = dot( rhs, x );
        rhs2.y = dot( rhs, y );

        const auto sol2 = mat2.inverse( det ) * rhs2;
        return Vector3f( p0 + x * sol2.x + y * sol2.y );
    }

    const auto det = mat.det();
    const auto tr = mat.trace();
    if ( DBL_EPSILON * std::abs( tr * tr * tr ) >= std::abs( det ) )
        return mesh.points[v]; // the linear system cannot be trusted

    return Vector3f( mat.inverse( det ) * rhs );
}

bool equalizeTriAreas( Mesh& mesh, const MeshEqualizeTriAreasParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = mesh.points;
    MR_WRITER( mesh );

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newPoints = mesh.points;
        if ( !BitSetParallelFor( zone, [&]( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            auto np = newPoints[v];
            auto pushForce = params.force * ( vertexPosEqualNeiAreas( mesh, v, params.noShrinkage ) - np );
            np += pushForce;
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb ) )
            return false;
        mesh.points.swap( newPoints );
    }
    if ( params.hardSmoothTetrahedrons )
        hardSmoothTetrahedrons( mesh, params.region );
    return true;
}

bool relaxKeepVolume( Mesh& mesh, const MeshRelaxParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = mesh.points;
    MR_WRITER( mesh );

    VertCoords newPoints;

    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    std::vector<Vector3f> vertPushForces( zone.size() );
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb1 = subprogress( cb, [&]( float p ) { return ( float( i ) + p * 0.5f ) / float( params.iterations ); } );
        auto internalCb2 = subprogress( cb, [&]( float p ) { return ( float( i ) + p * 0.5f + 0.5f ) / float( params.iterations ); } );
        newPoints = mesh.points;
        if ( !BitSetParallelFor( zone, [&]( VertId v )
        {
            Vector3d sum;
            int count = 0;
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                sum += Vector3d( mesh.points[mesh.topology.dest( e )] );
                ++count;
            }
            vertPushForces[v] = params.force * ( Vector3f{sum / double( count )} - mesh.points[v] );
        }, internalCb1 ) )
            return false;

        if ( !BitSetParallelFor( zone, [&]( VertId v )
        {
            Vector3d sum;
            int count = 0;
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                auto d = mesh.topology.dest( e );
                if ( zone.test( d ) )
                    sum += Vector3d( vertPushForces[d] );
                ++count;
            }
            auto np = newPoints[v] + vertPushForces[v] - Vector3f{ sum / double( count ) };
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb2 ) )
            return false;

        mesh.points.swap( newPoints );
    }
    if ( params.hardSmoothTetrahedrons )
        hardSmoothTetrahedrons( mesh, params.region );
    return true;
}

bool relaxApprox( Mesh& mesh, const MeshApproxRelaxParams& params, ProgressCallback cb )
{
    if ( params.iterations <= 0 )
        return true;

    MR_TIMER
    VertCoords initialPos;
    const auto maxInitialDistSq = sqr( params.maxInitialDist );
    if ( params.limitNearInitial )
        initialPos = mesh.points;
    MR_WRITER( mesh );

    float surfaceRadius = ( params.surfaceDilateRadius <= 0.0f ) ?
        ( float( std::sqrt( mesh.area() ) ) * 1e-3f ) : params.surfaceDilateRadius;

    VertCoords newPoints;
    const VertBitSet& zone = mesh.topology.getVertIds( params.region );
    for ( int i = 0; i < params.iterations; ++i )
    {
        auto internalCb = subprogress( cb, [&]( float p ) { return ( float( i ) + p ) / float( params.iterations ); } );
        newPoints = mesh.points;
        if ( !BitSetParallelFor( zone, [&] ( VertId v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0.valid() )
                return;
            VertBitSet neighbors( mesh.topology.lastValidVert() + 1 );
            neighbors.set( v );

            dilateRegion( mesh, neighbors, surfaceRadius );

            PointAccumulator accum;
            int count = 0;
            for ( auto newV : neighbors )
            {
                Vector3d ptD = Vector3d( mesh.points[newV] );
                accum.addPoint( ptD );
                ++count;
            }
            if ( count < 6 )
                return;

            auto np = newPoints[v];

            Vector3f target;
            if ( params.type == RelaxApproxType::Planar )
                target = accum.getBestPlanef().project( np );
            else if ( params.type == RelaxApproxType::Quadric )
            {
                AffineXf3d basis = accum.getBasicXf();
                basis.A = basis.A.transposed();
                std::swap( basis.A.x, basis.A.y );
                std::swap( basis.A.y, basis.A.z );
                basis.A = basis.A.transposed();
                auto basisInv = basis.inverse();

                QuadricApprox approxAccum;
                for ( auto newV : neighbors )
                    approxAccum.addPoint( basisInv( Vector3d( mesh.points[newV] ) ) );

                auto centerPoint = basisInv( Vector3d( mesh.points[v] ) );
                const auto coefs = approxAccum.calcBestCoefficients();
                centerPoint.z =
                    coefs[0] * centerPoint.x * centerPoint.x +
                    coefs[1] * centerPoint.x * centerPoint.y +
                    coefs[2] * centerPoint.y * centerPoint.y +
                    coefs[3] * centerPoint.x +
                    coefs[4] * centerPoint.y +
                    coefs[5];
                target = Vector3f( basis( centerPoint ) );
            }
            np += ( params.force * ( target - np ) );
            if ( params.limitNearInitial )
                np = getLimitedPos( np, initialPos[v], maxInitialDistSq );
            newPoints[v] = np;
        }, internalCb ) )
            return false;
        mesh.points.swap( newPoints );
    }
    if ( params.hardSmoothTetrahedrons )
        hardSmoothTetrahedrons( mesh, params.region );
    return true;
}

void removeSpikes( Mesh & mesh, int maxIterations, float minSumAngle, const VertBitSet * region )
{
    if ( maxIterations <= 0 )
        return;

    MR_TIMER;

    for ( int i = 0; i < maxIterations; ++i )
    {
        auto spikeVerts = mesh.findSpikeVertices( minSumAngle, region ).value();
        if ( spikeVerts.count() == 0 )
            break;
        relax( mesh, { { 1,&spikeVerts } } );
    }
}

void smoothRegionBoundary( Mesh & mesh, const FaceBitSet & regionFaces, int numIters )
{
    MR_TIMER
    assert( numIters > 0 );
    if ( !regionFaces.any() || numIters <= 0 )
        return;

    // value 1 for out-of-region vertices
    VertScalars scalarField( mesh.topology.vertSize(), 1 );

    const auto regionVerts = getIncidentVerts( mesh.topology, regionFaces );
    // value -1 for in-region vertices
    for( auto v : regionVerts )
        scalarField[v] = -1;

    /// free vertices must have both in-region and out-of-region neighbor faces
    VertBitSet freeVerts = getIncidentVerts( mesh.topology, mesh.topology.getValidFaces() - regionFaces ) & regionVerts;

    for ( const auto & cc : MeshComponents::getAllComponentsVerts( mesh ) )
    {
        auto freeCC = cc & freeVerts;
        auto numfree = freeCC.count();
        auto numCC = cc.count();
        assert( numfree <= numCC );
        if ( numfree <= 0 )
            continue; // too small connected component
        if ( numfree < numCC )
            continue; // at least one fixed vertex in the component

        // all component vertices are free, just fix them all (to -1) to avoid under-determined system of equations
        freeVerts -= cc;
    }

    // change topology: eliminate not boundary edges joining two boundary vertices
    for ( auto v : freeVerts )
    {
        mesh.topology.flipEdgesOut( v, [&]( EdgeId e )
        {
            assert( mesh.topology.org( e ) == v );
            if ( regionFaces.test( mesh.topology.left( e ) ) != regionFaces.test( mesh.topology.right( e ) ) )
                return false;

            auto c = mesh.topology.dest( e );

            if ( !freeVerts.test( c ) )
                return false;

            auto b = mesh.topology.dest( mesh.topology.prev( e ) );
            auto d = mesh.topology.dest( mesh.topology.next( e ) );
            if ( freeVerts.test( b ) && freeVerts.test( c ) )
              return false;

            if ( mesh.topology.findEdge( d, b ) )
                return false; // multiple edges between b and d will appear

            auto ap = mesh.points[v];
            auto bp = mesh.points[b];
            auto cp = mesh.points[c];
            auto dp = mesh.points[d];
            return isUnfoldQuadrangleConvex( ap, bp, cp, dp );
        } );
    }

    Laplacian lap( mesh );
    std::vector<Vector3f> newPos;
    for( int iter = 0; iter < numIters; ++iter )
    {
        lap.init( freeVerts, EdgeWeights::Cotan, Laplacian::RememberShape::No );
        lap.applyToScalar( scalarField );

        newPos.clear();
        newPos.reserve( freeVerts.count() );
        for ( const auto v : freeVerts )
        {
            const auto pt = mesh.points[v];
            Vector3f bestPos = pt;
            float bestDist2 = FLT_MAX;
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                if ( !mesh.topology.left( e ) )
                    continue;

                VertId vs[3] = {
                    v,
                    mesh.topology.dest( e ),
                    mesh.topology.dest( mesh.topology.next( e ) )
                };

                for ( int i = 0; i < 3; ++i )
                {
                    const auto val  = scalarField[vs[0]];
                    const auto val1 = scalarField[vs[1]];
                    const auto val2 = scalarField[vs[2]];

                    if ( val * val1 >= 0 )
                        continue;
                    if ( val * val1 >= 0 )
                        continue;

                    LineSegm3f ls;
                    const float c1 = val / ( val - val1 );
                    ls.a = ( 1 - c1 ) * mesh.points[vs[0]] + c1 * mesh.points[vs[1]];
                    const float c2 = val / ( val - val2 );
                    ls.b = ( 1 - c2 ) * mesh.points[vs[0]] + c2 * mesh.points[vs[2]];

                    const auto proj = closestPointOnLineSegm( pt, ls );
                    const auto dist2 = ( pt - proj ).lengthSq();
                    if ( dist2 < bestDist2 )
                    {
                        bestPos = proj;
                        bestDist2 = dist2;
                    }

                    std::rotate( vs, vs + 1, vs + 3 );
                }
            }

            newPos.push_back( bestPos );
        }

        int n = 0;
        for ( const auto v : freeVerts )
            // 0.75 to reduce oscillation on the next iteration
            mesh.points[v] += 0.75f * ( newPos[n++] - mesh.points[v] );
    }
}

void hardSmoothTetrahedrons( Mesh & mesh, const VertBitSet *region )
{
    MR_TIMER
    auto tetrahedrons = findNRingVerts( mesh.topology, 3, region );
    // in normal mesh two vertices from tetrahedrons cannot be neighbors, so it is safe to run it in parallel
    BitSetParallelFor( tetrahedrons, [&] ( VertId v )
    {
        Vector3f center;
        for ( auto e : orgRing( mesh.topology, v ) )
            center += mesh.destPnt( e );
        mesh.points[v] = center / 3.0f;
    } );
}

} //namespace MR
