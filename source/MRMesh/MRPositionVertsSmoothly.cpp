#include "MRPositionVertsSmoothly.h"
#include "MRRingIterator.h"
#include "MRMesh.h"
#include "MRMeshComponents.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRTriMath.h"
#include "MRMeshRelax.h"
#include "MRLaplacian.h"
#include "MRTimer.h"
#include <Eigen/SparseCholesky>

namespace MR
{

void positionVertsSmoothly( Mesh& mesh, const VertBitSet& verts,
    EdgeWeights edgeWeightsType,
    const VertBitSet * fixedSharpVertices )
{
    MR_TIMER

    Laplacian laplacian( mesh );
    laplacian.init( verts, edgeWeightsType, Laplacian::RememberShape::No );
    if ( fixedSharpVertices )
        for ( auto v : *fixedSharpVertices )
            laplacian.fixVertex( v, false );
    laplacian.apply();
}

void positionVertsSmoothlySharpBd( Mesh& mesh, const VertBitSet& verts,
    const Vector<Vector3f, VertId>* vertShifts, const VertScalars* vertStabilizers )
{
    MR_TIMER
    assert( vertStabilizers || !MeshComponents::hasFullySelectedComponent( mesh, verts ) );

    const auto sz = verts.count();
    if ( sz <= 0 )
        return;

    // vertex id -> position in the matrix
    HashMap<VertId, int> vertToMatPos = makeHashMapWithSeqNums( verts );

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );
    int n = 0;
    for ( auto v : verts )
    {
        double sumW = 0;
        Vector3d sumFixed;
        for ( auto e : orgRing( mesh.topology, v ) )
        {
            sumW += 1;
            auto d = mesh.topology.dest( e );
            if ( auto it = vertToMatPos.find( d ); it != vertToMatPos.end() )
            {
                // free neighbor
                int di = it->second;
                if ( n > di ) // row > col: fill only lower left part of matrix
                    mTriplets.emplace_back( n, di, -1 );
            }
            else
            {
                // fixed neighbor
                sumFixed += Vector3d( mesh.points[d] );
            }
        }
        if ( vertShifts )
            sumFixed += sumW * Vector3d( (*vertShifts)[v] );
        if ( vertStabilizers )
        {
            const auto s = (*vertStabilizers)[v];
            sumW += s;
            sumFixed += Vector3d( s * mesh.points[v] );
        }
        mTriplets.emplace_back( n, n, sumW );
        for ( int i = 0; i < 3; ++i )
            rhs[i][n] = sumFixed[i];
        ++n;
    }

    using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
    SparseMatrix A;
    A.resize( sz, sz );
    A.setFromTriplets( mTriplets.begin(), mTriplets.end() );
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute( A );

    Eigen::VectorXd sol[3];
    ParallelFor( 0, 3, [&]( int i )
    {
        sol[i] = solver.solve( rhs[i] );
    } );

    // copy solution back into mesh points
    n = 0;
    for ( auto v : verts )
    {
        auto & pt = mesh.points[v];
        pt.x = (float) sol[0][n];
        pt.y = (float) sol[1][n];
        pt.z = (float) sol[2][n];
        ++n;
    }
}

void positionVertsWithSpacing( Mesh& mesh, const SpacingSettings & settings )
{
    MR_TIMER
    assert( settings.maxSumNegW > 0 );

    const auto & verts = mesh.topology.getVertIds( settings.region );
    const auto sz = verts.count();
    if ( sz <= 0 || settings.numIters <= 0 )
        return;

    FaceBitSet myFaces;
    const FaceBitSet * incidentFaces = nullptr;
    if ( settings.isInverted && settings.region )
    {
        myFaces = getIncidentFaces( mesh.topology, *settings.region );
        incidentFaces = &myFaces;
    }

    // vertex id -> position in the matrix
    HashMap<VertId, int> vertToMatPos = makeHashMapWithSeqNums( verts );

    std::vector< Eigen::Triplet<double> > mTriplets;
    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( sz );

    VertBitSet shiftedVerts;
    for ( int iter = 0; iter < settings.numIters; ++iter )
    {
        mTriplets.clear();
        int n = 0;
        for ( auto v : verts )
        {
            double sumW = 0;
            float sumNegW = 0;
            Vector3d sumFixed;
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                const auto d = mesh.topology.dest( e );
                const auto l = ( mesh.points[v] - mesh.points[d] ).length();
                const auto t = settings.dist( e );
                float w = 0;
                if ( t > l )
                    w = l > 0 ? 1 - t / l : -1;
                else if ( l > t )
                    w = t > 0 ? l / t - 1 : 1;
                sumW += w;
                if ( w < 0 )
                     sumNegW -= w;
                if ( auto it = vertToMatPos.find( d ); it != vertToMatPos.end() )
                {
                    // free neighbor
                    int di = it->second;
                    if ( n > di ) // row > col: fill only lower left part of matrix
                        mTriplets.emplace_back( n, di, -w );
                }
                else
                {
                    // fixed neighbor
                    sumFixed += Vector3d( w * mesh.points[d] );
                }
            }
            auto s = settings.stabilizer;
            if ( sumNegW > settings.maxSumNegW )
                s += sumNegW / settings.maxSumNegW;
            sumFixed += Vector3d( s * mesh.points[v] );
            mTriplets.emplace_back( n, n, sumW + s );
            for ( int i = 0; i < 3; ++i )
                rhs[i][n] = sumFixed[i];
            ++n;
        }

        using SparseMatrix = Eigen::SparseMatrix<double,Eigen::RowMajor>;
        SparseMatrix A;
        A.resize( sz, sz );
        A.setFromTriplets( mTriplets.begin(), mTriplets.end() );
        Eigen::SimplicialLDLT<SparseMatrix> solver;
        solver.compute( A );

        Eigen::VectorXd sol[3];
        ParallelFor( 0, 3, [&]( int i )
        {
            sol[i] = solver.solve( rhs[i] );
        } );

        // copy solution back into mesh points
        n = 0;
        for ( auto v : verts )
        {
            auto & pt = mesh.points[v];
            pt.x = (float) sol[0][n];
            pt.y = (float) sol[1][n];
            pt.z = (float) sol[2][n];
            ++n;
        }

        if ( settings.isInverted )
        {
            shiftedVerts.clear();
            shiftedVerts.resize( mesh.topology.vertSize(), false );
            bool anyInverted = false;
            for ( auto f : mesh.topology.getFaceIds( incidentFaces ) )
            {
                if ( !settings.isInverted( f ) )
                    continue;
                anyInverted = true;
                auto vs = mesh.topology.getTriVerts( f );
                Triangle3f t0;
                for ( int i = 0; i < 3; ++i )
                    t0[i] = mesh.points[ vs[i] ];
                auto t = makeDegenerate( t0 );

                if ( settings.region )
                {
                    // some triangle's vertices can be fixed
                    int numFree = 0;
                    for ( int i = 0; i < 3; ++i )
                        numFree += settings.region->test( vs[i] );
                    assert( numFree >= 1 && numFree <= 3 );
                    if ( numFree == 1 )
                    {
                        // 2 out of 3 vertices are fixed
                        int freeI = -1;
                        for ( int i = 0; i < 3; ++i )
                            if ( settings.region->test( vs[i] ) )
                            {
                                freeI = i;
                                break;
                            }
                        int fixedI0 = ( freeI + 1 ) % 3;
                        int fixedI1 = ( fixedI0 + 1 ) % 3;
                        t = t0;
                        const auto d = ( t[fixedI1] - t[fixedI0] ).normalized();
                        const auto c = 0.5f * ( t[fixedI1] + t[fixedI0] );
                        t[freeI] = c + d * dot( d, t0[freeI] - c );
                    }
                    else if ( numFree == 2 )
                    {
                        // only one vertex is fixed
                        int fixedI = -1;
                        for ( int i = 0; i < 3; ++i )
                            if ( !settings.region->test( vs[i] ) )
                            {
                                fixedI = i;
                                break;
                            }
                        const auto d = t0[fixedI] - t[fixedI];
                        for ( int i = 0; i < 3; ++i )
                            t[i] += d;
                        t[fixedI] = t0[fixedI]; // keep coordinates exactly
                    }
                }

                for ( int i = 0; i < 3; ++i )
                {
                    if ( mesh.points[ vs[i] ] != t[i] )
                    {
                        shiftedVerts.set( vs[i] );
                        mesh.points[ vs[i] ] = t[i];
                    }
                }
            }
            if ( anyInverted )
            {
                // move each point from degenerated triangle a little toward the center of its neighbor,
                // otherwise they will not be pushed away and the degeneracy remains forever
                MeshRelaxParams relaxParams;
                relaxParams.region = &shiftedVerts;
                relaxParams.force = 0.1f;
                relax( mesh, relaxParams );
            }
        }
    }
}

void inflate( Mesh& mesh, const VertBitSet& verts, const InflateSettings & settings )
{
    MR_TIMER
    if ( !verts.any() )
        return;
    if ( settings.preSmooth )
        positionVertsSmoothlySharpBd( mesh, verts );
    if ( settings.iterations <= 0 || settings.pressure == 0 )
        return;

    VertScalars a( verts.find_last() + 1 );
    BitSetParallelFor( verts, [&]( VertId v )
    {
        a[v] = mesh.dblArea( v );
    } );
    double sumArea = 0;
    for ( auto v : verts )
        sumArea += a[v];
    if ( sumArea <= 0 )
        return;
    float rAvgArea = float( 1 / sumArea );
    BitSetParallelFor( verts, [&]( VertId v )
    {
        a[v] *= rAvgArea;
    } );
    // a[v] contains relative area around vertex #v in the whole region, sum(a[v]) = 1

    Vector<Vector3f, VertId> vertShifts( a.size() );
    for ( int i = 0; i < settings.iterations; ++i )
    {
        const auto currPressure = settings.gradualPressureGrowth ?
            ( i + 1 ) * settings.pressure / settings.iterations : settings.pressure;
        BitSetParallelFor( verts, [&]( VertId v )
        {
            vertShifts[v] = currPressure * a[v] * mesh.normal( v );
        } );
        positionVertsSmoothlySharpBd( mesh, verts, &vertShifts );
    }
}

} //namespace MR
