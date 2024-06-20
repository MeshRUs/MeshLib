#include "MRLaplacian.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRExpandShrink.h"
#include "MRRingIterator.h"
#include "MRMakeSphereMesh.h"
#include "MRMeshComponents.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

namespace MR
{

void Laplacian::init( const VertBitSet & freeVerts, EdgeWeights weights, RememberShape rem )
{
    MR_TIMER;
    assert( !MeshComponents::hasFullySelectedComponent( mesh_, freeVerts ) );

    if ( !solver_ )
        solver_ = std::make_shared<SimplicialLDLTSolver>();

    solverValid_ = false;

    freeVerts_ = freeVerts;
    region_ = freeVerts;
    // free vertices and the first layer around the region
    expand( mesh_.topology, region_ );

    // build matrix of equations: vertex pos = mean pos of its neighbors
    regionVert2id_.resize( region_.size() );
    equations_.clear();
    nonZeroElements_.clear();
    std::vector<Element> rowElements;
    for ( auto v : region_ )
    {
        rowElements.clear();
        regionVert2id_[v] = (int)equations_.size();
        Equation eq;
        eq.firstElem = (int)nonZeroElements_.size();
        double sumW = 0;
        Vector3d sumWPos;
        for ( auto e : orgRing( mesh_.topology, v ) )
        {
            double w = 1;
            if ( weights == EdgeWeights::Cotan || weights == EdgeWeights::CotanWithAreaEqWeight ) 
                w = std::clamp( mesh_.cotan( e ), -1.0f, 10.0f ); // cotan() can be arbitrary high for degenerate edges
            else if ( weights == EdgeWeights::CotanTimesLength ) 
                w = mesh_.edgeLength( e ) * mesh_.cotan( e );
            auto d = mesh_.topology.dest( e );
            rowElements.push_back( { -w, d } );
            sumWPos -= w * Vector3d( mesh_.points[d] );
            sumW += w;
        }
        double a = 1;
        if ( weights == EdgeWeights::CotanWithAreaEqWeight )
            if ( auto d = mesh_.dblArea( v ); d > 0 )
                a =  1 / std::sqrt( d );
        const double rSumW = a / sumW;
        for ( auto el : rowElements )
        {
            el.coeff *= rSumW;
            nonZeroElements_.push_back( el );
        }
        eq.rhs = ( rem == RememberShape::Yes ) ? sumWPos * rSumW + a * Vector3d( mesh_.points[v] ) : Vector3d();
        eq.centerCoeff = a;
        equations_.push_back( eq );
    }
    Equation eq;
    eq.firstElem = (int)nonZeroElements_.size();
    equations_.push_back( eq );
}

void Laplacian::fixVertex( VertId v, bool smooth ) 
{
    rhsValid_ = false;
    if ( freeVerts_.autoResizeTestSet( v, false ) )
        solverValid_ = false;
    if ( fixedSharpVertices_.autoResizeTestSet( v, !smooth ) != !smooth )
        solverValid_ = false;
}

void Laplacian::fixVertex( VertId v, const Vector3f & fixedPos, bool smooth ) 
{ 
    mesh_.points[v] = fixedPos; 
    fixVertex( v, smooth ); 
}

void Laplacian::updateSolver()
{
    updateSolver_();
    updateRhs_();
}

void Laplacian::updateSolver_()
{
    if ( solverValid_ )
        return;
    solverValid_ = true;

    MR_TIMER

    const auto sz = freeVerts_.count();
    if ( sz <= 0 )
    {
        rhsValid_ = true;
        return;
    }
    rhsValid_ = false;

    freeVert2id_ = makeVectorWithSeqNums( freeVerts_ );

    firstLayerFixedVerts_ = freeVerts_;
    expand( mesh_.topology, firstLayerFixedVerts_ );
    firstLayerFixedVerts_ -= fixedSharpVertices_;
    const auto rowSz = firstLayerFixedVerts_.count();
    firstLayerFixedVerts_ -= freeVerts_;

    std::vector< Eigen::Triplet<double> > mTriplets;
    // equations for free vertices
    int n = 0;
    for ( auto v : freeVerts_ )
    {
        assert( n == freeVert2id_[v] );
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        mTriplets.emplace_back( n, n, eq.centerCoeff );
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( freeVerts_.test( el.neiVert ) )
                mTriplets.emplace_back( n, freeVert2id_[el.neiVert], el.coeff );
        }
        ++n;
    }

    // equations for free neighbors of fixed vertices
    for ( auto v : firstLayerFixedVerts_ )
    {
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( freeVerts_.test( el.neiVert ) )
                mTriplets.emplace_back( n, freeVert2id_[el.neiVert], el.coeff );
        }
        ++n;
    }
    assert( n == rowSz );

    M_.resize( rowSz, sz );
    M_.setFromTriplets( mTriplets.begin(), mTriplets.end() );

    SparseMatrix A = M_.adjoint() * M_;

    solver_->compute( A );
}

template <typename I, typename G, typename S>
void Laplacian::prepareRhs_( I && iniRhs, G && g, S && s )
{
    // equations for free vertices
    int n = 0;
    for ( auto v : freeVerts_ )
    {
        assert( n == freeVert2id_[v] );
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        auto r = iniRhs( eq );
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( !freeVerts_.test( el.neiVert ) )
                r -= el.coeff * g( el.neiVert );
        }
        s( n, r );
        ++n;
    }

    // equations for free neighbors of fixed vertices
    for ( auto v : firstLayerFixedVerts_ )
    {
        int eqN = regionVert2id_[v];
        const auto eq = equations_[eqN];
        auto r = iniRhs( eq );
        r -= eq.centerCoeff * g( v );
        const auto lastElem = equations_[eqN+1].firstElem;
        for ( int ei = eq.firstElem; ei < lastElem; ++ ei )
        {
            const auto el = nonZeroElements_[ei];
            if ( !freeVerts_.test( el.neiVert ) )
                r -= el.coeff * g( el.neiVert );
        }
        s( n, r );
        ++n;
    }
    assert( n == M_.rows() );
}

void Laplacian::updateRhs_()
{
    assert( solverValid_ );
    if ( rhsValid_ )
        return;
    rhsValid_ = true;

    MR_TIMER

    Eigen::VectorXd rhs[3];
    for ( int i = 0; i < 3; ++i )
        rhs[i].resize( M_.rows() );

    prepareRhs_(
        [&]( const Equation & eq ) { return eq.rhs; },
        [&]( VertId v ) { return Vector3d{ mesh_.points[v] }; },
        [&]( int n, const Vector3d & r )
        {
            for ( int i = 0; i < 3; ++i )
                rhs[i][n] = r[i];
        }
    );

    tbb::parallel_for( tbb::blocked_range<int>( 0, 3, 1 ), [&]( const tbb::blocked_range<int> & range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
            rhs_[i] = M_.adjoint() * rhs[i];
    } );
}

void Laplacian::apply()
{
    MR_TIMER;
    if ( !freeVerts_.any() )
        return;
    updateSolver();

    Eigen::VectorXd sol[3];
    tbb::parallel_for( tbb::blocked_range<int>( 0, 3, 1 ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int i = range.begin(); i < range.end(); ++i )
            sol[i] = solver_->solve( rhs_[i] );
    } );

    // copy solution back into mesh points
    for ( auto v : freeVerts_ )
    {
        int mapv = freeVert2id_[v];
        auto & pt = mesh_.points[v];
        pt.x = (float) sol[0][mapv];
        pt.y = (float) sol[1][mapv];
        pt.z = (float) sol[2][mapv];
    }
    mesh_.invalidateCaches();
}

void Laplacian::applyToScalar( VertScalars & scalarField )
{
    MR_TIMER;
    if ( !freeVerts_.any() )
        return;
    updateSolver();

    Eigen::VectorXd rhs( M_.rows() );

    prepareRhs_(
        [&]( const Equation & ) { return 0.0; },
        [&]( VertId v ) { return scalarField[v]; },
        [&]( int n, double r ) { rhs[n] = r; }
    );

    Eigen::VectorXd sol = solver_->solve( M_.adjoint() * rhs );
    for ( auto v : freeVerts_ )
    {
        int mapv = freeVert2id_[v];
        scalarField[v] = float( sol[mapv] );
    }
}

TEST(MRMesh, Laplacian) 
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );

    {
        VertBitSet vs;
        vs.autoResizeSet( 0_v );
        Laplacian laplacian( sphere );
        laplacian.init( vs, Laplacian::EdgeWeights::Cotan );
        laplacian.apply();

        // fix the only free vertex
        laplacian.fixVertex( 0_v );
        laplacian.apply();
    }

    {
        Laplacian laplacian( sphere );
        // no free verts
        laplacian.init( {}, Laplacian::EdgeWeights::Cotan );
        laplacian.apply();
    }
}

} //namespace MR
