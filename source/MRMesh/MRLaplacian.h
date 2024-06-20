#pragma once

#include "MRBitSet.h"
#include "MRVector.h"
#include "MRVector3.h"
#include "MRSparseMatrixSolver.h"

#pragma warning(push)
#pragma warning(disable: 4068) // unknown pragmas
#pragma warning(disable: 4127) // conditional expression is constant
#pragma warning(disable: 4464) // relative include path contains '..'
#pragma warning(disable: 5054) // operator '|': deprecated between enumerations of different types
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma clang diagnostic ignored "-Wunknown-warning-option" // for next one
#pragma clang diagnostic ignored "-Wunused-but-set-variable" // for newer clang
#include <Eigen/SparseCore>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

// Laplacian to smoothly deform a region preserving mesh fine details.
// How to use:
// 1. Initialize Laplacian for the region being deformed, here region properties are remembered.
// 2. Change positions of some vertices within the region and call fixVertex for them.
// 3. Optionally call updateSolver()
// 4. Call apply() to change the remaining vertices within the region
// Then steps 1-4 or 2-4 can be repeated.
class Laplacian
{
public:
    enum class EdgeWeights
    {
        Unit = 0,  // all edges have same weight=1
        Cotan,     // edge weight depends on local geometry and uses cotangent values
        CotanTimesLength, // [deprecated] edge weight is equal to edge length times cotangent weight
        CotanWithAreaEqWeight // cotangent edge weights and equation weights inversely proportional to square root of local area
    };

    enum class RememberShape
    {
        Yes,  // true Laplacian mode when initial mesh shape is remembered and copied in apply
        No    // ignore initial mesh shape in the region and just position vertices smoothly in the region
    };

    Laplacian( Mesh & mesh ) : mesh_( mesh ) { }
    // initialize Laplacian for the region being deformed, here region properties are remembered and precomputed;
    // \param freeVerts must not include all vertices of a mesh connected component
    MRMESH_API void init( const VertBitSet & freeVerts, EdgeWeights weights, RememberShape rem = RememberShape::Yes );
    // notify Laplacian that given vertex has changed after init and must be fixed during apply;
    // \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
    MRMESH_API void fixVertex( VertId v, bool smooth = true );
    // sets position of given vertex after init and it must be fixed during apply (THIS METHOD CHANGES THE MESH);
    // \param smooth whether to make the surface smooth in this vertex (sharp otherwise)
    MRMESH_API void fixVertex( VertId v, const Vector3f & fixedPos, bool smooth = true );
    // if you manually call this method after initialization and fixing vertices then next apply call will be much faster
    MRMESH_API void updateSolver();
    // given fixed vertices, computes positions of remaining region vertices
    MRMESH_API void apply();
    // given a pre-resized scalar field with set values in fixed vertices, computes the values in free vertices
    MRMESH_API void applyToScalar( VertScalars & scalarField );

    // return all initially free vertices and the first layer around the them
    const VertBitSet & region() const { return region_; }
    // return currently free vertices
    const VertBitSet & freeVerts() const { return freeVerts_; }
    // return fixed vertices from the first layer around free vertices
    VertBitSet firstLayerFixedVerts() const { assert( solverValid_ ); return firstLayerFixedVerts_; }

    // set sparse matrix solver
    void setSMSolver( std::shared_ptr<SparseMatrixSolver> solver ) { solver_ = solver; }
private:
    // updates solver_ only
    void updateSolver_();
    // updates rhs_ only
    void updateRhs_();
    template <typename I, typename G, typename S>
    void prepareRhs_( I && iniRhs, G && g, S && s );

    Mesh & mesh_;

    // all initially free vertices and the first layer around the them
    VertBitSet region_;

    // currently free vertices
    VertBitSet freeVerts_;

    // fixed vertices where no smoothness is required
    VertBitSet fixedSharpVertices_;

    // fixed vertices from the first layer around free vertices
    VertBitSet firstLayerFixedVerts_;

    // for all vertices in the region
    struct Equation
    {
        Vector3d rhs;           // equation right hand side
        double centerCoeff = 0; // coefficient on matrix diagonal
        int firstElem = 0;      // index in nonZeroElements_
    };
    std::vector<Equation> equations_;

    struct Element
    {
        double coeff = 0;
        VertId neiVert;
    };
    std::vector<Element> nonZeroElements_;

    // map from vertex index to matrix row/col
    Vector< int, VertId > regionVert2id_;
    Vector< int, VertId > freeVert2id_;

    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using SparseMatrixColMajor = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    SparseMatrix M_;

    // if true then we do not need to recompute solver_ in the apply
    bool solverValid_ = false;

    std::shared_ptr<SparseMatrixSolver> solver_;

    // if true then we do not need to recompute rhs_ in the apply
    bool rhsValid_ = false;
    Eigen::VectorXd rhs_[3];
};

} //namespace MR
