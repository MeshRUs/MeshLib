#pragma once
#include "MRMeshFwd.h"

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
#include <Eigen/SparseCholesky>
#pragma clang diagnostic pop
#pragma warning(pop)

namespace MR
{

class SparseMatrixSolver
{
public:
    virtual ~SparseMatrixSolver() = default;

    using SparseMatrixColMajor = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    virtual void compute( const SparseMatrixColMajor& A ) = 0;
    virtual Eigen::VectorXd solve( const Eigen::VectorXd& rhs ) = 0;
};

class SimplicialLDLTSolver final : public SparseMatrixSolver
{
public:
    MRMESH_API virtual void compute( const SparseMatrixColMajor& A ) override final;

    MRMESH_API virtual Eigen::VectorXd solve( const Eigen::VectorXd& rhs ) override final;
private:
    Eigen::SimplicialLDLT<SparseMatrixColMajor> solver_;
};

// 1 rework Laplacian to use SparseMatrixSolver instead of Laplacian::Solver
// 2 change parent class of CuSparseLDLTSolver (rename as CudaCholeskySolver)
// 3 make wrapper TestSolverClass under Solver class


}
