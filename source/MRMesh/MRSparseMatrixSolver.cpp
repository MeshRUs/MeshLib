#include "MRSparseMatrixSolver.h"

namespace MR
{

void SimplicialLDLTSolver::compute( const SparseMatrixColMajor& A )
{
    solver_.compute( A );
}

Eigen::VectorXd SimplicialLDLTSolver::solve( const Eigen::VectorXd& rhs )
{
    return solver_.solve( rhs );
}

}
