#pragma once
#include "exports.h"
#include "MRMesh/MRSparseMatrixSolver.h"

struct cusparseContext;
typedef struct cusparseContext* cusparseHandle_t;
struct cusparseDnVecDescr;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
struct cusparseSpMatDescr;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
struct cusolverSpContext;
typedef struct cusolverSpContext* cusolverSpHandle_t;
struct cusparseMatDescr;
typedef struct cusparseMatDescr* cusparseMatDescr_t;

namespace MR::Cuda
{

class CudaSparseMatrixSolver : public SparseMatrixSolver
{
public:
    using SparseMatrixColMajor = Eigen::SparseMatrix<double, Eigen::ColMajor>;

    void init();
    MRCUDA_API void compute( const SparseMatrixColMajor& A );
    MRCUDA_API Eigen::VectorXd solve( const Eigen::VectorXd& rhs );
    MRCUDA_API bool isError() { return error_; }
private:
    void sparseMatrixEigenToCuda_( const SparseMatrixColMajor& mat, int*& dRow, int*& dCol, double*& dVal, int& numNonZero );
    void denseVectorEigenToCuda_( const Eigen::VectorXd& vec, double*& dVal );
    void cudaTransposeToEigenSparse_( const int* row, const int* col, const double* val, const int num_non0,
        const int mat_row, const int mat_col, Eigen::SparseMatrix<double>& mat );
    void denseVectorCudaToEigen_( const double* dVal, const int size, Eigen::VectorXd& vec );

    void compute_( const SparseMatrixColMajor& A );
    Eigen::VectorXd solve_( const Eigen::VectorXd& rhs );

    int matANumRows_ = 0;
    int matANumCols_ = 0;

    // version 2
    cusolverSpHandle_t cusolverHandle_;
    cusparseMatDescr_t descrA_;
    int* dRow_ = nullptr;
    int* dCol_ = nullptr;
    double* dVal_ = nullptr;
    int numNonZero_ = 0;

    bool error_ = false;
};

} // namespace MR::Cuda
