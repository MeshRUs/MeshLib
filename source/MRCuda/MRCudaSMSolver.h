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

class CuSparseLDLTSolver : public SparseMatrixSolver
{
public:
    using SparseMatrixColMajor = Eigen::SparseMatrix<double, Eigen::ColMajor>;

    void init();
    MRCUDA_API void compute( const SparseMatrixColMajor& A );
    MRCUDA_API Eigen::VectorXd solve( const Eigen::VectorXd& rhs );
    MRCUDA_API bool isError() { return error_; }
private:
    void eigenSparseToCuSparse_( const SparseMatrixColMajor& mat, int*& row, int*& col, double*& val, int& num_non0, int& num_outer );
    void eigenToCuSparse_( const Eigen::VectorXd& vec, double*& val, int& size );
    void cuSparseTransposeToEigenSparse_( const int* row, const int* col, const double* val, const int num_non0,
    const int mat_row, const int mat_col, Eigen::SparseMatrix<double>& mat );
    void cuSparseToEigen_( const double* val, const int size, Eigen::VectorXd& vec );

    void compute2_( const SparseMatrixColMajor& A );
    Eigen::VectorXd solve2_( const Eigen::VectorXd& rhs );

    // version 1
    cusparseHandle_t handle_ = NULL;
    cusparseSpMatDescr_t matA_;
    cusparseDnVecDescr_t vecX_;
    cusparseDnVecDescr_t vecY_;

    int A_num_rows = 0;
    int A_num_cols = 0;

    // version 2
    cusolverSpHandle_t cusolverHandle_;
    cusparseMatDescr_t descrA_;
    int* dRow_ = nullptr;
    int* dCol_ = nullptr;
    double* dVal_ = nullptr;
    int num_non0_ = 0;

    bool error_ = false;
};

} // namespace MR::Cuda
