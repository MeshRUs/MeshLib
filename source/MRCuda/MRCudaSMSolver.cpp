#include "MRCudaSMSolver.h"
#include <cusparse.h>         // cusparseSpSV
#include <cusolverSp.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <Eigen/SparseCholesky>
#include <iostream>
#include "MRMesh/MRTimer.h"
#include "MRPch/MRTBB.h"
#pragma warning(disable: 4505)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        error_ = true;                                                         \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUDA_RES(func)                                                   \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        error_ = true;                                                         \
        return {};                                                             \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        error_ = true;                                                         \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUSPARSE_RES(func)                                               \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        error_ = true;                                                         \
        return {};                                                             \
    }                                                                          \
}

#define CHECK_CUSOLVER(func)                                                   \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("CUSOLVER API failed at line %d with error: (%d)\n",            \
               __LINE__, status);                                              \
        error_ = true;                                                         \
        return;                                                                \
    }                                                                          \
}

#define CHECK_CUSOLVER_RES(func)                                               \
{                                                                              \
    cusolverStatus_t status = (func);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        printf("CUSOLVER API failed at line %d with error: (%d)\n",            \
               __LINE__, status);                                              \
        error_ = true;                                                         \
        return {};                                                             \
    }                                                                          \
}


namespace MR::Cuda
{

void CuSparseLDLTSolver::init()
{
    error_ = false;
}

void CuSparseLDLTSolver::compute( const SparseMatrixColMajor& A )
{
    compute2_( A );
}

Eigen::VectorXd CuSparseLDLTSolver::solve( const Eigen::VectorXd& rhs )
{
    return solve2_( rhs );
}

void CuSparseLDLTSolver::eigenSparseToCuSparse_( const SparseMatrixColMajor& mat, int*& row, int*& col, double*& val, int& num_non0, int& num_outer )
{
    num_non0 = ( int ) mat.nonZeros();
    num_outer = ( int ) mat.cols() + 1;
    row = ( int* )malloc( sizeof( int ) * num_outer );
    col = ( int* )malloc( sizeof( int ) * num_non0 );
    val = ( double* )malloc( sizeof( double ) * num_non0 );

    memcpy( row, mat.outerIndexPtr(), sizeof( int ) * num_outer );
    memcpy( col, mat.innerIndexPtr(), sizeof( int ) * num_non0 );
    memcpy( val, mat.valuePtr(), sizeof( double ) * num_non0 );
}

void CuSparseLDLTSolver::eigenToCuSparse_( const Eigen::VectorXd& vec, double*& val, int& size )
{
    size = ( int ) vec.size();
    val = ( double* )malloc( sizeof( double ) * size );

    memcpy( val, vec.data(), sizeof( double ) * size );
}

void CuSparseLDLTSolver::cuSparseTransposeToEigenSparse_( const int* row, const int* col, const double* val,
    const int num_non0, const int mat_row, const int mat_col, Eigen::SparseMatrix<double>& mat )
{
    std::vector<int> outer( mat_col + 1 );
    std::vector<int> inner( num_non0 );
    std::vector<double> value( num_non0 );

    cudaMemcpy( outer.data(), row, sizeof( int ) * ( mat_col + 1 ), cudaMemcpyDeviceToHost );
    cudaMemcpy( inner.data(), col, sizeof( int ) * num_non0, cudaMemcpyDeviceToHost );
    cudaMemcpy( value.data(), val, sizeof( double ) * num_non0, cudaMemcpyDeviceToHost );

    Eigen::Map<Eigen::SparseMatrix<double>> mat_map( mat_row, mat_col, num_non0, outer.data(), inner.data(), value.data() );
    mat = mat_map.eval();
}

void CuSparseLDLTSolver::cuSparseToEigen_( const double* val, const int size, Eigen::VectorXd& vec )
{
    vec.resize( size );
    memcpy( vec.data(), ( void* )val, sizeof( double ) * size );
}

void CuSparseLDLTSolver::compute2_( const SparseMatrixColMajor& A )
{
    MR_TIMER;

    const SparseMatrixColMajor At = A.transpose();

    int* hRow = nullptr;
    int* hCol = nullptr;
    double* hVal = nullptr;
    int num_outer;
    eigenSparseToCuSparse_( At, hRow, hCol, hVal, num_non0_, num_outer );

    A_num_rows = ( int )At.rows();
    A_num_cols = ( int )At.cols();

    CHECK_CUDA( cudaMalloc( ( void** )&dRow_, num_outer * sizeof( int ) ) );
    CHECK_CUDA( cudaMalloc( ( void** )&dCol_, num_non0_ * sizeof( int ) ) );
    CHECK_CUDA( cudaMalloc( ( void** )&dVal_, num_non0_ * sizeof( double ) ) );
    CHECK_CUDA( cudaMemcpy( dRow_, hRow, num_outer * sizeof( int ), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( dCol_, hCol, num_non0_ * sizeof( int ), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( dVal_, hVal, num_non0_ * sizeof( double ), cudaMemcpyHostToDevice ) );


    CHECK_CUSOLVER( cusolverSpCreate( &cusolverHandle_ ) );
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA_ ) );
}

Eigen::VectorXd CuSparseLDLTSolver::solve2_( const Eigen::VectorXd& rhs )
{
    MR_TIMER;

    assert( A_num_rows == A_num_cols );

    Eigen::VectorXd res;
    res.resize( rhs.size() );
    // init X and B
    double* hX = nullptr;
    double* hB = nullptr;
    int hXsize = 0;
    int hBsize = 0;
    eigenToCuSparse_( res, hX, hXsize );
    eigenToCuSparse_( rhs, hB, hBsize );
    assert( hBsize == A_num_rows );

    double* dX = nullptr;
    double* dB = nullptr;
    CHECK_CUDA_RES( cudaMalloc( ( void** )&dX, hXsize * sizeof( double ) ) );
    CHECK_CUDA_RES( cudaMalloc( ( void** )&dB, hBsize * sizeof( double ) ) );
    CHECK_CUDA_RES( cudaMemcpy( dX, hX, hXsize * sizeof( double ), cudaMemcpyHostToDevice ) );
    CHECK_CUDA_RES( cudaMemcpy( dB, hB, hBsize * sizeof( double ), cudaMemcpyHostToDevice ) );

    int singularity;
    CHECK_CUSOLVER_RES( cusolverSpDcsrlsvchol( cusolverHandle_, A_num_rows, num_non0_, descrA_,
        dVal_, dRow_, dCol_, dB,
        0., 1, dX, &singularity ) );

    CHECK_CUDA_RES( cudaMemcpy( hX, dX, hXsize * sizeof( double ), cudaMemcpyDeviceToHost ) );
    cuSparseToEigen_( hX, hXsize, res );
    return res;
}

}
