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
        assert( true );                                                        \
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
        assert( true );                                                        \
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
        assert( true );                                                        \
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
        assert( true );                                                        \
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
        assert( true );                                                        \
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
        assert( true );                                                        \
        return {};                                                             \
    }                                                                          \
}


namespace MR::Cuda
{

void CudaSparseMatrixSolver::init()
{
    error_ = false;
}

void CudaSparseMatrixSolver::compute( const SparseMatrixColMajor& A )
{
    compute_( A );
}

Eigen::VectorXd CudaSparseMatrixSolver::solve( const Eigen::VectorXd& rhs )
{
    return solve_( rhs );
}

void CudaSparseMatrixSolver::sparseMatrixEigenToCuda_( const SparseMatrixColMajor& mat, int*& dRow, int*& dCol, double*& dVal, int& numNonZero )
{
    numNonZero = int( mat.nonZeros() );
    int numOuter = int( mat.cols() + 1 );

    CHECK_CUDA( cudaMalloc( ( void** )&dRow, numOuter * sizeof( int ) ) );
    CHECK_CUDA( cudaMalloc( ( void** )&dCol, numNonZero * sizeof( int ) ) );
    CHECK_CUDA( cudaMalloc( ( void** )&dVal, numNonZero * sizeof( double ) ) );
    CHECK_CUDA( cudaMemcpy( dRow, mat.outerIndexPtr(), numOuter * sizeof( int ), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( dCol, mat.innerIndexPtr(), numNonZero * sizeof( int ), cudaMemcpyHostToDevice ) );
    CHECK_CUDA( cudaMemcpy( dVal, mat.valuePtr(), numNonZero * sizeof( double ), cudaMemcpyHostToDevice ) );
}

void CudaSparseMatrixSolver::denseVectorEigenToCuda_( const Eigen::VectorXd& vec, double*& dVal )
{
    int size = int( vec.size() );
    CHECK_CUDA( cudaMalloc( ( void** )&dVal, size * sizeof( double ) ) );
    CHECK_CUDA( cudaMemcpy( dVal, vec.data(), size * sizeof(double), cudaMemcpyHostToDevice));
}

void CudaSparseMatrixSolver::cudaTransposeToEigenSparse_( const int* row, const int* col, const double* val,
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

void CudaSparseMatrixSolver::denseVectorCudaToEigen_( const double* dVal, const int size, Eigen::VectorXd& vec )
{
    vec.resize( size );
    CHECK_CUDA( cudaMemcpy( vec.data(), dVal, size * sizeof( double ), cudaMemcpyDeviceToHost ) );
}

void CudaSparseMatrixSolver::compute_( const SparseMatrixColMajor& A )
{
    MR_TIMER;

    const SparseMatrixColMajor At = A.transpose();

    matANumRows_ = ( int )At.rows();
    matANumCols_ = ( int )At.cols();
    assert( matANumRows_ == matANumCols_ );

    sparseMatrixEigenToCuda_( At, dRow_, dCol_, dVal_, numNonZero_ );

    CHECK_CUSOLVER( cusolverSpCreate( &cusolverHandle_ ) );
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA_ ) );
}

Eigen::VectorXd CudaSparseMatrixSolver::solve_( const Eigen::VectorXd& rhs )
{
    if ( error_ )
        return {};

    MR_TIMER;

    int vecSize = int( rhs.size() );
    assert( vecSize == matANumRows_ );

    double* dB = nullptr;
    denseVectorEigenToCuda_( rhs, dB );

    Eigen::VectorXd res;
    res.resize( vecSize );
    double* dX = nullptr;
    CHECK_CUDA_RES( cudaMalloc( ( void** )&dX, vecSize * sizeof( double ) ) );

    int singularity;
    CHECK_CUSOLVER_RES( cusolverSpDcsrlsvchol( cusolverHandle_, matANumRows_, numNonZero_, descrA_,
        dVal_, dRow_, dCol_, dB,
        0., 1, dX, &singularity ) );

    denseVectorCudaToEigen_( dX, vecSize, res );
    return res;
}

}
