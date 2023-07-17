#include "MRCudaPointsToMeshProjector.h"
#include "MRCudaPointsToMeshProjector.cuh"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRAABBTree.h"
#include "MRMesh/MRMatrix3Decompose.h"
#include "MRMesh/MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace Cuda
{

struct MeshProjectorData
{
    DynamicArray<float3> cudaPoints;
    DynamicArray<MeshProjectionResult> cudaResult;
    DynamicArray<float3> cudaMeshPoints;
    DynamicArray<Node3> cudaNodes;
    DynamicArray<FaceToThreeVerts> cudaFaces;

    Matrix4 xf;
    Matrix4 refXf;
};

PointsToMeshProjector::PointsToMeshProjector()
{
    meshData_ = std::make_shared<MeshProjectorData>();
}

void PointsToMeshProjector::updateMeshData( const MR::Mesh* mesh )
{
    if ( !mesh )
    {
        mesh_ = nullptr;
        return;
    }

    const AABBTree& tree = mesh->getAABBTree();
    const auto& nodes = tree.nodes();
    const auto& meshPoints = mesh->points;
    const auto tris = mesh->topology.getTriangulation();

    meshData_->cudaMeshPoints.fromVector( meshPoints.vec_ );
    meshData_->cudaNodes.fromVector( nodes.vec_ );
    meshData_->cudaFaces.fromVector( tris.vec_ );

    mesh_ = mesh;
}

void PointsToMeshProjector::findProjections(
      std::vector<MR::MeshProjectionResult>& res, const std::vector<Vector3f>& points, const AffineXf3f* objXf, const AffineXf3f* refObjXf, float upDistLimitSq, float loDistLimitSq )
{
    MR_TIMER
    if ( !mesh_ )
    {
        assert( false );
        return;
    }
    
    CUDA_EXEC( cudaSetDevice( 0 ) );

    const auto getCudaMatrix = [] ( const AffineXf3f& xf )
    {
        Matrix4 res;
        res.x.x = xf.A.x.x; res.x.y = xf.A.x.y; res.x.z = xf.A.x.z;
        res.y.x = xf.A.y.x; res.y.y = xf.A.y.y; res.y.z = xf.A.y.z;
        res.z.x = xf.A.z.x; res.z.y = xf.A.z.y; res.z.z = xf.A.z.z;
        res.b.x = xf.b.x; res.b.y = xf.b.y; res.b.z = xf.b.z;
        res.isIdentity = false;
        return res;
    };

    const AffineXf3f* notRigidRefXf{ nullptr };
    if ( refObjXf && !isRigid( refObjXf->A ) )
        notRigidRefXf = refObjXf;

    AffineXf3f xf;
    const AffineXf3f* xfPtr{ nullptr };
    if ( notRigidRefXf || !refObjXf )
        xfPtr = objXf;
    else
    {
        xf = refObjXf->inverse();
        if ( objXf )
            xf = xf * ( *objXf );
        xfPtr = &xf;
    }

    if ( notRigidRefXf )
        meshData_->refXf = getCudaMatrix( *notRigidRefXf );
    if ( xfPtr )
        meshData_->xf = getCudaMatrix( *xfPtr );
    
    const size_t size = points.size();
    res.resize( size );
    meshData_->cudaPoints.fromVector( points );
    meshData_->cudaResult.resize( size );

   std::ofstream f( "c:/temp/cudaproj.txt" );
   f << "threads,time" << std::endl;
   for ( int i = 4; i <= 1024; i += 4 )
   {
       Timer t( "kernel" );
        meshProjectionKernel( i, meshData_->cudaPoints.data(), meshData_->cudaNodes.data(), meshData_->cudaMeshPoints.data(), meshData_->cudaFaces.data(), meshData_->cudaResult.data(), meshData_->xf, meshData_->refXf, upDistLimitSq, loDistLimitSq, size );
       f << i << ',';
       if ( cudaDeviceSynchronize() )
           f << "error";
        else
           f << t.secondsPassed().count();
       f << std::endl;
       std::this_thread::sleep_for( std::chrono::duration<float>( 5 ) );
   }

    meshData_->cudaResult.toVector( res );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, res.size() ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( res[i].proj.face )
                res[i].mtp.e = mesh_->topology.edgeWithLeft( res[i].proj.face );
            else
                assert( !res[i].mtp.e );
        }
    } );
}

size_t PointsToMeshProjector::projectionsHeapBytes( size_t numProjections ) const
{
    size_t currentSize = 0;
    if ( meshData_ )
    {
        currentSize += meshData_->cudaPoints.size() * sizeof( float3 );
        currentSize += meshData_->cudaResult.size() * sizeof( MeshProjectionResult );
    }
    size_t newSize = numProjections * ( sizeof( float3 ) + sizeof( MeshProjectionResult ) );
    if ( newSize <= currentSize )
        return 0;
    return newSize - currentSize;
}

}

}