#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshRelax.h>
#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRTorus.h>

int main()
{
    MR::Mesh mesh = MR::makeTorus();

    // Relax mesh (5 iterations)
    MR::relax( mesh, {{ .iterations = 5 }} );

    // Subdivide mesh
    MR::SubdivideSettings props;
    props.maxDeviationAfterFlip = 0.5f;
    MR::subdivideMesh( mesh, props );

    // Rotate mesh
    MR::AffineXf3f rotationXf = MR::AffineXf3f::linear( MR::Matrix3f::rotation( MR::Vector3f::plusZ(), MR::PI_F * 0.5f ) );
    mesh.transform( rotationXf );

    return 0;
}
