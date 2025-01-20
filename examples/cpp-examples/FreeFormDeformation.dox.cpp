#include <MRMesh/MRFreeFormDeformer.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

int main()
{
    // Load mesh
    auto mesh = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    assert( mesh );

    // Compute mesh bounding box
    const auto box = mesh->computeBoundingBox();

    // Construct deformer on mesh vertices
    MR::FreeFormDeformer ffDeformer( mesh->points, mesh->topology.getValidVerts() );

    // Init deformer with 3x3 grid on mesh box
    ffDeformer.init( MR::Vector3i::diagonal( 3 ), box );

    // Move some control points of the grid to the center
    ffDeformer.setRefGridPointPosition( { 1, 1, 0 }, box.center() );
    ffDeformer.setRefGridPointPosition( { 1, 1, 2 }, box.center() );
    ffDeformer.setRefGridPointPosition( { 0, 1, 1 }, box.center() );
    ffDeformer.setRefGridPointPosition( { 2, 1, 1 }, box.center() );
    ffDeformer.setRefGridPointPosition( { 1, 0, 1 }, box.center() );
    ffDeformer.setRefGridPointPosition( { 1, 2, 1 }, box.center() );

    // Apply the deformation to the mesh vertices
    ffDeformer.apply();

    // Invalidate the mesh because of external vertex changes
    mesh->invalidateCaches();

    // Save deformed mesh
    MR::MeshSave::toAnySupportedFormat( *mesh, "deformed_mesh.stl" );
}
