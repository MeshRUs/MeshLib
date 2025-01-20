#include <MRMesh/MRAddNoise.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRNormalDenoising.h>

int main()
{
    // Load mesh
    auto mesh = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    assert( mesh );

    // Add noise to the mesh
    MR::addNoise( mesh->points, mesh->topology.getValidVerts(), {
        .sigma = mesh->computeBoundingBox().diagonal() * 0.0001f,
    } );

    // Invalidate the mesh because of the external vertex changes
    mesh->invalidateCaches();

    // Save the noised mesh
    MR::MeshSave::toAnySupportedFormat( *mesh, "noised_mesh.stl" );

    // Denoise the mesh with sharpening for sharp edges
    // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
    MR::meshDenoiseViaNormals( *mesh );

    // Save the denoised mesh
    MR::MeshSave::toAnySupportedFormat( *mesh, "denoised_mesh.stl" );
}
