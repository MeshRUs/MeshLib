#include <MRMesh/MRBox.h>
#include <MRMesh/MRExpandShrink.h>
#include <MRMesh/MRLaplacian.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

int main()
{
    // Load mesh
    auto mesh = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    assert( mesh );

    // Construct deformer on the mesh vertices
    MR::Laplacian lDeformer( *mesh );

    // Find an area for the deformation anchor points
    const auto ancV0 = mesh->topology.getValidVerts().find_first();
    const auto ancV1 = mesh->topology.getValidVerts().find_last();
    // Mark the anchor points in the free area
    MR::VertBitSet freeVerts;
    freeVerts.resize( mesh->topology.getValidVerts().size() );
    freeVerts.set( ancV0, true );
    freeVerts.set( ancV1, true );
    // Expand the free area
    MR::expand( mesh->topology, freeVerts, 5 );

    // Initialize laplacian
    lDeformer.init( freeVerts, MR::EdgeWeights::CotanWithAreaEqWeight );

    const auto shiftAmount = mesh->computeBoundingBox().diagonal() * 0.01f;
    // Fix the anchor vertices in the required position
    lDeformer.fixVertex( ancV0, mesh->points[ancV0] + mesh->normal( ancV0 ) * shiftAmount );
    lDeformer.fixVertex( ancV1, mesh->points[ancV1] + mesh->normal( ancV1 ) * shiftAmount );

    // Move the free vertices according to the anchor ones
    lDeformer.apply();

    // Invalidate the mesh because of the external vertex changes
    mesh->invalidateCaches();

    // Save the deformed mesh
    MR::MeshSave::toAnySupportedFormat( *mesh, "deformed_mesh.stl" );
}
