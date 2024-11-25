#include <MRMesh/MRMeshFwd.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRBuffer.h>

int main()
{
    // Load mesh
    MR::Mesh mesh = *MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    
    // Repack mesh optimally.
    // It's not necessary but highly recommended to achieve the best performance in parallel processing
    mesh.packOptimally();

    // Setup decimate parameters
    MR::DecimateSettings settings;

    // Decimation stop thresholds, you may specify one or both
    settings.maxDeletedFaces = 1000; // Number of faces to be deleted
    settings.maxError = 0.05f; // Maximum error when decimation stops

    // Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
    // Recommended to set to the number of available CPU cores or more for the best performance
    settings.subdivideParts = 64;

    // Decimate mesh
    MR::decimateMesh( mesh, settings );

    // Save result
    MR::MeshSave::toAnySupportedFormat( mesh, "decimated_mesh.stl" );

    return 0;
}
