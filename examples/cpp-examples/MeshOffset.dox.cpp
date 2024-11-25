#include <MRMesh/MRBox.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRVoxels/MROffset.h>

#include <iostream>

int main()
{
    // Create mesh
    MR::Mesh mesh = MR::makeCube();

    // Setup parameters
    MR::GeneralOffsetParameters params;
    // calculate voxel size depending on desired accuracy and/or memory consumption
    params.voxelSize = suggestVoxelSize( mesh, 10000000.f );
    if ( !MR::findRightBoundary( mesh.topology ).empty() )
        params.signDetectionMode = MR::SignDetectionMode::HoleWindingRule; // use if you have holes in mesh

    // Make offset mesh
    float offset = mesh.computeBoundingBox().diagonal() * 0.1f;
    auto meshRes = MR::generalOffsetMesh( mesh, offset, params );
    if ( !meshRes.has_value() )
    {
        std::cerr << meshRes.error() << std::endl;
        return 1;
    }

    // Save result
    MR::MeshSave::toAnySupportedFormat( *meshRes, "mesh_offset.stl" );

    return 0;
}
