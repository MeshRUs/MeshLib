#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>

#include <iostream>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRMeshDecimate.h>

int main()
{
    auto mesh = MR::MeshLoad::fromAnySupportedFormat( "mesh.stl" );
    if ( !mesh )
    {
        std::cerr << mesh.error() << std::endl;
        return 1;
    }

    // you can set various parameters for the resolving process; see the documentation for more info
    auto changed = MR::resolveMeshDegenerations( *mesh, {
        .maxDeviation = 1e-5f * mesh->computeBoundingBox().diagonal(),
        .tinyEdgeLength = 1e-3f,
    } );

    return 0;
}
