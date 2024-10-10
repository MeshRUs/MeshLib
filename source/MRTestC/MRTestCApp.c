#include "MRBitSet.h"
#include "MRMeshBoolean.h"
#include "MRMeshCollidePrecise.h"
#include "MRMeshDecimate.h"
#include "MRMeshFillHole.h"
#include "MRMeshNormals.h"

int main( void )
{
    testBitSet();
    testMeshBoolean();
    testBooleanMultipleEdgePropogationSort();
    testMeshCollidePrecise();
    testMeshDecimate();
    testMeshFillHole();
    testMeshNormals();
}
