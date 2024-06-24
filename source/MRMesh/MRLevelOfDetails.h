#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// divides given mesh into hierarchy of mesh objects:
/// the deepest level of the hierarchy has all details from the original mesh;
/// top levels have progressively less number of objects and less details in each;
/// the number of faces in any object on any level is about the same.
[[nodiscard]] MRMESH_API std::shared_ptr<Object> makeLevelOfDetails( Mesh && mesh, int maxDepth );

} //namespace MR
