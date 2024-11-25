#pragma once

#include "MRMeshFwd.h"
#include <memory>

namespace MR
{

using ObjectPtr = std::shared_ptr<Object>;

/// results of loading e.g. from a file
struct LoadedObjects
{
    std::vector<ObjectPtr> objs;
    std::string warnings;
};

} //namespace MR
