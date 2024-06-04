#include "MRSceneCache.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRTimer.h"

#include <iostream>

namespace MR
{

void SceneCache::invalidateAll()
{
    for ( auto& data : instance_().cachedData_ )
        data.second.reset();
}

MR::SceneCache& SceneCache::instance_()
{
    static SceneCache sceneCahce;
    return sceneCahce;
}

}
