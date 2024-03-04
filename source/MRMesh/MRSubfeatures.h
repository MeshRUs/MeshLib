#pragma once

#include "MRMesh/MRFeatures.h"

#include <functional>

namespace MR::Features
{

// Describes a single feature produced by `forEachSubfeature()`.
struct SubfeatureInfo
{
    // A user-friendly name.
    std::string_view name;

    // Whether the feature has infinite length.
    bool isInfinite = false;

    // Call this to create this subfeature.
    std::function<Primitives::Variant()> create;
};

// A user callback for `forEachSubfeature()`.
using SubfeatureFunc = std::function<void( const SubfeatureInfo& info )>;

// Decomposes a feature to its subfeatures, by calling `func()` on each subfeature.
MRMESH_API void forEachSubfeature( const Features::Primitives::Variant& feature, const SubfeatureFunc& func );

}
