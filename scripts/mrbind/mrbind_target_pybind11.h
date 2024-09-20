// Intentionally no pragma once, though currently it doesn't matter (see the mrbind header included below).

#include <mrbind/targets/pybind11.h>

#include "MRPch/MRJson.h"

// Disable all functions accepting or returning `Json::Value`.

template <typename T> requires std::is_same_v<typename MRBind::detail::pb11::StripToUnderlyingType<T>::type, Json::Value>
struct MRBind::detail::pb11::ReturnTypeTraits<T>
{
    using no_adjust = void;
    using disables_func = void;
};

template <typename T> requires std::is_same_v<typename MRBind::detail::pb11::StripToUnderlyingType<T>::type, Json::Value>
struct MRBind::detail::pb11::ParamTraits<T>
{
    using disables_func = void;
};
