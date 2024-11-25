// Intentionally no pragma once, to allow re-including multiple times with different values of `MB_PB11_STAGE`.

#include <mrbind/targets/pybind11.h>

// A scope guard doesn't include the above header, because this header can be included multiple times, and then `pybind11.h` must also be included multiple times.
#ifndef MR_MRBIND_TARGET_PYBIND11_H_
#define MR_MRBIND_TARGET_PYBIND11_H_

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRSignal.h"
#include "MRPch/MREigen.h"
#include "MRPch/MRJson.h"

#include <boost/multiprecision/number.hpp>

// Disable all functions accepting some problematic types.

namespace MR::MrbindDetail
{
    template <typename T>
    struct IsEigenMatrix : std::false_type {};
    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct IsEigenMatrix<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> : std::true_type {};

    template <typename T>
    struct IsSignal : std::false_type {};
    template <typename T>
    struct IsSignal<Signal<T>> : std::true_type {};

    template <typename T>
    struct IgnoreTypeInBindings : std::bool_constant<
        std::same_as<T, Json::Value> ||
        IsEigenMatrix<T>::value ||
        boost::multiprecision::is_number<T>::value ||
        IsSignal<T>::value
    > {};

    // Recurse into `tl::Expected`.
    template <typename T, typename U>
    struct IgnoreTypeInBindings<Expected<T, U>> : std::bool_constant<IgnoreTypeInBindings<T>::value || IgnoreTypeInBindings<U>::value> {};

    // Recurse into vectors and matrices.
    template <typename T> struct IgnoreTypeInBindings<Vector2<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Vector3<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Vector4<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Matrix2<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Matrix3<T>> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<Matrix4<T>> : IgnoreTypeInBindings<T> {};

    // Recurse into qualifiers.
    template <typename T> struct IgnoreTypeInBindings<const T> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<T *> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<T &> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<T &&> : IgnoreTypeInBindings<T> {};
    template <typename T> struct IgnoreTypeInBindings<std::shared_ptr<T>> : IgnoreTypeInBindings<T> {};
    template <typename T, typename D> struct IgnoreTypeInBindings<std::unique_ptr<T, D>> : IgnoreTypeInBindings<T> {};
}

template <typename T> requires MR::MrbindDetail::IgnoreTypeInBindings<T>::value
struct MRBind::pb11::IgnoreFuncsWithReturnType<T> : std::true_type {};

template <typename T> requires MR::MrbindDetail::IgnoreTypeInBindings<T>::value
struct MRBind::pb11::ParamTraits<T>
{
    using disables_func = void;
};

template <typename T> requires MR::MrbindDetail::IgnoreTypeInBindings<T>::value
struct MRBind::pb11::IgnoreFieldsWithType<T> : std::true_type {};

#endif
