#pragma once

#include "MRMesh/MRMacros.h"

// This helper macro is used to declare canonical typedefs. For example, if you have this:
//     template <typename T> struct MRMESH_CLASS Vector3;  // `MRMESH_CLASS` isn't necessary, just to demonstrate that it's supported.
//     using Vector3f = Vector3<float>;
//     using Vector3i = Vector3<int>;
// You should rewrite it as following:
//     MR_CANONICAL_TYPEDEFS( (template <typename T> struct MRMESH_CLASS), Vector3,
//         ( Vector3f, Vector3<float> )
//         ( Vector3i, Vector3<int> )
//     )
//
// WHAT THIS ACHIEVES?
//   This macro only has effect on Clang.
//   It primarily affects the error messages, using those typedefs instead of the full type when possible.
//   It also helps when parsing our code with libclang to automatically generate Python bindings.
//
// NOTE:
// * Only non-template `using`s can be declared like this. If you want to add additional templated ones, add them below the macro manually.

#define MR_CANONICAL_TYPEDEFS(type_, name_, aliases_) \
    MR_IDENTITY type_ name_; \
    MR_END(DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_A aliases_) \
    DETAIL_MR_CANONICAL_TYPEDEFS(type_, name_, aliases_)

#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_BODY(name_, ...) using name_ = __VA_ARGS__;
// `MR_IDENTITY` here keeps the legacy MSVC preprocessor happy.
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_A(...) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_BODY MR_IDENTITY()(__VA_ARGS__) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_B
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_B(...) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_BODY MR_IDENTITY()(__VA_ARGS__) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_A
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_A_END
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_USING_B_END

#if defined(__has_attribute)
#if __has_attribute(__preferred_name__)
#define DETAIL_MR_CANONICAL_TYPEDEFS(type_, name_, aliases_) \
    MR_IDENTITY type_ \
        MR_END(DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_A aliases_) \
        name_; \
    DETAIL_MR_CANONICAL_TYPEDEFS_CLANG_WORKAROUND(aliases_)

#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_BODY(name_, ...) __attribute__((__preferred_name__(name_)))
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_A(...) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_BODY(__VA_ARGS__) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_B
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_B(...) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_BODY(__VA_ARGS__) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_A
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_A_END
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_ATTR_B_END

#ifdef __clang__ // Workaround for bug: https://github.com/llvm/llvm-project/issues/106358

#define DETAIL_MR_CANONICAL_TYPEDEFS_CLANG_WORKAROUND(aliases_) \
    MR_END(DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_A aliases_)

namespace MR::detail::CanonicalTypedefs
{
template <typename> struct RegisterType {};
}

#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_BODY(name_, ...) static_assert((void(::MR::detail::CanonicalTypedefs::RegisterType<__VA_ARGS__>{}), true));
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_A(...) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_BODY(__VA_ARGS__) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_B
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_B(...) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_BODY(__VA_ARGS__) DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_A
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_A_END
#define DETAIL_MR_CANONICAL_TYPEDEFS_LOOP_TOUCH_B_END

#else // no workaround needed
#define DETAIL_MR_CANONICAL_TYPEDEFS_CLANG_WORKAROUND(aliases_)
#endif

#else // this attribute is not supported
#define DETAIL_MR_CANONICAL_TYPEDEFS(type_, name_, aliases_)
#endif

#else // no __has_attribute
#define DETAIL_MR_CANONICAL_TYPEDEFS(type_, name_, aliases_)
#endif
