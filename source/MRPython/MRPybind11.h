#pragma once

#include <MRPch/MRSuppressWarning.h>

MR_SUPPRESS_WARNING_PUSH
#if defined( __clang__ )
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#if defined( _MSC_VER )
#pragma warning( disable: 4100 ) // '_unused_op': unreferenced formal parameter
#pragma warning( disable: 4189 ) // 'has_args': local variable is initialized but not referenced
#pragma warning( disable: 4191 ) // 'reinterpret_cast': unsafe conversion from 'PyObject *(__cdecl *)(PyObject *,PyObject *,PyObject *)' to 'void (__cdecl *)(void)'
#pragma warning( disable: 4355 ) // 'this': used in base member initializer list
#pragma warning( disable: 4464 ) // relative include path contains '..'
#pragma warning( disable: 4548 ) // expression before comma has no effect; expected expression with side-effect
#endif

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

MR_SUPPRESS_WARNING_POP
