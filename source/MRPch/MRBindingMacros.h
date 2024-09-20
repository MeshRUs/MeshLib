#pragma once

// Those macros help control Python bindings generated using MRBind.

// MR_PARSING_FOR_PB11_BINDINGS - gets defined when parsing the source code
// MR_COMPILING_PB11_BINDINGS - gets defined when compiling the resulting bindings

#ifdef MR_PARSING_FOR_PB11_BINDINGS

// Use to specify valid template arguments for templates (usually function templates).
// For example:
//   template <typename T> void foo(T) {...};
//   MR_BIND_TEMPLATE( void foo(int) )
//   MR_BIND_TEMPLATE( void foo(float) )
//
// As with `extern template`, you might need to use `foo<...>` instead of `foo` if the template parameters can't be deduced from the
// parameter types and the return type.
#define MR_BIND_TEMPLATE(...) extern template __VA_ARGS__;

// Mark a declaration with this to avoid generating a binding for it.
#define MR_BIND_IGNORE __attribute__((__annotate__("mrbind::ignore")))

#else
#define MR_BIND_TEMPLATE(...)
#define MR_BIND_IGNORE
#endif
