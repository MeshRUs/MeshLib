#pragma once

#include "MRMeshFwd.h"
#include "MRPch/MRBindingMacros.h"

#include <version>
#if __cpp_lib_expected >= 202211
#include <expected>
#else
#include <tl/expected.hpp>
/// we have C++/CLI project MRMeshDotNet which doesn't support std::expected
/// So we have to wrap tl::expected with this class in std namespace for correct linking
#ifdef MR_DOT_NET_BUILD
namespace std
{
template<typename T, typename E>
class expected : public tl::expected<T, E>
{
    using tl::expected<T, E>::expected;
};

template <class E>
inline auto unexpected( E &&e )
{
    return tl::make_unexpected( std::forward<E>( e ) );
}
}
#endif
#endif

#include <string>

namespace MR
{

#if ( __cpp_lib_expected >= 202211  ||  defined( MR_DOT_NET_BUILD ) )

template<class T, class E = std::string>
using Expected = std::expected<T, E>;

template <class E>
MR_BIND_IGNORE inline auto unexpected( E &&e )
{
    return std::unexpected( std::forward<E>( e ) );
}

#else

template<class T, class E = std::string>
using Expected = tl::expected<T, E>;

template <class E>
MR_BIND_IGNORE inline auto unexpected( E &&e )
{
    return tl::make_unexpected( std::forward<E>( e ) );
}

#endif

/// return type for a void function that can produce an error string
using VoidOrErrStr = Expected<void>;

/// Common operation canceled line for all
MR_BIND_IGNORE inline std::string stringOperationCanceled()
{
    return "Operation was canceled";
}

/// Returns Expected error with `stringOperationCanceled()`
MR_BIND_IGNORE inline auto unexpectedOperationCanceled()
{
    return MR::unexpected(stringOperationCanceled());
}

} //namespace MR
