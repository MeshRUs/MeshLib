#pragma once

#pragma warning(push)
#pragma warning(disable:4275) // non dll-interface class 'std::runtime_error' used as base for dll-interface class 'fmt::v10::format_error'
#include <spdlog/tweakme.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#pragma warning(pop)

#include "MRPch/MRBindingMacros.h"

namespace MR
{
#if FMT_VERSION < 80000
MR_BIND_IGNORE inline std::string_view runtimeFmt( std::string_view str )
{
    return str;
}
#else
MR_BIND_IGNORE inline auto runtimeFmt( std::string_view str )
{
    return fmt::runtime( str );
}
#endif
}
