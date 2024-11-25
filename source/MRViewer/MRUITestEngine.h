#pragma once

#include "MRMesh/MRExpected.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/exports.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string_view>
#include <string>
#include <variant>
#include <vector>

// This is a low-level header for implementing GUIs that can be interacted with programmatically.
// Most likely you don't need to touch this, just use widgets from `MRUIStyle.h`.

namespace MR::UI::TestEngine
{

namespace detail
{
    template <typename T>
    struct BoundedValue
    {
        T value{};
        T min{};
        T max{};
    };
    template <>
    struct BoundedValue<std::string>
    {
        std::string value;

        std::optional<std::vector<std::string>> allowedValues;
    };

    template <typename T>
    [[nodiscard]] MRVIEWER_API std::optional<T> createValueLow( std::string_view name, std::optional<BoundedValue<T>> value );

    extern template MRVIEWER_API std::optional<std::int64_t> createValueLow( std::string_view name, std::optional<BoundedValue<std::int64_t>> value );
    extern template MRVIEWER_API std::optional<std::uint64_t> createValueLow( std::string_view name, std::optional<BoundedValue<std::uint64_t>> value );
    extern template MRVIEWER_API std::optional<double> createValueLow( std::string_view name, std::optional<BoundedValue<double>> value );
    extern template MRVIEWER_API std::optional<std::string> createValueLow( std::string_view name, std::optional<BoundedValue<std::string>> value );

    template <typename T>
    using UnderlyingValueType = std::conditional_t<std::is_floating_point_v<T>, double, std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>>;
}

// Call this every frame when drawing a button you want to track (regardless of whether it returns true of false).
// If this returns true, simulate a button click.
[[nodiscard]] MRVIEWER_API bool createButton( std::string_view name );

template <typename T>
concept AllowedValueType = std::is_arithmetic_v<T> || std::is_same_v<T, std::string>;

// Create a "value" (slider/drag/...).
// `T` must be a scalar; vector support must be implemented manually.
// Pass `min >= max` to disable the range checks.
// If this returns true, use the new value in place of the current one.
template <AllowedValueType T>
requires std::is_arithmetic_v<T>
[[nodiscard]] std::optional<T> createValue( std::string_view name, T value, T min, T max )
{
    if ( !( min < max ) )
    {
        min = std::numeric_limits<T>::lowest();
        max = std::numeric_limits<T>::max();
    }

    using U = detail::UnderlyingValueType<T>;
    static_assert(sizeof(T) <= sizeof(U), "The used type is too large.");

    auto ret = detail::createValueLow<U>( name, detail::BoundedValue<U>{ .value = U( value ), .min = U( min ), .max = U( max ) } );
    return ret ? std::optional<T>( T( *ret ) ) : std::nullopt;
}
// This overload is for strings.
[[nodiscard]] MRVIEWER_API std::optional<std::string> createValue( std::string_view name, std::string value, std::optional<std::vector<std::string>> allowedValues = std::nullopt );

// Usually you don't need this function.
// This is for widgets that require you to specify the value override before drawing it, such as `ImGui::CollapsingHeader()`.
// For those, call this version first to read the value override, then draw the widget, then call the normal `CreateValue()` with the same name
//   and with the new value, and discard its return value.
template <AllowedValueType T>
[[nodiscard]] std::optional<T> createValueTentative( std::string_view name )
{
    auto ret = detail::createValueLow<detail::UnderlyingValueType<T>>( name, std::nullopt );
    return ret ? std::optional<T>( T( *ret ) ) : std::nullopt;
}

// Use those to group buttons into named groups.
MRVIEWER_API void pushTree( std::string_view name );
MRVIEWER_API void popTree();

struct Entry;

struct ButtonEntry
{
    // Set this to true to simulate a button click.
    mutable bool simulateClick = false;

    static constexpr std::string_view kindName = "button";
};

// For sliders, drags, etc.
struct ValueEntry
{
    template <typename T>
    struct Value
    {
        // The current value.
        T value = 0;

        // Min/max bounds, INCLUSIVE. If none, those are set to the min/max values representable in this type.
        T min = 0;
        T max = 0;

        // Set to override the value.
        mutable std::optional<T> simulatedValue;

        Value() {} // Make `std::variant` below happy.
    };
    template <std::same_as<std::string> T> // GCC chokes on full specializations at class scope, hence this.
    struct Value<T>
    {
        // The current value.
        std::string value;

        std::optional<std::vector<std::string>> allowedValues;

        // Set to override the value.
        mutable std::optional<std::string> simulatedValue;

        Value() {} // Make `std::variant` below happy.
    };
    using ValueVar = std::variant<Value<std::int64_t>, Value<std::uint64_t>, Value<double>, Value<std::string>>;
    ValueVar value;

    static constexpr std::string_view kindName = "value";
};

struct GroupEntry
{
    // Using `std::map` over `std::unordered_map` to be able to search by `std::string_view` keys directly.
    std::map<std::string, Entry, std::less<>> elems;

    static constexpr std::string_view kindName = "group";
};

struct Entry
{
    std::variant<ButtonEntry, ValueEntry, GroupEntry> value;

    // Mostly for internal use.
    // If this is false, the entry will be removed on the next frame.
    bool visitedOnThisFrame = false;

    // Returns a string describing the type currently stored in `value`, which is `T::kindName`.
    [[nodiscard]] MRVIEWER_API std::string_view getKindName() const;

    // Calls `std::get<T>(value)`, returns a user-friendly error on failure.
    // The returned pointer is never null.
    // If `selfName` is specified, it's added to the error message as the name of this entry.
    template <typename T>
    [[nodiscard]] Expected<T *> getAs( std::string_view selfName = {} )
    {
        Expected<T *> ret = std::get_if<T>( &value );
        if ( !*ret )
        {
            if ( selfName.empty() )
                ret = unexpected( fmt::format( "Expected UI entity to be a `{}` but got a `{}`.", T::kindName, getKindName() ) );
            else
                ret = unexpected( fmt::format( "Expected UI entity `{}` to be a `{}` but got a `{}`.", selfName, T::kindName, getKindName() ) );
        }
        return ret;
    }
    template <typename T>
    [[nodiscard]] Expected<const T *> getAs( std::string_view selfName = {} ) const
    {
        return const_cast<Entry *>( this )->template getAs<T>( selfName );
    }
};

// Returns the current entry tree.
[[nodiscard]] MRVIEWER_API const GroupEntry& getRootEntry();

}
