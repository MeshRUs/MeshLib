#pragma once

#include "MRViewer/exports.h"
#include "MRViewer/MRVectorTraits.h"

#include <cassert>
#include <optional>
#include <string>

namespace MR
{

// A stub measurement unit representing no unit.
enum class NoUnit
{
    _count [[maybe_unused]]
};

// Measurement units of length.
enum class LengthUnit
{
    mm,
    inches,
    _count [[maybe_unused]],
};

// Measurement units of angle.
enum class AngleUnit
{
    radians,
    degrees,
    _count [[maybe_unused]],
};

// A list of all unit enums, for internal use.
#define DETAIL_MR_UNIT_ENUMS(X) X(NoUnit) X(LengthUnit) X(AngleUnit)

// Whether `E` is one of the unit enums: NoUnit, LengthUnit, AngleUnit, ...
template <typename T>
concept UnitEnum =
    #define MR_X(E) || std::same_as<T, E>
    true DETAIL_MR_UNIT_ENUMS(MR_X);
    #undef MR_X

// ---

// Information about a single measurement unit.
struct UnitInfo
{
    // This is used to convert between units.
    // To convert from A to B, multiply by A's factor and divide by B's.
    float conversionFactor = 1;

    std::string_view prettyName;

    // The short unit name that's placed after values.
    // This may or may not start with a space.
    std::string_view unitSuffix;
};

// Returns information about a single measurement unit.
template <UnitEnum E>
[[nodiscard]] const UnitInfo& getUnitInfo( E unit ) = delete;

#define MR_X(E) template <> [[nodiscard]] MRVIEWER_API const UnitInfo& getUnitInfo( E unit );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

// Converts `value` from unit `from` to unit `to`. `value` is a floating-point number of a Vector2/3/4 or ImVec2/4 of them.
template <UnitEnum E, typename T>
requires std::is_floating_point_v<typename VectorTraits<T>::BaseType>
[[nodiscard]] constexpr T convertUnits( E from, E to, T value )
{
    if ( from != to )
    {
        for ( int i = 0; i < VectorTraits<T>::size; i++ )
        {
            auto& target = VectorTraits<T>::getElem( i, value );
            target = target * getUnitInfo( from ).conversionFactor / getUnitInfo( to ).conversionFactor;
        }
    }
    return value;
}

template <UnitEnum E>
struct UnitToStringParams;

// Returns the default parameters for converting a specific unit type to a string.
// You can modify those with `setDefaultUnitParams()`.
template <UnitEnum E>
[[nodiscard]] const UnitToStringParams<E>& getDefaultUnitParams();

#define MR_X(E) extern template MRVIEWER_API const UnitToStringParams<E>& getDefaultUnitParams();
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

// Modifies the default parameters for converting a specific unit type to a string.
template <UnitEnum E>
void setDefaultUnitParams( const UnitToStringParams<E>& newParams );

#define MR_X(E) extern template MRVIEWER_API void setDefaultUnitParams( const UnitToStringParams<E>& newParams );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

// This controls how the degrees are printed.
enum class DegreesMode
{
    degrees, // Fractional degrees.
    degreesMinutes, // Integral degrees, fractional arcminutes.
    degreesMinutesSeconds, // Integral degrees and minutes, fractional arcseconds.
};

// How the trailing zeroes are stripped.
// All of this only applies if the number has a decimal point.
enum class TrailingZeroes
{
    keep, // Don't touch trailing zeroes.
    stripAndKeepOne, // Strip trailing zeroes, but if the last character is `.` after that, add one zero back.
    stripAll, // Strip trailing zeroes unconditionally.
};

namespace detail::Units
{
    struct Empty {};
}

// Controls how a value with a unit is converted to a string.
template <UnitEnum E>
struct UnitToStringParams
{
    // --- Units:

    // The measurement unit of the input.
    // If null, assumed to be the same as `targetUnit`, and no conversion is performed.
    // If not null, the value is converted from this unit to `targetUnit`.
    std::optional<E> sourceUnit = getDefaultUnitParams<E>().sourceUnit;

    // The measurement unit of the result.
    E targetUnit = getDefaultUnitParams<E>().targetUnit;

    // Whether to show the unit suffix.
    bool unitSuffix = getDefaultUnitParams<E>().unitSuffix;

    // --- Precision:

    // If true, `precision` is the total number of digits. If false, `precision` is the number of digits after the decimal point.
    bool fixedPrecision = getDefaultUnitParams<E>().fixedPrecision;

    // How many digits of precision.
    int precision = getDefaultUnitParams<E>().precision;

    // --- Other:

    // Use a pretty Unicode minus sign instead of the ASCII `-`.
    bool unicodeMinusSign = getDefaultUnitParams<E>().unicodeMinusSign;

    // If non-zero, this character is inserted between every three digits.
    char thousandsSeparator = getDefaultUnitParams<E>().thousandsSeparator;

    // If false, remove zero before the fractional point (`.5` instead of `0.5`).
    bool leadingZero = getDefaultUnitParams<E>().leadingZero;

    // Remove trailing zeroes after the fractional point. If the point becomes the last symbol, remove the point too.
    bool stripTrailingZeroes = getDefaultUnitParams<E>().stripTrailingZeroes;

    // When printing degrees, this lets you display arcminutes and possibly arcseconds. Ignored for everything else.
    std::conditional_t<std::is_same_v<E, AngleUnit>, DegreesMode, detail::Units::Empty> degreesMode = getDefaultUnitParams<E>().degreesMode;

    // If you add new fields there, update the initializer for `defaultUnitToStringParams` in `MRUnits.cpp`.

    friend bool operator==( const UnitToStringParams&, const UnitToStringParams& ) = default;
};

// Converts value to a string, possibly converting it to a different unit.
// By default, length is kept as is, while angles are converted from radians to the current UI unit.
template <UnitEnum E>
std::string valueToString( float value, const UnitToStringParams<E>& params = getDefaultUnitParams<E>() );

#define MR_X(E) extern template MRVIEWER_API std::string valueToString( float value, const UnitToStringParams<E>& params = getDefaultUnitParams<E>() );
DETAIL_MR_UNIT_ENUMS(MR_X)
#undef MR_X

}
