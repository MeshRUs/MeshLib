#pragma once

#include "MRMesh/MRFlagOperators.h"
#include "MRViewer/exports.h"

#include <imgui.h>

#include <optional>
#include <span>

namespace MR::ImGuiMeasurementIndicators
{

struct Params
{
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    Color colorMain;
    Color colorOutline;
    Color colorText;

    float width = 1.5f;
    float smallWidth = 0.75f;
    float outlineWidth = 1.5f;
    float textOutlineWidth = 4.f;
    float textOutlineRounding = 3.f;

    float arrowLen = 12;
    float arrowHalfWidth = 4;

    // The spacing box around the text is extended by this amount.
    ImVec2 textToLineSpacingA = ImVec2( 0, 0 ); // Top-left corner.
    ImVec2 textToLineSpacingB = ImVec2( 0, 2 ); // Bottom-right corner.
    // Further, the lines around the text are shortened by this amount.
    float textToLineSpacingRadius = 8;

    // For a distance, when the total length is less than this, an alternative rendering method is used.
    // If the text is added, its size can make the actual threshold larger.
    float totalLenThreshold = 48;

    // When drawing inverted (short) distances, the line is extended by this amount on both sides.
    float invertedOverhang = 24;

    // The length of the leader lines (short lines attaching text to other things).
    float leaderLineLen = 20;

    // This picks the colors based on the current color theme.
    MRVIEWER_API Params();
};

enum class Element
{
    main = 1 << 0,
    outline = 1 << 1,
    both = main | outline, // Use this by default.
};
MR_MAKE_FLAG_OPERATORS( Element )

enum class StringIcon
{
    none,
    diameter,
};

// A string with an optional custom icon inserted in the middle.
struct StringWithIcon
{
    StringIcon icon{};
    std::size_t iconPos = 0;

    std::string string;

    constexpr StringWithIcon() {}

    // Need a bunch of constructors to allow implicit conversions when this is used as a function parameter.
    StringWithIcon( const char* string ) : string( string ) {}
    StringWithIcon( std::string string ) : string( std::move( string ) ) {}

    constexpr StringWithIcon( StringIcon icon, std::size_t iconPos, std::string_view string )
        : icon( icon ), iconPos( iconPos ), string( string )
    {}

    [[nodiscard]] constexpr bool isEmpty() const { return icon == StringIcon::none && string.empty(); }

    // Returns the icon width, already scaled to the current scale.
    [[nodiscard]] MRVIEWER_API float getIconWidth() const;
    // Calculates the text size with the icon, using the current font.
    [[nodiscard]] MRVIEWER_API ImVec2 calcTextSize() const;
    // Draws the text with an icon to the specified draw list.
    MRVIEWER_API void draw( ImDrawList& list, float menuScaling, ImVec2 pos, ImU32 color );
};

// Draws a floating text bubble.
// If `push` is specified, it's normalized and the text is pushed in that direction,
// by the amount necessarily to clear a perpendicular going through the center point.
MRVIEWER_API void text( Element elem, float menuScaling, const Params& params, ImVec2 center, StringWithIcon string, ImVec2 push = ImVec2() );

// Draws a triangle from an arrow.
MRVIEWER_API void arrowTriangle( Element elem, float menuScaling, const Params& params, ImVec2 point, ImVec2 dir );

struct LineCap
{
    enum class Decoration
    {
        none,
        arrow,
    };
    Decoration decoration{};

    StringWithIcon text;
};

enum class LineFlags
{
    narrow = 1 << 0,
};
MR_MAKE_FLAG_OPERATORS( LineFlags )

struct LineParams
{
    LineFlags flags{};

    LineCap capA{};
    LineCap capB{};

    std::span<const ImVec2> midPoints;
};

// Draws a line or an arrow.
MRVIEWER_API void line( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, const LineParams& lineParams = {} );

struct DistanceParams
{
    // If this is set, the text is moved from the middle of the line to one of the line ends (false = A, true = B).
    std::optional<bool> moveTextToLineEndIndex;
};

// Draws a distance arrow between two points, automatically selecting the best visual style.
// The `string` is optional.
MRVIEWER_API void distance( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, StringWithIcon string, const DistanceParams& distanceParams = {} );

} // namespace MR::ImGuiMeasurementIndicators
