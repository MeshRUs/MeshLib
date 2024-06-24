#pragma once
#include "MRMesh/MRFinally.h"
#include "MRPch/MRFmt.h"
#include "MRViewer/MRUnits.h"
#include "MRViewer/MRVectorTraits.h"
#include "exports.h"
#include "imgui.h"
#include <span>
#include <string>
#include <optional>

namespace MR
{

class ImGuiImage;

namespace UI
{

/// init internal parameters
MRVIEWER_API void init();

/// parameters to customize buttonEx
struct ButtonCustomizationParams
{
    /// gradient texture other than default
    /// {start, hover_start, acitve_start, disabled_start,
    ///  end, hover_end, acitve_end, disabled_end  }
    ImGuiImage* customTexture = nullptr;
    /// force use if ImGuiCol_Text for text
    bool forceImguiTextColor = false;
    /// show border or not
    bool border = false;

    /// Allow interacting with this button from `UI::TestEngine`.
    bool enableTestEngine = true;
};

struct ButtonIconCustomizationParams : public ButtonCustomizationParams
{
    ImGuiButtonFlags flags = ImGuiButtonFlags_None;
    // flag for buttonEx, which can be disabled
    bool active = true;
    // button without a gradient, always ative, configurable by an external style
    bool flatBackgroundColor = false;
};


/// draw gradient button, which can be disabled (active = false)
MRVIEWER_API bool buttonEx( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ),
    ImGuiButtonFlags flags = ImGuiButtonFlags_None, const ButtonCustomizationParams& custmParams = {} );
/// draw gradient button, which can be disabled (active = false)
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool button( const char* label, bool active, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );
/// draw gradient button
/// returns true if button is clicked in this frame, or key is pressed (optional)
inline bool button( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None )
{
    return button( label, true, size, key );
}
/// draw gradient button with the ordinary button size
/// returns true if button is clicked in this frame, or key is pressed (optional)
MRVIEWER_API bool buttonCommonSize( const char* label, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );
/// draw button with same logic as radioButton
MRVIEWER_API bool buttonUnique( const char* label, int* value, int ownValue, const Vector2f& size = Vector2f( 0, 0 ), ImGuiKey key = ImGuiKey_None );

// draw a button with an icon and text under it
MRVIEWER_API bool buttonIconEx( 
    const std::string& name, 
    const Vector2f& iconSize, 
    const std::string& text, 
    const ImVec2& buttonSize, 
    const ButtonIconCustomizationParams& params = {} );
// button with a gradient and the ability to make it inactive
inline bool buttonIcon( const std::string& name, const Vector2f& iconSize, const std::string& text, bool active, const ImVec2& buttonSize )
{
    ButtonIconCustomizationParams params;
    params.active = active;
    params.flatBackgroundColor = true;
    return buttonIconEx(name, iconSize, text, buttonSize, params );
}
// button with a gradient, always ative
inline bool buttonIcon( const std::string& name, const Vector2f& iconSize, const std::string& text, const ImVec2& buttonSize )
{
    return buttonIconEx( name, iconSize, text, buttonSize );
}
// button without a gradient, always ative, configurable by an external style
inline bool buttonIconFlatBG( const std::string& name, const Vector2f& iconSize, const std::string& text, const ImVec2& buttonSize )
{
    ButtonIconCustomizationParams params;
    params.flatBackgroundColor = true;
    params.forceImguiTextColor = true;
    return buttonIconEx( name, iconSize, text, buttonSize, params );
}

/// draw gradient checkbox
MRVIEWER_API bool checkbox( const char* label, bool* value );
/// draw gradient checkbox
/// if valid is false checkbox is disabled
MRVIEWER_API bool checkboxValid( const char* label, bool* value, bool valid );
/// draw gradient checkbox with mixed state
MRVIEWER_API bool checkboxMixed( const char* label, bool* value, bool mixed );
/// draw gradient checkbox
template <typename Getter, typename Setter>
bool checkbox( const char* label, Getter get, Setter set )
{
    bool value = get();
    bool ret = checkbox( label, &value );
    set( value );
    return ret;
}

/// Draw a checkbox toggling one or more bits in the mask.
template <typename T>
bool checkboxFlags( const char* label, T& target, T flags )
{
    bool value = bool( target & flags );
    bool mixed = value && ( target & flags ) != flags;
    if ( checkboxMixed( label, &value, mixed ) )
    {
        if ( value )
            target |= flags;
        else
            target &= ~flags;
        return true;
    }
    return false;
}


/// draw gradient radio button
MRVIEWER_API bool radioButton( const char* label, int* value, int valButton );

/// draw gradient color edit 4
MRVIEWER_API bool colorEdit4( const char* label, Vector4f& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );
MRVIEWER_API bool colorEdit4( const char* label, Color& color, ImGuiColorEditFlags flags = ImGuiColorEditFlags_None );

/// draw combo box
MRVIEWER_API bool combo( const char* label, int* v, const std::vector<std::string>& options,
    bool showPreview = true, const std::vector<std::string>& tooltips = {}, const std::string& defaultText = "Not selected" );

/// draw custom content combo box
MRVIEWER_API bool beginCombo( const char* label, const std::string& text = "Not selected", bool showPreview = true );
MRVIEWER_API void endCombo( bool showPreview = true );


MRVIEWER_API bool sliderFloat( const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0 );
MRVIEWER_API bool sliderInt( const char* label, int* v, int v_min, int v_max, const char* format = "%d", ImGuiSliderFlags flags = 0 );


/// draw input text box with text aligned by center
MRVIEWER_API bool inputTextCentered( const char* label, std::string& str, float width = 0.0f, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL );

/// draw read-only text box with text aligned by center
MRVIEWER_API void inputTextCenteredReadOnly( const char* label, const std::string& str, float width = 0.0f, const std::optional<ImVec4>& textColor = {} );


namespace detail
{
    // A type-erased slider.
    MRVIEWER_API bool genericSlider( const char* label, ImGuiDataType data_type, void* p_data, const void* p_min, const void* p_max, const char* format, ImGuiSliderFlags flags );

    // Whether `T` is a scalar type that we can use with our widgets.
    template <typename T>
    concept Scalar = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

    // Whether `T` is a scalar or vector that we can use with our widgets.
    template <typename T>
    concept VectorOrScalar = Scalar<typename VectorTraits<T>::BaseType>;

    // Whether `Bound` is a valid min/max bound for `Target`.
    // That is, either the same type, or if `Target` is a vector, `Bound` can also be a scalar of the same type.
    template <typename Bound, typename Target>
    concept ValidBoundForTargetType =
        std::same_as<Bound, Target> ||
        ( VectorTraits<Bound>::size == 1 && std::same_as<typename VectorTraits<Bound>::BaseType, typename VectorTraits<Target>::BaseType> );

    // Whether `Speed` is a valid drag speed type for `Target`.
    // That is, either a single/vector of `float` or the same type as target (or its element if it's a vector).
    template <typename Speed, typename Target>
    concept ValidDragSpeedForTargetType =
        std::same_as<Speed, typename VectorTraits<Target>::BaseType> || std::same_as<Speed, float> ||
        std::same_as<Speed, Target> || std::same_as<Speed, typename VectorTraits<Target>::template ChangeBase<float>>;

    // A common code for sliders and other widgets dealing with measurement units.
    // `E` must be explicitly set to a measurement unit enum. The other template parameters are deduced.
    // `label` is the widget label, `v` is the target value.
    // `func` draws the widget for an individual scalar. We call it more than once for vectors.
    // `func` is `( const char* label, auto& elem, int i ) -> bool`.
    // It receives `elem` already converted to the display units (so you must convert min/max bounds manually). `i` is the element index for vectors.
    // When `v` is integral, `func` will be instantiated for both integral and floating-point element type. The latter is required if we're doing conversions.
    // NOTE: For integral `v`, in `func` you must look at the type of `elem` and convert your min/max bounds (etc) to the same type.
    // Notice `unitParams` being accepted by an lvalue reference. For convenience, we reset the `sourceUnit` in it before calling the user callback,
    //   since at that point no further conversions are necessary.
    template <UnitEnum E, VectorOrScalar T, typename F>
    [[nodiscard]] bool unitWidget( const char* label, T& v, UnitToStringParams<E>& unitParams, F&& func );

    // Some default slider parameters. For now they are hardcoded here, but we can move them elsewhere later.

    // Default drag speed for `UI::drag()`.
    template <UnitEnum E, VectorOrScalar T>
    requires ( VectorTraits<T>::size == 1 )
    [[nodiscard]] float getDefaultDragSpeed();

    // Default step speed for `UI::input()`.
    template <UnitEnum E, VectorOrScalar T, VectorOrScalar TargetType>
    [[nodiscard]] T getDefaultStep( bool fast );
}

// Default flags for `slider()` and `drag()` below.
inline constexpr int defaultSliderFlags = ImGuiSliderFlags_AlwaysClamp;

// Draw a slider.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`, ...
// By default, for angles `v` will be converted to degrees for display (but `vMin`, `vMax` are still in radians, same as `v`),
//   while length and unit-less values will be left as is. This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::VectorOrScalar T, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool slider( const char* label, T& v, const U& vMin, const U& vMax, UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags );

// Draw a dragging widget. Also includes [+] and [-] buttons (for integers only by default_, like `ImGui::Input`).
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`, ...
// By default, for angles `v` will be converted to degrees for display (but `vSpeed` is still in radians, same as `v`),
//   while length and unit-less values will be left as is. This can be customized in `unitParams` or globally (see `MRUnits.h`).
// If only the min limit is specified, then the max limit is assumed to be the largest number.
template <UnitEnum E, detail::VectorOrScalar T, detail::ValidDragSpeedForTargetType<T> SpeedType = float, detail::ValidBoundForTargetType<T> U = typename VectorTraits<T>::BaseType>
bool drag( const char* label, T& v, SpeedType vSpeed = detail::getDefaultDragSpeed<E, SpeedType>(), const U& vMin = std::numeric_limits<U>::max(), const U& vMax = std::numeric_limits<U>::max(), UnitToStringParams<E> unitParams = {}, ImGuiSliderFlags flags = defaultSliderFlags, const U& step = detail::getDefaultStep<E, U, T>( false ), const U& stepFast = detail::getDefaultStep<E, U, T>( true ) );

// Draw a read-only copyable value.
// `E` must be specified explicitly, to one of: `NoUnit` `LengthUnit`, `AngleUnit`, ...
// By default, for angles `v` will be converted to degrees for display, while length and unit-less values will be left as is.
// This can be customized in `unitParams` or globally (see `MRUnits.h`).
template <UnitEnum E, detail::VectorOrScalar T>
void readOnlyValue( const char* label, const T& v, std::optional<ImVec4> textColor = {}, UnitToStringParams<E> unitParams = {} );


/// similar to ImGui::Text but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentText( const char* fmt, ... );
/// similar to ImGui::TextWrapped but use current text color with alpha channel = 0.5
MRVIEWER_API void transparentTextWrapped( const char* fmt, ... );

/// draw tooltip only if current item is hovered
MRVIEWER_API void setTooltipIfHovered( const std::string& text, float scaling );

/// add text with separator line
/// if issueCount is greater than zero, this number will be displayed in red color after the text.
/// If it equals zero - in green color
/// Otherwise it will not be displayed
MRVIEWER_API void separator( float scaling, const std::string& text = "", int issueCount = -1 );
MRVIEWER_API void separator(
    float scaling,
    const std::string& text,
    const ImVec4& color,
    const std::string& issueCount );

/// draws progress bar
/// note that even while scaling is given by argument size should still respect it
/// size: x(y)  < 0 - take all available width(height)
///       x(y) == 0 - use default width(height)
MRVIEWER_API void progressBar( float scaling, float fraction, const Vector2f& size = Vector2f( -1, 0 ) );

// create and append items into a TabBar: see corresponding ImGui:: functions
MRVIEWER_API bool beginTabBar( const char* str_id, ImGuiTabBarFlags flags = 0 );
MRVIEWER_API void endTabBar();
MRVIEWER_API bool beginTabItem( const char* label, bool* p_open = NULL, ImGuiTabItemFlags flags = 0 );
MRVIEWER_API void endTabItem();

} // namespace UI

}

#include "MRUIStyle.ipp"
