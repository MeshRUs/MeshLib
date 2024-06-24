#include "ImGuiHelpers.h"
#include "MRViewer/MRUIRectAllocator.h"
#include "MRViewer/MRUITestEngine.h"
#include "imgui_internal.h"
#include "MRMesh/MRBitSet.h"
#include "MRPch/MRSpdlog.h"
#include "MRRibbonButtonDrawer.h"
#include "MRPalette.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "MRRibbonConstants.h"
#include "MRRibbonMenu.h"
#include "MRImGuiImage.h"
#include "MRRenderLinesObject.h"
#include "MRViewer/MRRibbonFontManager.h"
#include "MRViewer/MRPlaneWidget.h"
#include "MRViewer/MRViewer.h"
#include "MRColorTheme.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRUIStyle.h"

namespace ImGui
{

using namespace MR;

const std::string dragTooltipStr = "Drag with Shift - faster, Alt - slower";

void drawCursorArrow()
{
    auto drawList = ImGui::GetForegroundDrawList();
    auto mousePos = ImGui::GetMousePos();
    mousePos.x += 5.f;

    const auto menuPlugin = MR::getViewerInstance().getMenuPlugin();
    const float scale = menuPlugin ? menuPlugin->menu_scaling() : 1.f;

    const float spaceX = 10 * scale;
    const float sizeX = 12 * scale;
    const float sizeY_2 = 5 * scale;
    // values are calculated so that width of the border line is 1 pixel
    const float shiftLeftX = 2.6f * scale;
    const float shiftRightX = 1.f * scale;
    const float shiftRightY = 1.5f * scale;

    drawList->AddTriangleFilled( ImVec2( mousePos.x - spaceX - sizeX - shiftLeftX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x - spaceX + shiftRightX, mousePos.y - shiftRightY ),
                                 ImVec2( mousePos.x - spaceX + shiftRightX, mousePos.y + sizeY_2 * 2.f + shiftRightY ), 0xFF000000 );
    drawList->AddTriangleFilled( ImVec2( mousePos.x - spaceX - sizeX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x - spaceX, mousePos.y ),
                                 ImVec2( mousePos.x - spaceX, mousePos.y + sizeY_2 * 2.f ), 0xFFFFFFFF );

    drawList->AddTriangleFilled( ImVec2( mousePos.x + spaceX - shiftRightX, mousePos.y - shiftRightY ),
                                 ImVec2( mousePos.x + spaceX + sizeX + shiftLeftX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x + spaceX - shiftRightX, mousePos.y + sizeY_2 * 2.f + shiftRightY ), 0xFF000000 );
    drawList->AddTriangleFilled( ImVec2( mousePos.x + spaceX, mousePos.y ),
                                 ImVec2( mousePos.x + spaceX + sizeX, mousePos.y + sizeY_2 ),
                                 ImVec2( mousePos.x + spaceX, mousePos.y + sizeY_2 * 2.f ), 0xFFFFFFFF );
}

template<typename T>
std::string getRangeStr( T min, T max )
{
    std::string text;
    if ( min > std::numeric_limits<T>::lowest() )
    {
        if ( max < std::numeric_limits<T>::max() )
            text += fmt::format( "valid range [{} - {}]", min, max );
        else
            text += fmt::format( "minimum value {}", min );
    }
    else if ( max < std::numeric_limits<T>::max() )
        text += fmt::format( "maximum value {}", max );
    return text;
}

template<typename T>
void drawTooltip( T min, T max )
{
    static bool inputMode = false;
    if ( IsItemActivated() )
        inputMode = ( GetIO().MouseClicked[0] && GetIO().KeyCtrl ) || GetIO().MouseDoubleClicked[0];

    if ( IsItemActive() )
    {
        if ( !inputMode )
        {
            SetMouseCursor( ImGuiMouseCursor_None );
            drawCursorArrow();
            BeginTooltip();
            Text( "%s", dragTooltipStr.c_str() );
            EndTooltip();
        }

        std::string rangeStr = getRangeStr( min, max );
        if ( !rangeStr.empty() )
        {
            BeginTooltip();
            Text( "%s", rangeStr.c_str() );
            EndTooltip();
        }
    }
}


void PlotCustomHistogram( const char* str_id,
                                 std::function<float( int idx )> values_getter,
                                 std::function<void( int idx )> tooltip,
                                 std::function<void( int idx )> on_click,
                                 int values_count, int values_offset,
                                 float scale_min, float scale_max,
                                 ImVec2 frame_size, int selectedBarId, int hoveredBarId )
{
    if ( frame_size.y < 0.0f )
        return;
    bool encolorSelected = selectedBarId >= 0;

    const ImGuiStyle& style = GetStyle();
    const ImGuiID id = GetID( str_id );

    if ( frame_size.x == 0.0f )
        frame_size.x = CalcItemWidth();
    if ( frame_size.y == 0.0f )
        frame_size.y = frame_size.x / 2.f + ( style.FramePadding.y * 2 );

    ImRect rect;
    rect.Min = GetCursorScreenPos();
    ImVec2 minPlus, maxPlus;
    rect.Max.x = rect.Min.x + frame_size.x; rect.Max.y = rect.Min.y + frame_size.y;
    ImVec2 innerMin = rect.Min; innerMin.x += style.FramePadding.x; innerMin.y += style.FramePadding.y;
    ImVec2 innerMax = rect.Max; innerMax.x -= style.FramePadding.x; innerMax.y -= style.FramePadding.y;
    if ( ( innerMax.y - innerMin.y ) <= 0.0f )
        return;

    // ImGui::Dummy did not handle click properly (it somehow breaks modal openenig) so we changed it to ButtonBehavior
    //Dummy( frame_size );

    ItemSize( rect.GetSize() );
    ItemAdd( rect, id );
    bool hovered, held;
    ButtonBehavior( rect, id, &hovered, &held );

    // Determine scale from values if not specified
    if ( scale_min == FLT_MAX || scale_max == FLT_MAX )
    {
        float v_min = FLT_MAX;
        float v_max = -FLT_MAX;
        for ( int i = 0; i < values_count; i++ )
        {
            const float v = values_getter( i );
            if ( v != v ) // Ignore NaN values
                continue;
            v_min = std::min( v_min, v );
            v_max = std::max( v_max, v );
        }
        if ( scale_min == FLT_MAX )
            scale_min = v_min;
        if ( scale_max == FLT_MAX )
            scale_max = v_max;
    }

    ImDrawList* drawList  = GetWindowDrawList();
    drawList->AddRectFilled( rect.Min, rect.Max, GetColorU32( ImGuiCol_FrameBg ), style.FrameRounding );
    const float border_size = style.FrameBorderSize;
    if ( border_size > 0.0f )
    {
        minPlus.x = rect.Min.x + 1; minPlus.y = rect.Min.y + 1;
        maxPlus.x = rect.Max.x + 1; minPlus.y = rect.Max.y + 1;
        drawList->AddRect( minPlus, maxPlus, GetColorU32( ImGuiCol_BorderShadow ), style.FrameRounding, ImDrawFlags_RoundCornersAll, border_size );
        drawList->AddRect( rect.Min, rect.Max, GetColorU32( ImGuiCol_Border ), style.FrameRounding, ImDrawFlags_RoundCornersAll, border_size );
    }

    constexpr int bar_halfthickness = 1;
    if ( hoveredBarId < 0 )
        hoveredBarId = -bar_halfthickness;
    constexpr int values_count_min = 1;
    if ( values_count >= values_count_min )
    {
        int res_w = std::min( (int) frame_size.x, values_count );
        int item_count = values_count;

        const ImVec2 mousePos = GetIO().MousePos;
        // Tooltip on hover
        if ( hovered && mousePos.x > innerMin.x && mousePos.y > innerMin.y &&mousePos.x < innerMax.x&&mousePos.y < innerMax.y )
        {
            const float t = std::clamp( ( mousePos.x - innerMin.x ) / ( innerMax.x - innerMin.x ), 0.0f, 0.9999f );
            const int v_idx = (int) ( t * item_count );
            IM_ASSERT( v_idx >= 0 && v_idx < values_count );

            tooltip( ( v_idx + values_offset ) % values_count );
            hoveredBarId = v_idx;
            if (GetIO().MouseClicked[0])
            {
                on_click((v_idx + values_offset) % values_count);
            }
        }

        const float t_step = 1.0f / (float) res_w;
        const float inv_scale = ( scale_min == scale_max ) ? 0.0f : ( 1.0f / ( scale_max - scale_min ) );

        float t0 = 0.0f;
        float histogram_zero_line_t = ( scale_min * scale_max < 0.0f ) ? ( -scale_min * inv_scale ) : ( scale_min < 0.0f ? 0.0f : 1.0f );   // Where does the zero line stands

        const ImU32 col_base = GetColorU32( ImGuiCol_PlotHistogram );
        const ImU32 col_hovered = GetColorU32(ImGuiCol_PlotHistogramHovered);
        const ImU32 col_hovered_top = GetColorU32(ImGuiCol_TabHovered);
        ImVec4 col{ 1.0f, 0.2f, 0.2f, 1.0f };
        const ImU32 col_selected = GetColorU32(col);
        const ImU32 col_selected_top = GetColorU32(ImGuiCol_TabActive);

        for ( int n = 0; n < res_w; n++ )
        {
            const float t1 = t0 + t_step;
            const int v1_idx = (int) ( t0 * item_count + 0.5f );
            IM_ASSERT( v1_idx >= 0 && v1_idx < values_count );

            float val = values_getter( v1_idx + values_offset );
            float top = 1.0f - std::clamp( ( val - scale_min ) * inv_scale, 0.0f, 1.0f );

            // NB: Draw calls are merged together by the DrawList system. Still, we should render our batch are lower level to save a bit of CPU.
            ImVec2 pos0 = ImVec2( innerMin.x + ( innerMax.x - innerMin.x ) * t0, innerMin.y + ( innerMax.y - innerMin.y ) * top );
            ImVec2 pos1 = ImVec2( innerMin.x + ( innerMax.x - innerMin.x ) * t1, innerMin.y + ( innerMax.y - innerMin.y ) * histogram_zero_line_t );
            {
                if ( pos1.x >= pos0.x + 2.0f )
                    pos1.x -= 1.0f;
                if ( abs(v1_idx - hoveredBarId) < bar_halfthickness )
                    drawList->AddRectFilled( ImVec2( pos0.x, innerMin.y ), ImVec2( pos1.x, pos0.y ), col_hovered_top );
                if ( encolorSelected && abs(v1_idx - selectedBarId) < bar_halfthickness )
                    drawList->AddRectFilled( ImVec2( pos0.x, innerMin.y ), ImVec2( pos1.x, pos0.y ), col_selected_top );

                auto getBarColor = [&](const int v1_idx)
                {
                 if ( abs(v1_idx - hoveredBarId) < bar_halfthickness )
                        return col_hovered;
                 if ( encolorSelected && abs(v1_idx - selectedBarId) < bar_halfthickness )
                        return col_selected;
                    return col_base;
                };
                drawList->AddRectFilled( pos0, pos1, getBarColor(v1_idx) );
            }

            t0 = t1;
        }
    }
}

bool DragFloatValid( const char* label, float* v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags )
{
    bool res = DragFloat( label, v, v_speed, v_min, v_max, format, flags );
    *v = std::clamp( *v, v_min, v_max );
    drawTooltip( v_min, v_max );
    return res;
}

bool DragFloatValidLineWidth( const char* label, float* value )
{
    const auto& range = MR::GetAvailableLineWidthRange();
    bool cannotChange = range.x == range.y;
    if ( cannotChange )
        ImGui::PushStyleColor( ImGuiCol_Text, MR::Color::gray().getUInt32() );
    bool res = DragFloatValid( label, value, 1.0f, range.x, range.y, "%.1f",
        cannotChange ? ImGuiSliderFlags_NoInput : ImGuiSliderFlags_None );
    if ( cannotChange )
    {
        ImGui::PopStyleColor();
        if ( IsItemHovered() && !IsItemActive() )
            SetTooltip( "Line width cannot be changed with current renderer." );
    }
    return res;
}

bool DragIntValid( const char *label, int* v, float speed,
                   int v_min, int v_max, const char* format )
{
    auto res = DragInt( label, v, speed, v_min, v_max, format );
    *v = std::clamp( *v, v_min, v_max );
    drawTooltip( v_min, v_max );
    return res;
}

bool InputIntValid( const char* label, int* v, int v_min, int v_max,
    int step, int step_fast, ImGuiInputTextFlags flags )
{
    auto res = InputInt( label, v, step, step_fast, flags );
    *v = std::clamp( *v, v_min, v_max );
    if ( IsItemActive() )
    {
        std::string text = getRangeStr( v_min, v_max );
        if ( !text.empty() )
        {
            BeginTooltip();
            Text( "%s", text.c_str() );
            EndTooltip();
        }
    }
    return res;
}

MultiDragRes DragFloatValid2( const char* label, float* valueArr, float step, float valueMin, float valueMax, const char* format, ImGuiSliderFlags flags, const char* ( *tooltips )[2] )
{
    MultiDragRes res;

    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if ( window->SkipItems )
        return res;

    BeginGroup();
    PushID( label );
    constexpr int components = 2;
    PushMultiItemsWidths( components, CalcItemWidth() );
    for ( int i = 0; i < components; i++ )
    {
        PushID( i );
        if ( i > 0 )
            SameLine( 0, g.Style.ItemInnerSpacing.x );
        res.valueChanged = DragFloatValid( "", valueArr + i, step, valueMin, valueMax, format, flags ) || res.valueChanged;
        if ( tooltips && IsItemHovered() && !IsItemActive() )
            SetTooltip( "%s", ( *tooltips )[i] );
        res.itemDeactivatedAfterEdit = res.itemDeactivatedAfterEdit || IsItemDeactivatedAfterEdit();
        PopID();
        PopItemWidth();
    }
    PopID();

    const char* label_end = FindRenderedTextEnd( label );
    if ( label != label_end )
    {
        SameLine( 0, g.Style.ItemInnerSpacing.x );
        TextEx( label, label_end );
    }

    EndGroup();
    return res;
}

MultiDragRes DragFloatValid3( const char * label, float* valueArr, float step, float valueMin, float valueMax, const char* format, ImGuiSliderFlags flags, const char* (*tooltips)[3] )
{
    MultiDragRes res;

    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if (window->SkipItems)
        return res;

    BeginGroup();
    PushID(label);
    constexpr int components = 3;
    PushMultiItemsWidths(components, CalcItemWidth());
    for (int i = 0; i < components; i++)
    {
        PushID(i);
        if (i > 0)
            SameLine(0, g.Style.ItemInnerSpacing.x);
        res.valueChanged = DragFloatValid("", valueArr + i, step, valueMin, valueMax, format, flags) || res.valueChanged;
        if ( tooltips && IsItemHovered() && !IsItemActive() )
            SetTooltip( "%s", (*tooltips)[i] );
        res.itemDeactivatedAfterEdit = res.itemDeactivatedAfterEdit || IsItemDeactivatedAfterEdit();
        PopID();
        PopItemWidth();
    }
    PopID();

    const char* label_end = FindRenderedTextEnd(label);
    if (label != label_end)
    {
        SameLine(0, g.Style.ItemInnerSpacing.x);
        TextEx(label, label_end);
    }

    EndGroup();
    return res;
}

MultiDragRes DragIntValid3( const char* label, int v[3], float speed, int min, int max, const char* format, const char* ( *tooltips )[3])
{
    MultiDragRes res;

    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if ( window->SkipItems )
        return res;

    BeginGroup();
    PushID( label );
    constexpr int components = 3;
    PushMultiItemsWidths( components, CalcItemWidth() );
    for ( int i = 0; i < components; i++ )
    {
        PushID( i );
        if ( i > 0 )
            SameLine( 0, g.Style.ItemInnerSpacing.x );
        res.valueChanged = DragIntValid( "", v + i, speed, min, max, format ) || res.valueChanged;
        if ( tooltips && IsItemHovered() && !IsItemActive() )
            SetTooltip( "%s", ( *tooltips )[i] );
        res.itemDeactivatedAfterEdit = res.itemDeactivatedAfterEdit || IsItemDeactivatedAfterEdit();
        PopID();
        PopItemWidth();
    }
    PopID();

    const char* label_end = FindRenderedTextEnd( label );
    if ( label != label_end )
    {
        SameLine( 0, g.Style.ItemInnerSpacing.x );
        TextEx( label, label_end );
    }

    EndGroup();
    return res;
}

bool BeginStatePlugin( const char* label, bool* open, float width )
{
    ImGuiWindow* window = FindWindowByName( label );
    if ( !window )
    {
        float yPos = 0.0f;
        auto menu = MR::getViewerInstance().getMenuPluginAs<MR::RibbonMenu>();
        if ( menu )
            yPos = menu->getTopPanelOpenedHeight() * menu->menu_scaling();
        SetNextWindowPos( ImVec2( GetIO().DisplaySize.x - width, yPos ), ImGuiCond_FirstUseEver );
        SetNextWindowSize( ImVec2( width, 0 ), ImGuiCond_FirstUseEver );
    }
    SetNextWindowSizeConstraints( ImVec2( width, -1.0f ), ImVec2( width, -1.0f ) );
    auto flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    return Begin( label, open, flags );
}

ImVec2 GetDownPosition( const float width )
{
    return { GetIO().DisplaySize.x - width, GetIO().DisplaySize.y };
}

float GetTitleBarHeght( float menuScaling )
{
    return 2 * MR::cRibbonItemInterval * menuScaling + ImGui::GetTextLineHeight() + 2 * ImGui::GetStyle().WindowBorderSize * menuScaling;
}

bool BeginCustomStatePlugin( const char* label, bool* open, const CustomStatePluginWindowParameters& params )
{
    const auto& style = ImGui::GetStyle();

    const float borderSize = style.WindowBorderSize * params.menuScaling;
    const float titleBarHeight = GetTitleBarHeght( params.menuScaling );
    auto height = params.height;
    if ( params.collapsed && *params.collapsed )
        height = titleBarHeight;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 12 * params.menuScaling, 8 * params.menuScaling ) );

    ImGuiWindow* window = FindWindowByName( label );
    auto menu = MR::getViewerInstance().getMenuPlugin();
    ImVec2 initialWindowPos;
    bool haveSavedWindowPos = false;
    bool windowIsInactive = window && !window->WasActive;
    if ( !window || windowIsInactive )
    {
        auto ribMenu = std::dynamic_pointer_cast<MR::RibbonMenu>( menu );
        float xPos = GetIO().DisplaySize.x - params.width;
        float yPos = 0.0f;
        if ( params.position )
        {
            xPos = params.position->x;
            yPos = params.position->y;
        }
        else if ( ribMenu )
            yPos = ( ribMenu->getTopPanelOpenedHeight() - 1.0f ) * menu->menu_scaling();

        const std::string configKey = std::string( label ) + std::string( "_position" );
        auto& config = MR::Config::instance();

        if ( menu->isSavedDialogPositionsEnabled() && config.hasJsonValue( "DialogPositions" ) )
        {
            auto json = config.getJsonValue( "DialogPositions" )[label];
            if ( json.empty() )
            {
                initialWindowPos = ImVec2( xPos, yPos );
            }
            else
            {
                haveSavedWindowPos = true;
                initialWindowPos = ImVec2( json["x"].asFloat(), json["y"].asFloat() );
            }
        }
        else
        {
            initialWindowPos = ImVec2( xPos, yPos );
        }
    }

    UI::getDefaultWindowRectAllocator().setFreeNextWindowPos( label, initialWindowPos, haveSavedWindowPos ? ImGuiCond_FirstUseEver : ImGuiCond_Appearing, params.pivot );

    if ( params.changedSize )
    {
        if ( params.collapsed && *params.collapsed )
            SetNextWindowSize( { params.changedSize->x, height }, ImGuiCond_Always );
        else
            SetNextWindowSize( *params.changedSize, ImGuiCond_Always );
    }
    else
    {
        SetNextWindowSize( ImVec2( params.width, height ), ImGuiCond_Appearing );
        auto pHeight = params.height;
        if ( pHeight <= 0.0f )
            pHeight = -1.0f;
        pHeight = std::min( pHeight, GetMainViewport()->Size.y - style.DisplaySafeAreaPadding.y * 2.0f );
        SetNextWindowSizeConstraints( ImVec2( params.width, pHeight ), ImVec2( params.width, pHeight ) );
    }

    auto context = ImGui::GetCurrentContext();
    auto flags = params.flags;
    if ( params.collapsed && *params.collapsed )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_WindowMinSize, { 0, 0 } );
        ImGui::SetNextWindowSizeConstraints( { context->NextWindowData.SizeVal.x, titleBarHeight }, { context->NextWindowData.SizeVal.x, titleBarHeight } );
        flags |= ImGuiWindowFlags_NoResize;
    }

    // needed for manual scrollbar
    bool hasPrevData = false;
    float prevCursorMaxPos = FLT_MAX;
    if ( window )
    {
        hasPrevData = true;
        prevCursorMaxPos = window->DC.CursorMaxPos.y;
    }

    if ( !Begin( label, open, flags | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse ) )
    {
        *open = false;
        ImGui::PopStyleVar( 2 );
        if ( params.collapsed && *params.collapsed )
            ImGui::PopStyleVar();
        return false;
    }

    window = context->CurrentWindow;
    // Manually draw Y scroll bar if window cannot be big enough
    if ( std::ceil( window->SizeFull.y ) < std::floor( window->ContentSizeIdeal.y + 2 * style.WindowPadding.y ) && !params.changedSize )
    {
        // Set scrollbar size
        window->ScrollbarSizes[ImGuiAxis_Y ^ 1] = style.ScrollbarSize;
        // Prevent "tremor" on holding scrollbar near bottom
        // (if skip this code window->ContentSize change beacuse of outside scrollbar)
        auto backUpContSizeY = window->ContentSize.y;
        if ( hasPrevData )
        {
            window->ContentSize.y = ( ( window->ContentSize.y - window->ContentSizeIdeal.y ) +
                prevCursorMaxPos - window->DC.CursorStartPos.y ) - ( titleBarHeight );
        }
        // Determine scrollbar position
        window->InnerRect.Min.y += ( titleBarHeight - borderSize );
        window->InnerRect.Max.y -= borderSize;
        window->InnerRect.Max.x -= ( window->ScrollbarSizes.x + borderSize );
        window->Size.x -= borderSize;
        // Needed for ImGui::GetAvailableContent functuions
        window->WorkRect.Min.y += ( titleBarHeight - borderSize );
        window->WorkRect.Max.x -= window->ScrollbarSizes.x;
        window->ContentRegionRect.Min.y += ( titleBarHeight + borderSize );
        window->ContentRegionRect.Max.x -= window->ScrollbarSizes.x;
        // Enable scroll by mouse if manual scrollbar
        window->Flags &= ~ImGuiWindowFlags_NoScrollWithMouse;
        // Draw scrollbar
        window->DrawList->PushClipRect( window->Rect().Min, window->Rect().Max );
        Scrollbar( ImGuiAxis_Y );
        window->DrawList->PopClipRect();
        // Reset old values
        window->ContentSize.y = backUpContSizeY;
    }

    if ( params.changedSize && params.collapsed && !*params.collapsed )
    {
        params.changedSize->x = window->Rect().GetWidth();
        params.changedSize->y = window->Rect().GetHeight();
    }

    if ( params.collapsed && *params.collapsed )
        ImGui::PopStyleVar();

    const auto bgColor = ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4( ImGuiCol_FrameBg ));

    ImGui::PushStyleColor( ImGuiCol_Button, bgColor );
    ImGui::PushStyleColor( ImGuiCol_Border, bgColor );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 0.0f,  0.0f } );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 2 * params.menuScaling );

    const float buttonSize = titleBarHeight - 2 * MR::cRibbonItemInterval * params.menuScaling - 2 * borderSize;
    const auto buttonOffset = ( titleBarHeight - buttonSize ) * 0.5f;
    ImGui::SetCursorScreenPos( { window->Rect().Min.x + buttonOffset, window->Rect().Min.y + buttonOffset } );

    ImFont* iconsFont = MR::RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::Icons );
    ImFont* titleFont = MR::RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::SemiBold );

    if ( iconsFont )
    {
        iconsFont->Scale = MR::cDefaultFontSize / MR::cBigIconSize;
        ImGui::PushFont( iconsFont );
    }

    const ImRect boundingBox( { window->Rect().Min.x + borderSize, window->Rect().Min.y + borderSize }, { window->Rect().Max.x - borderSize, window->Rect().Min.y + titleBarHeight - borderSize } );

    window->DrawList->PushClipRect( window->Rect().Min, window->Rect().Max );
    window->DrawList->AddRectFilled( boundingBox.Min, boundingBox.Max, bgColor );

    if ( params.collapsed )
    {
        if ( ImGui::Button( *params.collapsed ? "\xef\x84\x85" : "\xef\x84\x87", { buttonSize, buttonSize } ) )// minimize/maximize button
        {
            *params.collapsed = !*params.collapsed;
            ImGui::PopStyleVar( 4 );
            ImGui::PopStyleColor( 2 );

            if (iconsFont )
                ImGui::PopFont();

            window->DrawList->PopClipRect();
            ImGui::End();
            return false;
        }
        ImGui::SameLine();
    }

    if ( iconsFont )
        ImGui::PopFont();

    auto cursorScreenPos = ImGui::GetCursorScreenPos();
    if ( titleFont )
    {
        ImGui::PushFont( titleFont );
        // "+ 5 * params.menuScaling" eliminates shift of the font
        ImGui::SetCursorScreenPos( { cursorScreenPos.x, window->Rect().Min.y + 5 * params.menuScaling } );
    }
    else
        ImGui::SetCursorScreenPos( { cursorScreenPos.x, window->Rect().Min.y + 0.5f * ( titleBarHeight - ImGui::GetFontSize() ) } );

    ImGui::RenderText( ImGui::GetCursorScreenPos(), label );

    if ( titleFont )
        ImGui::PopFont();

    ImGui::SameLine();

    if ( params.helpBtnFn )
    {
        auto font = ImGui::GetFont();
        font->Scale = 0.9f;
        ImGui::PushFont( font );

        const float btnHelpWidth = ImGui::CalcTextSize( "HELP" ).x + 6.0f * params.menuScaling;
        const float btnHelpOffset = buttonSize / 2.f;
        ImGui::SetCursorScreenPos( { window->Rect().Max.x - ( buttonSize + buttonOffset ) - ( btnHelpWidth + btnHelpOffset ), window->Rect().Min.y + buttonOffset } );
        ImGui::PushStyleColor( ImGuiCol_Button, MR::Color( 60, 169, 20 ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, MR::Color( 66, 186, 22 ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, MR::Color( 73, 205, 24 ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Text, MR::Color::white().getUInt32() );
        if ( ImGui::Button( "HELP", { btnHelpWidth, buttonSize } ) )
            params.helpBtnFn();
        ImGui::PopStyleColor( 4 );

        ImGui::PopFont();
        font->Scale = 1.f;

        ImGui::SameLine();
    }

    if ( iconsFont )
        ImGui::PushFont( iconsFont );

    ImGui::SetCursorScreenPos( { window->Rect().Max.x - ( buttonSize + buttonOffset ), window->Rect().Min.y + buttonOffset } );
    bool escapeClose = params.closeWithEscape && ImGui::IsKeyPressed( ImGuiKey_Escape ) && !ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
    if ( escapeClose && menu )
        escapeClose = window == menu->getLastFocusedPlugin();
    if ( ImGui::Button( "\xef\x80\x8d", { buttonSize, buttonSize } ) || escapeClose ) //close button
    {
        *open = false;

        if ( iconsFont )
            ImGui::PopFont();

        ImGui::PopStyleColor( 2 );
        ImGui::PopStyleVar( 4 );
        window->DrawList->PopClipRect();
        ImGui::End();
        return false;
    }

    if ( iconsFont )
    {
        ImGui::PopFont();
        iconsFont->Scale = 1.0f;
    }

    ImGui::PopStyleVar( 3 );

    if ( params.collapsed && *params.collapsed )
    {
        ImGui::PopStyleVar();
        ImGui::PopStyleColor( 2 );
        const auto borderColor = ImGui::ColorConvertFloat4ToU32( ImGui::GetStyleColorVec4( ImGuiCol_Border ) );

        //ImGui doesn't draw bottom border if window is collapsed, so add it manually
        window->DrawList->AddLine( { window->Rect().Min.x, window->Rect().Max.y - borderSize }, { window->Rect().Max.x, window->Rect().Max.y - borderSize }, borderColor, borderSize );
        window->DrawList->PopClipRect();
        ImGui::End();
        return false;
    }

    ImGui::PopStyleColor( 2 );
    window->DrawList->PopClipRect();

    const ImGuiTableFlags tableFlags = ImGuiTableFlags_SizingStretchProp;

    ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, { 0,0 } );
    ImGui::SetCursorPosY( titleBarHeight + style.WindowPadding.y - borderSize );
    if ( !ImGui::BeginTable( "ContentTable", 1, tableFlags, { -1, -1 } ) )
    {
        ImGui::PopStyleVar( 2 );
        ImGui::End();
        return false;
    }
    ImGui::PopStyleVar();

    ImGui::TableNextColumn();
    window->ClipRect = window->InnerRect;
    window->DrawList->PushClipRect( window->InnerRect.Min, window->InnerRect.Max );

    { // Create a group for the UI testing engine.
        // Strip `##...` from the label.
        std::string_view strippedLabel = label;
        auto sep = strippedLabel.find( "##" );
        if ( sep != std::string_view::npos )
            strippedLabel = strippedLabel.substr( 0, sep );

        UI::TestEngine::pushTree( strippedLabel );
    }

    return true;
}

void EndCustomStatePlugin()
{
    EndTable();
    auto context = ImGui::GetCurrentContext();
    auto window = context->CurrentWindow;
    window->DrawList->PopClipRect();
    ImGui::PopStyleVar();
    End();

    UI::TestEngine::popTree();
}

bool BeginModalNoAnimation( const char* label, bool* open /*= nullptr*/, ImGuiWindowFlags flags /*= 0 */ )
{
    const auto color = MR::ColorTheme::getRibbonColor( MR::ColorTheme::RibbonColorsType::FrameBackground ).getUInt32();
    ImGui::PushStyleColor( ImGuiCol_TitleBgActive, color );
    ImGui::PushStyleColor( ImGuiCol_Text, 0 );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );

    auto context = ImGui::GetCurrentContext();
    ImGuiWindow* window = FindWindowByName( label );
    // needed for manual scrollbar
    bool hasPrevData = false;
    float prevCursorMaxPos = FLT_MAX;
    if ( window )
    {
        hasPrevData = true;
        prevCursorMaxPos = window->DC.CursorMaxPos.y;
    }
    if ( !BeginPopupModal( label, open, flags | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse ) )
    {
        ImGui::PopStyleVar();
        ImGui::PopStyleColor( 2 );
        return false;
    }

    window = context->CurrentWindow;
    const auto& style = GetStyle();
    // Manually draw Y scroll bar if window cannot be big enough
    if ( std::ceil( window->SizeFull.y ) < std::floor( window->ContentSizeIdeal.y + 2 * style.WindowPadding.y ) )
    {
        // Set scrollbar size
        window->ScrollbarSizes[ImGuiAxis_Y ^ 1] = style.ScrollbarSize;
        // Prevent "tremor" on holding scrollbar near bottom
        // (if skip this code window->ContentSize change beacuse of outside scrollbar)
        auto backUpContSizeY = window->ContentSize.y;
        if ( hasPrevData )
        {
            window->ContentSize.y = ( ( window->ContentSize.y - window->ContentSizeIdeal.y ) +
                prevCursorMaxPos - window->DC.CursorStartPos.y );
        }
        // Determine scrollbar position
        window->InnerRect.Max.x -= window->ScrollbarSizes.x;
        // Needed for ImGui::GetAvailableContent functuions
        window->WorkRect.Max.x -= window->ScrollbarSizes.x;
        window->ContentRegionRect.Max.x -= window->ScrollbarSizes.x;
        // Enable scroll by mouse if manual scrollbar
        window->Flags &= ~ImGuiWindowFlags_NoScrollWithMouse;
        // Draw scrollbar
        window->DrawList->PushClipRect( window->Rect().Min, window->Rect().Max );
        Scrollbar( ImGuiAxis_Y );
        window->DrawList->PopClipRect();
        // Reset old values
        window->ContentSize.y = backUpContSizeY;
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor( 2 );
    GetCurrentContext()->DimBgRatio = 1.0f;

    if ( !window || ( flags & ImGuiWindowFlags_NoTitleBar ) )
        return true;

    auto font = MR::RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::SemiBold );
    if ( font )
        ImGui::PushFont( font );

    const auto backupPos = ImGui::GetCursorPos();

    float menuScaling = 1.0f;
    if ( auto menu = MR::getViewerInstance().getMenuPlugin() )
        menuScaling = menu->menu_scaling();

    ImGui::PushClipRect( { window->Pos.x, window->Pos.y }, { window->Pos.x + window->Size.x, window->Pos.y + window->Size.y }, false );
    // " 4.0f * params.menuScaling" eliminates shift of the font
    ImGui::SetCursorPos( { ImGui::GetStyle().WindowPadding.x, 4.0f * menuScaling } );
    ImGui::TextUnformatted( label, strstr( label, "##" ) );

    ImGui::SetCursorPos( backupPos );
    ImGui::PopClipRect();

    if ( font )
        ImGui::PopFont();

    return true;
}

bool InputIntBitSet( const char* label, int* v, const MR::BitSet& bs, int step /*= 1*/, int step_fast /*= 100*/, ImGuiInputTextFlags flags /*= 0 */ )
{
    int& value = *v;
    const int oldValue = value;

    const int firstValid = int( bs.find_first() );
    const bool validBitSet = firstValid >= 0;
    ImGuiInputTextFlags inputFlag = ImGuiInputTextFlags_None;
    if ( !validBitSet )
    {
        const auto color = GetStyle().Colors[ImGuiCol_TextDisabled];
        PushStyleColor( ImGuiCol_Button, color );
        PushStyleColor( ImGuiCol_ButtonActive, color );
        PushStyleColor( ImGuiCol_ButtonHovered, color );
        inputFlag |= ImGuiInputTextFlags_ReadOnly;
    }

    const bool changed = ImGui::InputInt( label, v, step, step_fast, flags | inputFlag );

    if ( !validBitSet )
    {
        PopStyleColor( 3 );
        value = oldValue;
        return false;
    }

    if ( !bs.test( value ) && ImGui::IsItemDeactivatedAfterEdit() )
    {
        const int lastValid = int( bs.find_last() );
        if ( value < firstValid )
            value = firstValid;
        else if ( value > lastValid )
            value = lastValid;
        else
        {
            if ( changed )
            {
                if ( value > oldValue )
                    value = int( bs.find_next( value ) );
                else
                    while ( !bs.test( --value ) );
            }
            else
                value = int( bs.find_next( value ) );
        }
    }

    return value != oldValue && bs.test( value );
}

bool DragInputInt( const char* label, int* value, float speed /*= 1*/, int min /*= std::numeric_limits<int>::lowest()*/,
                   int max /*= std::numeric_limits<int>::max()*/, const char* format /*= "%d" */, ImGuiSliderFlags flags /*= ImGuiSliderFlags_None*/ )
{
    ImGuiContext& g = *ImGui::GetCurrentContext();
    ImGuiWindow* window = g.CurrentWindow;
    if ( window->SkipItems )
        return false;

    BeginGroup();
    PushID( label );

    int& valueRef = *value;
    const int valueOld = valueRef;

    const std::string labelStr = std::string( "##" ) + label;

    const auto& style = GetStyle();
    const float sizeSide = style.FramePadding.y * 2 + ImGui::GetTextLineHeight();
    SetNextItemWidth( ImMax( 1.0f, CalcItemWidth() - ( sizeSide + style.ItemInnerSpacing.x ) * 2 ) );
    DragInt( labelStr.c_str(), value, speed, min, max, format, flags );
    drawTooltip( min, max );
    ImGui::SameLine( 0, style.ItemInnerSpacing.x );
    ImGui::PushButtonRepeat( true );

    if ( MR::UI::button( "-", MR::Vector2f( sizeSide, sizeSide ) ) )
        --valueRef;
    ImGui::SameLine( 0, style.ItemInnerSpacing.x );
    if ( MR::UI::button( "+", MR::Vector2f( sizeSide, sizeSide ) ) )
        ++valueRef;
    ImGui::PopButtonRepeat();
    valueRef = std::clamp( valueRef, min, max );

    PopID();

    const char* label_end = FindRenderedTextEnd( label );
    if ( label != label_end )
    {
        SameLine( 0, g.Style.ItemInnerSpacing.x );
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() - g.CurrentWindow->DC.CurrLineTextBaseOffset + style.FramePadding.y );
        TextEx( label, label_end );
    }

    EndGroup();

    return valueRef != valueOld;
}

bool Link( const char* label, uint32_t color )
{
    auto window = GetCurrentContext()->CurrentWindow;
    assert( window );
    if ( !window )
        return false;

    auto linkSize = CalcTextSize( label );

    auto basePos = ImVec2( window->DC.CursorPos.x,
                           window->DC.CursorPos.y + window->DC.CurrLineTextBaseOffset );
    ImVec2 linkBbMaxPoint( basePos.x + linkSize.x, basePos.y + linkSize.y );
    ImRect linkRect( basePos, linkBbMaxPoint );

    auto linkId = window->GetID( label );
    ItemAdd( linkRect, linkId );
    bool hovered, held;
    bool pressed = ButtonBehavior( linkRect, linkId, &hovered, &held );

    if ( hovered )
    {
        SetMouseCursor( ImGuiMouseCursor_Hand );
        window->DrawList->AddLine( ImVec2( basePos.x, linkBbMaxPoint.y - 1.0f ),
                                   ImVec2( linkBbMaxPoint.x, linkBbMaxPoint.y - 1.0f ),
                                   color );
    }

    PushStyleColor( ImGuiCol_Text, color );
    ImGui::Text( "%s", label );
    PopStyleColor();

    return pressed;
}

PaletteChanges Palette(
    const char* label,
    MR::Palette& palette,
    std::string& presetName,
    float width,
    float menuScaling,
    bool* fixZero,
    float speed,
    float min,
    float max )
{
    using namespace MR;
    PaletteChanges changes = PaletteChanges::None;
    float scaledWidth = width * menuScaling;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { cDefaultInnerSpacing * menuScaling, cDefaultInnerSpacing * menuScaling } );
    const auto& presets = PalettePresets::getPresetNames();
    if ( !presets.empty() )
    {
        int currentIndex = -1;
        if ( !presetName.empty() )
        {
            for ( int i = 0; i < presets.size(); i++ )
            {
                if ( presets[i] == presetName )
                {
                    currentIndex = i;
                    break;
                }
            }
        }

        ImGui::SetNextItemWidth( scaledWidth );
        int presetIndex = currentIndex;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { ImGui::GetStyle().FramePadding.x, cInputPadding * menuScaling } );
        if ( UI::combo( "Load preset", &presetIndex, presets, true, {}, "Select Palette Preset" ) )
        {
            if ( presetIndex != currentIndex )
            {
                if ( PalettePresets::loadPreset( presets[presetIndex], palette ) )
                    presetName = presets[presetIndex];
                else
                    showError( "Cannot load preset with name: \"" + presets[presetIndex] + "\"" );
            }

            if ( fixZero )
                *fixZero = false;
            changes = PaletteChanges::All;
            CloseCurrentPopup();
        }
        ImGui::PopStyleVar();
        UI::setTooltipIfHovered( "Load one of custom presets", menuScaling );
    }

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cSeparateBlocksSpacing * menuScaling, cSeparateBlocksSpacing * menuScaling } );
    bool fixZeroChanged = false;
    if ( fixZero )
    {
        fixZeroChanged = UI::checkbox( "Set Zero to Green", fixZero );
        UI::setTooltipIfHovered( "If checked, zero value always will be green", menuScaling );
    }
    bool isDiscrete = palette.getTexture().filter == FilterType::Discrete;

    const auto& params = palette.getParameters();

    if ( UI::checkbox( "Discrete Palette", &isDiscrete ) )
    {
        palette.setFilterType( isDiscrete ? FilterType::Discrete : FilterType::Linear );
        changes |= PaletteChanges::Texture | PaletteChanges::Ranges; // both the texture and uv-coordinates must be recomputed
        presetName.clear();
    }
    UI::setTooltipIfHovered( "If checked, palette will have several discrete levels. Otherwise it will be smooth.", menuScaling );
    if ( isDiscrete )
    {
        ImGui::SameLine();
        int discretization = params.discretization;
        ImGui::SetNextItemWidth( ImGui::GetContentRegionAvail().x );
        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + cCheckboxPadding - ImGui::GetStyle().FramePadding.y );
        if ( UI::drag<NoUnit>( "###Discretization", discretization, 1.f, 2, 100 ) )
        {
            palette.setDiscretizationNumber( discretization );
            palette.resetLabels();
            changes |= PaletteChanges::Texture;
            presetName.clear();
        }
        UI::setTooltipIfHovered( "Number of discrete levels", menuScaling );
    }

    ImGui::PopStyleVar();

    assert( params.ranges.size() == 2 || params.ranges.size() == 4 );
    int paletteRangeMode = params.ranges.size() == 2 ? 0 : 1;
    int paletteRangeModeBackUp = paletteRangeMode;
    ImGui::PushItemWidth( scaledWidth );

    const auto oldPaletteRangeMode = paletteRangeMode;
    UI::combo( "Palette Type", &paletteRangeMode, { "Even Space", "Central Zone" } );
    UI::setTooltipIfHovered( "If \"Central zone\" selected you can separately fit values which are higher or lower then central one. Otherwise only the whole scale can be fit", menuScaling );
    if ( oldPaletteRangeMode != paletteRangeMode )
        changes |= PaletteChanges::Texture;
    ImGui::PopItemWidth();

    ImGui::PushItemWidth( 0.5f * scaledWidth );
    float ranges[4];
    ranges[0] = params.ranges.front();
    ranges[3] = params.ranges.back();
    if ( paletteRangeMode == 1 )
    {
        if ( paletteRangeModeBackUp == 0 )
            ranges[1] = ranges[2] = ( ranges[0] + ranges[3] ) * 0.5f;
        else
        {
            ranges[1] = params.ranges[1];
            ranges[2] = params.ranges[2];
        }
    }

    bool rangesChanged = false;
    if ( paletteRangeMode == 0 )
    {
        if ( fixZero && ( *fixZero ) )
        {
            if ( ranges[3] < 0.0f )
                ranges[3] = 0.0f;
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { ImGui::GetStyle().ItemSpacing.x, cSeparateBlocksSpacing * menuScaling } );
            rangesChanged |= UI::drag<NoUnit>( "Min/Max", ranges[3], speed, 0.0f, max );
            ImGui::PopStyleVar();
            ranges[0] = -ranges[3];
        }
        else
        {
            rangesChanged |= UI::drag<LengthUnit>( "Max (red)", ranges[3], speed, min, max );
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { ImGui::GetStyle().ItemSpacing.x, cSeparateBlocksSpacing * menuScaling } );
            rangesChanged |= UI::drag<LengthUnit>( "Min (blue)", ranges[0], speed, min, max );
            ImGui::PopStyleVar();
        }
    }
    else if ( paletteRangeMode == 1 )
    {
        if ( fixZero && ( *fixZero ) )
        {
            if ( ranges[3] < 0.0f )
                ranges[3] = 0.0f;

            rangesChanged |= UI::drag<NoUnit>( "Max positive / Min negative", ranges[3], speed, min, max );
            ranges[0] = -ranges[3];

            if ( ranges[2] < 0.0f )
                ranges[2] = 0.0f;

            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { ImGui::GetStyle().ItemSpacing.x, cSeparateBlocksSpacing * menuScaling } );
            rangesChanged |= UI::drag<NoUnit>( "Min positive / Max negative", ranges[2], speed, min, max );
            ImGui::PopStyleVar();
            ranges[1] = -ranges[2];
        }
        else
        {
            rangesChanged |= UI::drag<NoUnit>( "Max positive (red)", ranges[3], speed, min, max );
            rangesChanged |= UI::drag<NoUnit>( "Min positive (green)", ranges[2], speed, min, max );
            rangesChanged |= UI::drag<NoUnit>( "Max negative (green)", ranges[1], speed, min, max );
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { ImGui::GetStyle().ItemSpacing.x, cSeparateBlocksSpacing * menuScaling } );
            rangesChanged |= UI::drag<NoUnit>( "Min negative (blue)", ranges[0], speed, min, max );
            ImGui::PopStyleVar();
        }
    }
    ImGui::PopItemWidth();

    bool correctOrder = true;
    int orderStep = ( 1 - paletteRangeMode ) * 2 + 1;
    for ( int i = 0; i < 4; )
    {
        int next = i + orderStep;
        if ( next >= 4 )
            break;
        if ( ranges[i] > ranges[next] )
        {
            correctOrder = false;
            break;
        }
        i = next;
    }

    if ( !correctOrder )
    {
        ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
        ImGui::TextWrapped( "Invalid values order" );
        ImGui::PopStyleColor();
    }
    if ( correctOrder && ( fixZeroChanged || ( paletteRangeMode != paletteRangeModeBackUp ) || rangesChanged ) )
    {
        changes |= PaletteChanges::Ranges;
        presetName.clear();
        if ( paletteRangeMode == 0 )
            palette.setRangeMinMax( ranges[0], ranges[3] );
        else
            palette.setRangeMinMaxNegPos( ranges[0], ranges[1], ranges[2], ranges[3] );
    }
    ImGui::PopStyleVar();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { ImGui::GetStyle().ItemSpacing.x, cSeparateBlocksSpacing * menuScaling } );

    std::string popupName = std::string( "Save Palette##Config" ) + std::string( label );
    if ( UI::button( "Save Palette as", Vector2f( scaledWidth, 0 ) ) )
        ImGui::OpenPopup( popupName.c_str() );
    UI::setTooltipIfHovered( "Save the current palette settings to file. You can load it later as a preset.", menuScaling );
    ImGui::PopStyleVar();

    ImVec2 windowSize( cModalWindowWidth * menuScaling, 0.0f );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { 2.0f * cDefaultInnerSpacing * menuScaling, cDefaultInnerSpacing * menuScaling } );
    if ( ImGui::BeginModalNoAnimation( popupName.c_str(), nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            PushFont( headerFont );

        const auto headerWidth = CalcTextSize( "Save Palette" ).x;

        ImGui::SetCursorPosX( ( windowSize.x - headerWidth ) * 0.5f );
        Text( "Save Palette" );

        if ( headerFont )
            PopFont();

        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cInputPadding * menuScaling } );
        static std::string currentPaletteName;

        ImGui::SetNextItemWidth( windowSize.x - 2 * style.WindowPadding.x - style.ItemInnerSpacing.x - CalcTextSize( "Palette Name" ).x );
        ImGui::InputText( "Palette Name", currentPaletteName );
        ImGui::PopStyleVar();

        const float btnWidth = cModalButtonWidth * menuScaling;

        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        bool valid = !currentPaletteName.empty() && !hasProhibitedChars( currentPaletteName );
        if ( UI::button( "Save", valid, Vector2f( btnWidth, 0 ) ) )
        {
            std::error_code ec;
            if ( std::filesystem::is_regular_file( PalettePresets::getPalettePresetsFolder() / ( currentPaletteName + ".json" ), ec ) )
            {
                OpenPopup( "Palette already exists##PaletteHelper" );
            }
            else
            {
                auto res = PalettePresets::savePreset( currentPaletteName, palette );
                if ( res.has_value() )
                {
                    presetName = currentPaletteName;
                    ImGui::CloseCurrentPopup();
                }
                else
                {
                    showError( res.error() );
                }
            }
        }
        ImGui::PopStyleVar();
        if ( !valid )
        {
            UI::setTooltipIfHovered( currentPaletteName.empty() ?
                "Cannot save palette with empty name" :
                "Please do not any of these symbols: \? * / \\ \" < >", menuScaling );
        }

        bool closeTopPopup = false;
        if ( ImGui::BeginModalNoAnimation( "Palette already exists##PaletteHelper", nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar ) )
        {
            ImGui::Text( "Palette preset with this name already exists, override?" );
            auto w = GetContentRegionAvail().x;
            auto p = GetStyle().FramePadding.x;
            if ( UI::buttonCommonSize( "Yes", Vector2f( ( w - p ) * 0.5f, 0 ), ImGuiKey_Enter ) )
            {
                auto res = PalettePresets::savePreset( currentPaletteName, palette );
                if ( res.has_value() )
                {
                    presetName = currentPaletteName;
                    closeTopPopup = true;
                    ImGui::CloseCurrentPopup();
                }
                else
                {
                    showError( res.error() );
                }
            }
            ImGui::SameLine( 0, p );
            if ( UI::buttonCommonSize( "No", Vector2f( ( w - p ) * 0.5f, 0 ),ImGuiKey_Escape ) )
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if ( closeTopPopup )
            ImGui::CloseCurrentPopup();

        ImGui::SameLine();

        ImGui::SetCursorPosX( windowSize.x - btnWidth - style.WindowPadding.x );
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        if ( UI::buttonCommonSize( "Cancel", Vector2f( btnWidth, 0 ), ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();

        ImGui::EndPopup();
    }
    PopStyleVar( 3 );

    // for linear texture filter, uv-coordinates depend on texture size
    if ( bool( changes & ImGui::PaletteChanges::Texture ) && palette.getTexture().filter == FilterType::Linear )
        changes |= ImGui::PaletteChanges::Ranges;

    return changes;
}

void Plane( MR::PlaneWidget& planeWidget, float menuScaling, PlaneWidgetFlags flags )
{
    float dragspeed = planeWidget.box().diagonal() * 1e-3f;
    auto setDefaultPlane = [&] ( const MR::Vector3f& normal )
    {
        planeWidget.definePlane();
        planeWidget.updatePlane( MR::Plane3f::fromDirAndPt( normal, planeWidget.box().min + normal * dragspeed ) );
        if ( planeWidget.isInLocalMode() )
            planeWidget.setLocalShift( dragspeed );
    };
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { MR::cDefaultItemSpacing * menuScaling, MR::cDefaultWindowPaddingY * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { MR::cDefaultItemSpacing * menuScaling, MR::cDefaultItemSpacing * menuScaling } );

    float p = ImGui::GetStyle().FramePadding.x;
    ImVec2 iconSize = { 32 * menuScaling, 32 * menuScaling };
    ImVec2 buttonSize = { ( GetContentRegionAvail().x - 3 * p ) / 4, 70.0f * menuScaling };
    if ( MR::UI::buttonIcon( "Plane YZ", iconSize, "Plane YZ", buttonSize ) )
        setDefaultPlane( MR::Vector3f::plusX() );
    ImGui::SameLine( 0, p );
    if ( MR::UI::buttonIcon( "Plane XZ", iconSize, "Plane XZ", buttonSize ) )
        setDefaultPlane( MR::Vector3f::plusY() );
    ImGui::SameLine( 0, p );
    if ( MR::UI::buttonIcon( "Plane XY", iconSize, "Plane XY", buttonSize ) )
        setDefaultPlane( MR::Vector3f::plusZ() );
    ImGui::SameLine( 0, p );

    const bool importPlaneModeOld = planeWidget.importPlaneMode();
    if ( importPlaneModeOld )
        ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ButtonActive ) );

    if ( MR::UI::buttonIcon( "Plane Import", iconSize, "Import Plane", buttonSize ) )
    {
        planeWidget.setImportPlaneMode( !planeWidget.importPlaneMode() );
    }
    else if ( ImGui::IsMouseReleased( ImGuiMouseButton_Left ) && importPlaneModeOld == planeWidget.importPlaneMode() )
    {
        planeWidget.setImportPlaneMode( false );
    }
    if ( importPlaneModeOld )
        ImGui::PopStyleColor();

    if ( planeWidget.importPlaneMode() )
        ImGui::Text( "%s", "Click on the plane object in scene to import its parameters" );

    if ( !planeWidget.getPlaneObject() )
    {
        ImGui::PopStyleVar( 2 );
        return;
    }

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { ImGui::GetStyle().FramePadding.x, MR::cGradientButtonFramePadding * menuScaling } );

    auto localShift = planeWidget.getLocalShift();
    auto planeBackUp = planeWidget.getPlane();
    auto plane = planeWidget.getPlane();

    ImGui::SetNextItemWidth( 200.0f * menuScaling );
    UI::drag<NoUnit>( "Normal", plane.n, 0.001f );
    ImGui::PushButtonRepeat( true );

    const float arrowButtonSize = 2.0f * MR::cGradientButtonFramePadding * menuScaling + ImGui::GetTextLineHeight();
    ImFont* iconsFont = MR::RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::Icons );
    if ( iconsFont )
    {
        iconsFont->Scale = MR::cDefaultFontSize / MR::cBigIconSize;
        ImGui::PushFont( iconsFont );
    }

    auto& shift = planeWidget.isInLocalMode() ? localShift : plane.d;
    auto shiftBackUp = shift;

    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { MR::cDefaultItemSpacing * menuScaling * 0.5f, MR::cDefaultWindowPaddingY * menuScaling } );
    if ( MR::UI::button( "\xef\x84\x84", { arrowButtonSize, arrowButtonSize } ) )
        shift -= dragspeed;

    ImGui::SameLine();
    if ( MR::UI::button( "\xef\x84\x85", { arrowButtonSize, arrowButtonSize } ) )
        shift += dragspeed;
    ImGui::PopStyleVar();
    if ( iconsFont )
    {
        iconsFont->Scale = 1.0f;
        ImGui::PopFont();
    }

    ImGui::SameLine();
    ImGui::PopButtonRepeat();

    ImGui::SetNextItemWidth( 80.0f * menuScaling );
    UI::drag<LengthUnit>( "Shift", shift, dragspeed );

    ImGui::SameLine();
    if ( MR::UI::button( "Flip", { 60.0f * menuScaling, 0 } ) )
        plane = -plane;

    ImGui::PopStyleVar();
    if ( !bool( flags & PlaneWidgetFlags::DisableVisibility ) )
        ImGui::Separator();

    auto planeObj = planeWidget.getPlaneObject();
    if ( planeObj && !bool( flags & PlaneWidgetFlags::DisableVisibility ) )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { ImGui::GetStyle().FramePadding.x, MR::cCheckboxPadding * menuScaling } );
        bool showPlane = planeWidget.getPlaneObject()->isVisible();
        if ( MR::UI::checkbox( "Show Plane", &showPlane ) )
            planeWidget.getPlaneObject()->setVisible( showPlane );
        ImGui::PopStyleVar();
    }

    if ( planeWidget.isInLocalMode() && shiftBackUp != shift )
    {
        planeWidget.setLocalShift( shift );
        plane.d += ( shift - shiftBackUp );
    }

    if ( planeBackUp != plane )
        planeWidget.updatePlane( plane, plane.n != planeBackUp.n );

    ImGui::PopStyleVar( 2 );
}

void Image( const MR::ImGuiImage& image, const ImVec2& size, const MR::Color& multColor )
{
    MR::Vector4f tintColor { multColor };
    Image( image, size, tintColor );
}

void Image( const MR::ImGuiImage& image, const ImVec2& size, const ImVec4& multColor )
{
    Image( image.getImTextureId(), size, ImVec2( 0, 1 ), ImVec2( 1, 0 ), multColor );
}

MR::Vector2i GetImagePointerCoord( const MR::ImGuiImage& image, const ImVec2& size, const ImVec2& imagePos )
{
    const auto& io = ImGui::GetIO();
    return  { int( ( io.MousePos.x - imagePos.x ) / size.x * image.getImageWidth() ), int( ( size.y - io.MousePos.y + imagePos.y ) / size.y * image.getImageHeight() ) };
}

void Spinner( float radius, float scaling )
{
    auto pos = GetCursorScreenPos();

    static float angle = 0.0f;
    const int numCircles = 7;
    auto color = ImGui::GetColorU32( ImGui::GetStyleColorVec4( ImGuiCol_Text ) );
    for ( int i = 0; i < numCircles; ++i )
    {
        float angleShift = float( i ) / float( numCircles ) * MR::PI_F * 2.0f;
        ImVec2 center = ImVec2( pos.x + radius * std::cos( angle + angleShift ), pos.y + radius * std::sin( angle + angleShift ) );
        ImGui::GetWindowDrawList()->AddCircleFilled( center, radius * 0.1f * scaling, color );
    }
    angle += ImGui::GetIO().DeltaTime * 2.2f;

    SetCursorPosY( GetCursorPosY() + radius );
    ImGui::Dummy( ImVec2( 0, 0 ) );
    MR::getViewerInstance().incrementForceRedrawFrames();
}

bool ModalBigTitle( const char* title, float scaling )
{
    auto font = MR::RibbonFontManager::getFontByTypeStatic( MR::RibbonFontManager::FontType::Headline );
    if ( font )
        ImGui::PushFont( font );
    ImGui::Text( "%s", title);
    if ( font )
        ImGui::PopFont();

    const float exitButtonSize = MR::StyleConsts::Modal::exitBtnSize * scaling;
    ImGui::SameLine( ImGui::GetWindowContentRegionMax().x - exitButtonSize );
    const bool shoudClose = ModalExitButton( scaling );
    ImGui::NewLine();

    return shoudClose;
}

bool ModalExitButton( float scaling )
{
    const uint32_t crossColor = MR::ColorTheme::getRibbonColor( MR::ColorTheme::RibbonColorsType::TabClicked ).getUInt32();
    ImGui::PushStyleColor( ImGuiCol_Button, 0 );
    ImGui::PushStyleColor( ImGuiCol_Border, 0 );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, 0x80808080 );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, 0x80808080 );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 3.0f * scaling );

    auto drawList = ImGui::GetWindowDrawList();
    const auto pos = ImGui::GetCursorScreenPos();
    const float buttonSize = MR::StyleConsts::Modal::exitBtnSize * scaling;

    if ( ImGui::Button( "##ExitButton", ImVec2( buttonSize, buttonSize ) ) || ImGui::IsKeyPressed( ImGuiKey_Escape ) )
    {
        ImGui::CloseCurrentPopup();
        ImGui::PopStyleColor( 4 );
        ImGui::PopStyleVar();
        return true;
    }
    const float crossSize = 10.0f * scaling;
    auto shift = ( buttonSize - crossSize ) * 0.5f;
    drawList->AddLine( { pos.x + shift, pos.y + shift }, { pos.x + buttonSize - shift - scaling, pos.y + buttonSize - shift - scaling }, crossColor, 2.0f * scaling );
    drawList->AddLine( { pos.x + shift, pos.y + buttonSize - shift - scaling }, { pos.x + buttonSize - shift - scaling, pos.y + shift }, crossColor, 2.0f * scaling );

    ImGui::PopStyleColor( 4 );
    ImGui::PopStyleVar();
    return false;
}

} // namespace ImGui
