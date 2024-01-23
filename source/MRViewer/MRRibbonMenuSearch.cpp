#include "MRRibbonMenuSearch.h"
#include "ImGuiHelpers.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRRibbonConstants.h"
#include "MRRibbonButtonDrawer.h"
#include <imgui_internal.h>
#include "MRViewerInstance.h"
#include "MRViewer/MRViewer.h"
#include "MRRibbonMenu.h"

namespace MR
{

constexpr float cSearchSize = 250.f;

void RibbonMenuSearch::pushRecentItem( const std::shared_ptr<RibbonMenuItem>& item )
{
    if ( !item )
        return;
    auto it = std::find_if( recentItems_.begin(), recentItems_.end(), [&] ( const auto& other )
    {
        return other.item->item == item;
    } );
    if ( it != recentItems_.end() )
    {
        std::rotate( recentItems_.begin(), it, it + 1 );
        return;
    }

    auto sIt = RibbonSchemaHolder::schema().items.find( item->name() );
    if ( sIt == RibbonSchemaHolder::schema().items.end() )
    {
        assert( false );
        return;
    }

    RibbonSchemaHolder::SearchResult res;
    res.item = &sIt->second;
    if ( recentItems_.size() < 10 )
        recentItems_.insert( recentItems_.begin(), std::move( res ) );
    else
    {
        std::rotate( recentItems_.begin(), recentItems_.end() - 1, recentItems_.end() );
        recentItems_.front() = std::move( res );
    }
}

void RibbonMenuSearch::drawWindow_( const Parameters& params )
{
    const auto& resultsList = searchLine_.empty() ? recentItems_ : searchResult_;
    if ( !isSmallUI() && resultsList.empty() )
        return;

    const float screenWidth = float( getViewerInstance().framebufferSize.x );
    const float windowPaddingX = ImGui::GetStyle().WindowPadding.x;
    ImVec2 pos;
    pos.x = std::max( screenWidth - ( 70.f + cSearchSize + 16.f ) * params.scaling - windowPaddingX, 0.f );
    pos.y = ( cTabYOffset + cTabHeight ) * params.scaling;
    ImGui::SetNextWindowPos( pos );
    ImGui::SetNextWindowSize( ImVec2( ( cSearchSize + 20 ) * params.scaling, -1 ) );

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoFocusOnAppearing;
    if ( ImGui::Begin( windowName(), NULL, window_flags ) )
    {
        if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
            deactivateSearch_();

        const float minSearchSize = cSearchSize * params.scaling;
        if ( isSmallUI() )
        {
            if ( !isSmallUILast_ || ImGui::IsWindowAppearing() )
                ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth( minSearchSize );
            if ( ImGui::InputText( "##SearchLine", searchLine_ ) )
            {
                searchResult_ = RibbonSchemaHolder::search( searchLine_ );
                hightlightedSearchItem_ = -1;
            }
            if ( !ImGui::IsWindowAppearing() &&
                !( ImGui::IsWindowFocused() || ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) ) )
                deactivateSearch_();
        }
        else
        {
            if ( !mainInputFocused_ && !isSmallUILast_ &&
                !( ImGui::IsWindowFocused() || ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) ) )
                deactivateSearch_();
        }

        if ( resultsList.empty() )
            hightlightedSearchItem_ = -1;
        else
        {
            if ( ImGui::IsKeyPressed( ImGuiKey_DownArrow ) && hightlightedSearchItem_ + 1 < resultsList.size() )
                hightlightedSearchItem_++;
            else if ( ImGui::IsKeyPressed( ImGuiKey_UpArrow ) && hightlightedSearchItem_ > 0 )
                hightlightedSearchItem_--;
        }

        ImGui::PushFont( RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Small ) );
        auto ySize = ( cSmallIconSize + 2 * cRibbonButtonWindowPaddingY ) * params.scaling;
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive,
                               ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabActive ).getUInt32() );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
        for ( int i = 0; i < resultsList.size(); ++i )
        {
            const auto& foundItem = resultsList[i];
            if ( !foundItem.item )
                continue;
            auto width = params.btnDrawer.calcItemWidth( *foundItem.item, DrawButtonParams::SizeType::SmallText );
            DrawButtonParams dbParams;
            dbParams.sizeType = DrawButtonParams::SizeType::SmallText;
            dbParams.iconSize = cSmallIconSize;
            dbParams.itemSize.y = ySize;
            dbParams.itemSize.x = width.baseWidth + width.additionalWidth + 2.0f * cRibbonButtonWindowPaddingX * params.scaling;
            dbParams.forceHovered = hightlightedSearchItem_ == i;
            dbParams.forcePressed = dbParams.forceHovered &&
                ( ImGui::IsKeyPressed( ImGuiKey_Enter ) || ImGui::IsKeyPressed( ImGuiKey_KeypadEnter ) );
            const bool pluginActive = foundItem.item->item->isActive();
            params.btnDrawer.drawButtonItem( *foundItem.item, dbParams );
            if ( foundItem.item->item->isActive() != pluginActive )
                deactivateSearch_();
        }
        ImGui::PopStyleVar( 1 );
        ImGui::PopStyleColor( 3 );
        ImGui::PopFont();
        ImGui::End();
    }
}

void RibbonMenuSearch::deactivateSearch_()
{
    active_ = false;
    searchLine_.clear();
    searchResult_.clear();
    hightlightedSearchItem_ = -1;
}

void RibbonMenuSearch::drawMenuUI( const Parameters& params )
{
    if ( isSmallUI() )
    {
        if ( smallSearchButton_( params ) && !( !active_ && activeLast_ ) )
            active_ = !active_;
    }
    else
    {
        if ( isSmallUILast_ && active_ )
            ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth( cSearchSize * params.scaling );
        if ( ImGui::InputText( "##SearchLine", searchLine_ ) )
        {
            searchResult_ = RibbonSchemaHolder::search( searchLine_ );
            hightlightedSearchItem_ = -1;
        }
        if ( mainInputFocused_ && !ImGui::IsItemFocused() && !searchLine_.empty() && searchResult_.empty() )
            deactivateSearch_();
        mainInputFocused_ = ImGui::IsItemFocused(); 
        if ( ImGui::IsItemActivated() )
            active_ = true;
        if ( ImGui::IsItemDeactivated() )
        {
            if ( ImGui::IsKeyPressed( ImGuiKey_Escape ) )
                deactivateSearch_();
        }
    }
    if ( active_ )
        drawWindow_( params );
    activeLast_ = active_;
    isSmallUILast_ = isSmallUI();
}

bool RibbonMenuSearch::isSmallUI() const
{
    auto ribbonMenu = getViewerInstance().getMenuPluginAs<RibbonMenu>();
    const auto scaling = ribbonMenu ? ribbonMenu->menu_scaling() : 1.f;
    return getViewerInstance().framebufferSize.x < 1000 * scaling;
}

float RibbonMenuSearch::getWidthMenuUI() const
{
    return isSmallUI() ? 40.f : cSearchSize + 16.f;
}

bool RibbonMenuSearch::smallSearchButton_( const Parameters& params )
{
    ImFont* font = params.fontManager.getFontByType( RibbonFontManager::FontType::Icons );
    if ( font )
        font->Scale = 0.7f;

    ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, cHeaderQuickAccessFrameRounding * params.scaling );
    ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
    if ( active_ )
        ImGui::PushStyleColor( ImGuiCol_Button, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
    else
        ImGui::PushStyleColor( ImGuiCol_Button, ImVec4( 0, 0, 0, 0 ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabHovered ) );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4( ImGuiCol_ScrollbarGrabActive ) );
    ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TabText ).getUInt32() );

    float btnSize = params.scaling * cTopPanelAditionalButtonSize;
    if ( font )
        ImGui::PushFont( font );
    bool pressed = ImGui::Button( "\xef\x80\x82", ImVec2( btnSize, btnSize ) ) ;
    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    ImGui::PopStyleColor( 4 );
    ImGui::PopStyleVar( 2 );

    return pressed;
}

}
