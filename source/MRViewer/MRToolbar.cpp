#include "MRToolbar.h"
#include "MRViewer/MRUITestEngine.h"
#include "imgui.h"
#include "MRRibbonConstants.h"
#include "ImGuiHelpers.h"
#include "MRRibbonButtonDrawer.h"
#include "MRPch/MRSpdlog.h"
#include "MRRibbonSchema.h"
#include "MRColorTheme.h"
#include "MRRibbonMenu.h"
#include "MRMesh/MRVector2.h"
#include "imgui/imgui_internal.h"
#include "MRUIStyle.h"
#include "MRViewer.h"

namespace MR
{

const int cToolbarMaxItemCount = 14;


void Toolbar::openCustomize()
{
    openCustomizeFlag_ = true;
    itemsListCustomize_ = itemsList_;
    customizeTabNum_ = 0;
    searchResult_.clear();
    searchResult_.resize( RibbonSchemaHolder::schema().tabsMap.size() );
}

void Toolbar::setRibbonMenu( RibbonMenu* ribbonMenu )
{
    ribbonMenu_ = ribbonMenu;
    if ( ribbonMenu_ )
        setScaling( ribbonMenu_->menu_scaling() );
}

void Toolbar::drawToolbar()
{
    if ( !ribbonMenu_ )
        return;

    const auto& buttonDrawer = ribbonMenu_->getRibbonButtonDrawer();
    const auto& fontManager = ribbonMenu_->getFontManager();

    auto windowPadding = ImVec2( 12 * scaling_, 4 * scaling_ );
    auto itemSpacing = ImVec2( 12 * scaling_, 0 );
    const ImVec2 itemSize = { cQuickAccessBarHeight * scaling_ - 2.0f * windowPadding.y, cQuickAccessBarHeight * scaling_ - 2.0f * windowPadding.y };
    const ImVec2 customizeBtnSize = ImVec2( itemSize.x / 2.f, itemSize.y );

    int itemCount = 0;
    int droppedItemCount = 0;
    //TODO calc if list changes
    for ( const auto& item : itemsList_ )
    {
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
            continue;
        ++itemCount;
        if ( it->second.item->type() == RibbonItemType::ButtonWithDrop )
            ++droppedItemCount;
    }

    if ( !itemCount )
    {
        currentWidth_ = 0.0f;
        return;
    }
    ++itemCount;

    currentWidth_ = windowPadding.x * 2
        + itemSize.x * itemCount
        + itemSize.x * cSmallItemDropSizeModifier * droppedItemCount
        + itemSpacing.x * ( itemCount - 1 )
        + customizeBtnSize.x
        + itemSpacing.x / 2.f;

    const Vector2i sceneSize = ribbonMenu_->getSceneSize();
    if ( currentWidth_ >= getViewerInstance().framebufferSize.x - sceneSize.x )
    {
        currentWidth_ = 0.0f;
        return; // dont show quick panel if window is too small
    }

    const float windowPosX = std::max( getViewerInstance().framebufferSize.x / 2.f - currentWidth_ / 2.f, sceneSize.x - 1.0f );

    const int currentTopPanelHeight = ribbonMenu_->getTopPanelCurrentHeight();
    ImGui::SetNextWindowPos( ImVec2( windowPosX, float( currentTopPanelHeight ) * scaling_ - 1 ) );
    ImGui::SetNextWindowSize( ImVec2( currentWidth_, cQuickAccessBarHeight * scaling_ ), ImGuiCond_Always );

    ImGui::PushStyleColor( ImGuiCol_WindowBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, windowPadding );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 1.0f );
    ImGui::Begin(
        "QuickAccess##[rect_allocator_ignore]", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoFocusOnAppearing
    );
    ImGui::PopStyleVar( 2 );
    ImGui::PopStyleColor();

    DrawButtonParams params{ DrawButtonParams::SizeType::Small, itemSize, cMiddleIconSize,DrawButtonParams::RootType::Toolbar };

    ImGui::PushFont( fontManager.getFontByType( RibbonFontManager::FontType::Small ) );
    UI::TestEngine::pushTree( "Toolbar" );
    for ( const auto& item : itemsList_ )
    {
        auto it = RibbonSchemaHolder::schema().items.find( item );
        if ( it == RibbonSchemaHolder::schema().items.end() )
        {
#ifndef __EMSCRIPTEN__
            spdlog::warn( "Plugin \"{}\" not found!", item ); // TODO don't flood same message
#endif
            continue;
        }

        buttonDrawer.drawButtonItem( it->second, params );
        ImGui::SameLine();
    }

    auto activeListIt = RibbonSchemaHolder::schema().items.find( "Active Plugins List" );
    if ( activeListIt != RibbonSchemaHolder::schema().items.end() )
    {
        ribbonMenu_->setActiveListPos( ImGui::GetCursorScreenPos() );
        CustomButtonParameters cParams;
        cParams.iconType = RibbonIcons::IconType::RibbonItemIcon;
        cParams.pushColorsCb = [] ( bool enabled, bool )->int
        {
            if ( !enabled )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_Button, Color( 0, 0, 0, 0 ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarClicked ).getUInt32() );
            }
            else
            {
                ImGui::PushStyleColor( ImGuiCol_Text, Color::white().getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_Button, Color( 60, 169, 20, 255 ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonHovered, Color( 60, 169, 20, 200 ).getUInt32() );
                ImGui::PushStyleColor( ImGuiCol_ButtonActive, Color( 60, 169, 20, 255 ).getUInt32() );
            }
            return 4;
        };
        buttonDrawer.drawCustomButtonItem( activeListIt->second, cParams, params );
        ImGui::SameLine();
    }
    UI::TestEngine::popTree(); // "Toolbar"

    ImGui::SetCursorPosX( ImGui::GetCursorPosX() - ImGui::GetStyle().ItemSpacing.x / 2.f );

    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarClicked ).getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_Button, Color( 0, 0, 0, 0 ).getUInt32() );
    ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Text ).getUInt32() );

    ImFont* font = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
    if ( font )
    {
        font->Scale = customizeBtnSize.y * 0.5f / ( cBigIconSize * scaling_ );
        ImGui::PushFont( font );
    }

    const char* text = "\xef\x85\x82";
    auto textSize = ImGui::CalcTextSize( text );
    auto textPos = ImVec2( ImGui::GetCursorPosX() + ( customizeBtnSize.x - textSize.x ) / 2.f,
                           ImGui::GetCursorPosY() + ( customizeBtnSize.y - textSize.y ) / 2.f );
    if ( ImGui::Button( "##ToolbarCustomizeBtn", customizeBtnSize ) )
        openCustomize();

    ImGui::SetCursorPos( textPos );
    ImGui::Text( "%s", text );

    if ( font )
    {
        ImGui::PopFont();
        font->Scale = 1.0f;
    }

    ImGui::PopStyleColor( 4 );

    ImGui::PopStyleVar();
    ImGui::PopFont();

    ImGui::End();
}

void Toolbar::drawCustomize()
{
    ImGui::SetNextWindowPos( ImVec2( -100, -100 ) );
    ImGui::SetNextWindowSize( ImVec2( 1, 1 ) );
    ImGui::Begin( "Toolbar Customize##BaseWindow", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs );
    if ( openCustomizeFlag_ )
    {
        openCustomizeFlag_ = false;
        ImGui::OpenPopup( "Toolbar Customize" );
    }
    drawCustomizeModal_();
    ImGui::End();
}

void Toolbar::readItemsList( const Json::Value& root )
{
    RibbonSchemaLoader::readMenuItemsList( root, itemsList_ );
    for ( auto it = itemsListMigrations_.upper_bound( itemsListVersion_ ); it != itemsListMigrations_.end(); ++it )
    {
        const auto& [migrationVersion, migrationRule] = *it;
        migrationRule( itemsList_ );
        itemsListVersion_ = migrationVersion;
    }
}

void Toolbar::resetItemsList()
{
    itemsList_ = RibbonSchemaHolder::schema().defaultQuickAccessList;
}

void Toolbar::drawCustomizeModal_()
{
    if ( !ribbonMenu_ )
        return;

    const auto& buttonDrawer = ribbonMenu_->getRibbonButtonDrawer();

    ImVec2 windowPaddingSize = ImVec2( 3 * cDefaultItemSpacing * scaling_, 3 * cDefaultItemSpacing * scaling_ );
    ImVec2 childWindowPadding = ImVec2( 12 * scaling_, 4 * scaling_ );
    ImVec2 itemSpacing = ImVec2( 12 * scaling_, 0 );
    const ImVec2 smallItemSize = { cQuickAccessBarHeight * scaling_ - 2.0f * childWindowPadding.y, cQuickAccessBarHeight * scaling_ - 2.0f * childWindowPadding.y };

    const float itemsWindowWidth = childWindowPadding.x * 2
        + smallItemSize.x * cToolbarMaxItemCount
        + itemSpacing.x * ( cToolbarMaxItemCount - 1 );

    ImVec2 windowSize( itemsWindowWidth + windowPaddingSize.x * 2, 530 * scaling_ );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos( center, ImGuiCond_Appearing, ImVec2( 0.5f, 0.5f ) );
    ImGui::SetNextWindowSizeConstraints( ImVec2( windowSize.x, -1 ), ImVec2( windowSize.x, 0 ) );

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, windowPaddingSize );
    if ( !ImGui::BeginModalNoAnimation( "Toolbar Customize", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        ImGui::PopStyleVar();
        return;
    }

    if ( ImGui::ModalBigTitle( "Customize Viewport Toolbar", scaling_ ) )
    {
        ImGui::CloseCurrentPopup();
        searchString_.clear();
    }

    ImGui::Text( "%s", "Select icons to show in Toolbar" );

    ImGui::SameLine();
    auto& style = ImGui::GetStyle();
    float textPosX = windowSize.x - ImGui::CalcTextSize( "Icons in Toolbar : 00/00" ).x - style.WindowPadding.x;
    ImGui::SetCursorPosX( textPosX );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 0, 12 * scaling_ ) );
    ImGui::Text( "Icons in Toolbar : %02d/%02d", int( itemsListCustomize_.size() ), cToolbarMaxItemCount );
    ImGui::PopStyleVar();

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, childWindowPadding );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );

    DrawButtonParams params{ DrawButtonParams::SizeType::Small, smallItemSize, cMiddleIconSize, DrawButtonParams::RootType::Toolbar };

    ImGui::PushStyleColor( ImGuiCol_ChildBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
    ImGui::BeginChild( "##QuickAccessCustomizeItems", ImVec2( itemsWindowWidth, smallItemSize.y + childWindowPadding.y * 2 ), true );
    ImGui::PopStyleColor();

    ImVec2 tooltipSize = ImVec2( Vector2f::diagonal( 4 * scaling_ ) + Vector2f( params.itemSize ) );
    Vector2f tooltipContourSize = Vector2f::diagonal( 2 * scaling_ ) + Vector2f( params.itemSize );
    for ( int i = 0; i < itemsListCustomize_.size(); ++i )
    {
        const auto& itemPreview = itemsListCustomize_[i];
        auto iterItemPreview = RibbonSchemaHolder::schema().items.find( itemPreview );
        if ( iterItemPreview == RibbonSchemaHolder::schema().items.end() )
        {
#ifndef __EMSCRIPTEN__
            spdlog::warn( "Plugin \"{}\" not found!", itemPreview ); // TODO don't flood same message
#endif
            continue;
        }

        ImVec2 cursorPos = ImGui::GetCursorPos();
        ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::ToolbarHovered ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Border, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 0.f );
        ImGui::Button( ( "##ItemBtn" + std::to_string( i ) ).c_str(), params.itemSize );
        ImGui::PopStyleVar();
        ImGui::SetNextItemAllowOverlap();

        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2() );
        ImGui::SetNextWindowSize( tooltipSize );

        ImGui::PushStyleColor( ImGuiCol_Border, 0 );
        ImGui::PushStyleColor( ImGuiCol_WindowBg, 0 );
        if ( ImGui::BeginDragDropSource( ImGuiDragDropFlags_AcceptNoDrawDefaultRect ) )
        {
            ImGui::SetDragDropPayload( "ToolbarItemNumber", &i, sizeof( int ) );
            const auto& item = itemsList_[i];
            auto iterItem = RibbonSchemaHolder::schema().items.find( item );
            ImGui::SetCursorPos( ImVec2( 1 * scaling_, 1 * scaling_ ) );
            UI::button( "##ToolbarDragDropBtnHighlight", tooltipContourSize );
            ImGui::SetCursorPos( ImVec2( 2 * scaling_, 2 * scaling_ ) );
            ImGui::Button( "##ToolbarDragDropBtn", params.itemSize);
            ImGui::SetCursorPos( ImVec2( 2 * scaling_, 2 * scaling_ ) );
            if ( iterItem != RibbonSchemaHolder::schema().items.end() )
                buttonDrawer.drawButtonIcon( iterItem->second, params );
            ImGui::EndDragDropSource();
            dragDrop_ = true;
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor( 6 );

        const ImGuiPayload* peekPayload = ImGui::GetDragDropPayload();
        if ( dragDrop_ && ( !peekPayload || !peekPayload->IsDataType( "ToolbarItemNumber" ) ) )
        {
            itemsListCustomize_ = itemsList_;
            dragDrop_ = false;
        }
        if ( ImGui::IsItemHovered( ImGuiHoveredFlags_AllowWhenBlockedByActiveItem ) )
        {
            if ( peekPayload && peekPayload->IsDataType( "ToolbarItemNumber" ) )
            {
                assert( peekPayload->DataSize == sizeof( int ) );
                int oldIndex = *( const int* )peekPayload->Data;
                itemsListCustomize_ = itemsList_;
                auto movedItem = itemsListCustomize_[oldIndex];
                itemsListCustomize_.erase( itemsListCustomize_.begin() + oldIndex );
                itemsListCustomize_.insert( itemsListCustomize_.begin() + i, movedItem );
            }
        }

        if ( ImGui::BeginDragDropTarget() )
        {
            const ImGuiPayload* payload = ImGui::AcceptDragDropPayload( "ToolbarItemNumber" );
            if ( payload )
            {
                assert( payload->DataSize == sizeof( int ) );
                itemsList_ = itemsListCustomize_;
                dragDrop_ = false;
            }
            ImGui::EndDragDropTarget();
        }
        ImGui::SetCursorPos( cursorPos );

        auto screenPos = Vector2f( ImGui::GetCursorScreenPos() );
        dashedRect_( screenPos, screenPos + Vector2f::diagonal( params.itemSize.x - 1 * scaling_ ), 10.f, 0.5f,
            ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ) );
        buttonDrawer.drawButtonIcon( iterItemPreview->second, params );

        ImGui::SameLine( 0, childWindowPadding.x + 3 * scaling_ );
    }

    for ( int i = int( itemsListCustomize_.size() ); i < cToolbarMaxItemCount; ++i )
    {
        auto screenPos = Vector2f( ImGui::GetCursorScreenPos() );
        ImGui::PushStyleColor( ImGuiCol_Button, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
        ImGui::PushStyleColor( ImGuiCol_Border, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::QuickAccessBackground ).getUInt32() );
        ImGui::PushStyleVar( ImGuiStyleVar_FrameRounding, 0.f );
        ImGui::Button( ( "##ItemBtn" + std::to_string( i ) ).c_str(), params.itemSize );
        ImGui::PopStyleVar();
        ImGui::PopStyleColor( 4 );
        dashedRect_( screenPos, screenPos + Vector2f::diagonal( params.itemSize.x - 1 * scaling_ ), 10.f, 0.5f,
            ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ) );

        ImGui::SameLine( 0, childWindowPadding.x );
    }

    ImGui::PopStyleVar();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( ImGui::GetStyle().ItemSpacing.x, 12 * scaling_ ) );
    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::BeginChild( "##QuickAccessCustomizeTabsList", ImVec2( 130 * scaling_, -1 ) );
    drawCustomizeTabsList_();
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0, 0 ) );
    ImGui::BeginChild( "##QuickAccessCustomizeAndSearch", ImVec2( -1, -1 ) );
    ImGui::PopStyleVar();
    const float buttonWidth = cGradientButtonFramePadding * 2 * scaling_ + ImGui::CalcTextSize( "Reset to default" ).x;
    const float searchWidth = ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x - buttonWidth;

    ImGui::SetNextItemWidth( searchWidth );
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( style.FramePadding.x, 8 * scaling_ ) );
    if ( ImGui::InputText( "##QuickAccessSearch", searchString_ ) )
    {
        searchResult_.clear();
        searchResult_.resize( RibbonSchemaHolder::schema().tabsMap.size() );
        auto searchResRaw_ = RibbonSchemaHolder::search( searchString_ );
        for ( const auto& sr : searchResRaw_ )
        {
            if ( sr.tabIndex < 0 )
                continue;
            searchResult_[sr.tabIndex].push_back( sr.item->item->name() );
        }
    }
    ImGui::PopStyleVar();
    ImGui::SameLine();
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( ImGui::GetStyle().ItemSpacing.x, 12 * scaling_ ) );
    if ( UI::button( "Reset to default", Vector2f( buttonWidth, 0 ) ) )
    {
        resetItemsList();
        itemsListCustomize_ = itemsList_;
    }
    ImGui::PopStyleVar();

    ImGui::PushStyleColor( ImGuiCol_ChildBg, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Background ).getUInt32() );
    ImGui::BeginChild( "##QuickAccessCustomizeItemsList", ImVec2( -1, -1 ), true );
    ImGui::PopStyleColor();

    drawCustomizeItemsList_();

    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::PopStyleVar();

    ImGui::PopStyleVar();

    ImGui::EndPopup();
}

void Toolbar::drawCustomizeTabsList_()
{
    auto& schema = RibbonSchemaHolder::schema();
    auto& tabsOrder = schema.tabsOrder;
    auto& tabsMap = schema.tabsMap;
    auto& groupsMap = schema.groupsMap;

    auto colorActive = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::GradientStart ).getUInt32();
    auto colorInactive = ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::Borders ).getUInt32();
    auto colorBg = ImGui::GetStyleColorVec4( ImGuiCol_ChildBg );
    ImGui::PushStyleColor( ImGuiCol_Button, colorBg );
    ImGui::PushStyleColor( ImGuiCol_ButtonHovered, colorBg );
    ImGui::PushStyleColor( ImGuiCol_ButtonActive, colorBg );
    ImGui::PushStyleColor( ImGuiCol_Border, colorBg );
    const float circleShiftY = ImGui::GetTextLineHeight() / 2 + ImGui::GetStyle().ItemSpacing.y;
    for ( int i = 0; i < tabsOrder.size(); ++i )
    {
        const auto& tabName = tabsOrder[i].name;
        auto tabIt = tabsMap.find( tabName );
        if ( tabIt == tabsMap.end() )
            continue;

        bool anySelected = false;
        bool anyFounded = searchString_.empty() || ( !searchString_.empty() && !searchResult_[i].empty() );
        auto& tab = tabIt->second;
        for ( auto& group : tab )
        {
            auto itemsIt = groupsMap.find( tabName + group );
            if ( itemsIt == groupsMap.end() )
                continue;
            auto& items = itemsIt->second;
            for ( auto& item : items )
            {
                if ( item == "Quick Access Settings" )
                    continue;

                auto itemIt = std::find( itemsListCustomize_.begin(), itemsListCustomize_.end(), item );
                if ( itemIt != itemsListCustomize_.end() )
                    anySelected = true;

            }
        }

        int changedColor = 0;
        if ( !anyFounded )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
            ++changedColor;
        }
        if ( i == customizeTabNum_ )
        {
            const auto& color = anyFounded ? colorActive : colorInactive;
            ImGui::PushStyleColor( ImGuiCol_Button, color );
            ImGui::PushStyleColor( ImGuiCol_ButtonHovered, color );
            ImGui::PushStyleColor( ImGuiCol_ButtonActive, color );
            ImGui::PushStyleColor( ImGuiCol_Border, color );
            changedColor += 4;
        }

        if ( ImGui::Button( tabName.c_str() ) )
            customizeTabNum_ = i;
        if ( anySelected )
        {
            ImGui::SameLine();
            ImVec2 pos = ImGui::GetCursorScreenPos();
            pos.y += circleShiftY;
            ImGui::GetWindowDrawList()->AddCircleFilled( pos, 2 * scaling_, colorActive );
            ImGui::NewLine();
        }
        if ( changedColor )
            ImGui::PopStyleColor( changedColor );
    }
    ImGui::PopStyleColor( 4 );
}

void Toolbar::drawCustomizeItemsList_()
{
    if ( !ribbonMenu_ )
        return;
    const auto& buttonDrawer = ribbonMenu_->getRibbonButtonDrawer();

    auto& schema = RibbonSchemaHolder::schema();
    auto& tabsOrder = schema.tabsOrder;
    auto& tabsMap = schema.tabsMap;
    auto& groupsMap = schema.groupsMap;

    bool canAdd = int( itemsListCustomize_.size() ) < cToolbarMaxItemCount;

    if ( customizeTabNum_ >= tabsOrder.size() || customizeTabNum_ < 0 )
        return;

    const auto& tabName = tabsOrder[customizeTabNum_].name;
    auto tabIt = tabsMap.find( tabName );
    if ( tabIt == tabsMap.end() )
        return;
    auto& tab = tabIt->second;
    float width = ImGui::GetWindowContentRegionMax().x;
    int countInColumn = 11;
    float heightStep = ImGui::GetWindowContentRegionMax().y / countInColumn;
    auto posShift = ImGui::GetCursorPos();
    posShift.y = heightStep / 2 - ImGui::GetTextLineHeight() / 2.f;

    auto drawItemCheckbox = [&, searchMode = !searchString_.empty()]( const std::string& item, bool founded )
    {
        auto itemIt = std::find( itemsListCustomize_.begin(), itemsListCustomize_.end(), item );
        bool itemInQA = itemIt != itemsListCustomize_.end();

        bool disabled = !canAdd && !itemInQA;
        int colorChanged = 0;
        if ( disabled )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
            ImGui::PushStyleColor( ImGuiCol_FrameBgActive, ImGui::GetColorU32( ImGuiCol_FrameBg ) );
            ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ImGui::GetColorU32( ImGuiCol_FrameBg ) );
            colorChanged += 3;
        }
        else if ( searchMode && !founded )
        {
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::TextDisabled ).getUInt32() );
            ++colorChanged;
        }

        bool checkboxChanged = false;
        auto schemaItem = RibbonSchemaHolder::schema().items.find( item );
        if ( schemaItem != RibbonSchemaHolder::schema().items.end() )
            checkboxChanged = buttonDrawer.GradientCheckboxItem( schemaItem->second, &itemInQA );
        else
            checkboxChanged = UI::checkbox( item.c_str(), &itemInQA );

        if ( checkboxChanged )
        {
            if ( itemInQA )
            {
                if ( canAdd )
                {
                    itemsListCustomize_.emplace_back( item );
                    itemsList_ = itemsListCustomize_;
                }
                else
                    itemInQA = false;
            }
            else
            {
                itemsListCustomize_.erase( itemIt );
                itemsList_ = itemsListCustomize_;
            }
        }

        if ( colorChanged )
            ImGui::PopStyleColor( colorChanged );
    };


    int itemCounter = 0;
    const auto& toolbarSearchTabItems = searchResult_[customizeTabNum_];
    for ( const auto& item : toolbarSearchTabItems )
    {
        if ( item == "Quick Access Settings" )
            continue;

        ImGui::SetCursorPos( ImVec2( itemCounter / countInColumn * width / 2 + posShift.x, itemCounter % countInColumn * heightStep + posShift.y ) );
        ++itemCounter;

        drawItemCheckbox( item, true );
    }
    bool skipFounded = !searchString_.empty() && !toolbarSearchTabItems.empty();
    for ( auto& group : tab )
    {
        auto itemsIt = groupsMap.find( tabName + group );
        if ( itemsIt == groupsMap.end() )
            continue;
        auto& items = itemsIt->second;
        for ( auto& item : items )
        {
            if ( item == "Quick Access Settings" )
                continue;

            if ( skipFounded && std::find( toolbarSearchTabItems.begin(), toolbarSearchTabItems.end(), item ) != toolbarSearchTabItems.end() )
                continue;

            ImGui::SetCursorPos( ImVec2( itemCounter / countInColumn * width / 2 + posShift.x, itemCounter % countInColumn * heightStep + posShift.y ) );
            ++itemCounter;

            drawItemCheckbox( item, false );
        }
    }
}

void Toolbar::dashedLine_( const Vector2f& org, const Vector2f& dest, float periodLength /*= 10.f*/, float fillRatio /*= 0.5f*/,
    const Color& color /*= Color::gray()*/, float /*periodStart*/ /*= 0.f */ )
{
    fillRatio = std::clamp( fillRatio, 0.f, 1.f );
    float periodCointF = ( dest - org ).length() / periodLength;
    int periodCointI = int( std::floor( periodCointF ) );

    Vector2f dir = ( dest - org ) / periodCointF;
    for ( int i = 0; i < periodCointI; ++i )
    {
        const ImVec2 begin = ImVec2( org + dir * float( i ) );
        const ImVec2 end = ImVec2( org + dir * ( i + fillRatio ) );
        ImGui::GetForegroundDrawList()->AddLine( begin, end, color.getUInt32() );
    }

    const ImVec2 begin = ImVec2( org + dir * float( periodCointI ) );
    const ImVec2 end = ImVec2( org + dir * std::min( periodCointI + fillRatio, periodCointF ) );
    ImGui::GetForegroundDrawList()->AddLine( begin, end, color.getUInt32() );
}

void Toolbar::dashedRect_( const Vector2f& leftTop, const Vector2f& rightBottom, float periodLength /*= 10.f*/, float fillRatio /*= 0.5f*/,
    const Color& color /*= Color::gray()*/ )
{
    const Vector2f rightTop( rightBottom.x, leftTop.y );
    const Vector2f leftBottom( leftTop.x, rightBottom.y );
    dashedLine_( leftTop, rightTop, periodLength, fillRatio, color );
    dashedLine_( rightTop, rightBottom, periodLength, fillRatio, color );
    dashedLine_( rightBottom, leftBottom, periodLength, fillRatio, color );
    dashedLine_( leftBottom, leftTop, periodLength, fillRatio, color );
}

}
