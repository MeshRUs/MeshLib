#include "MRRibbonNotification.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRColorTheme.h"
#include "MRRibbonFontManager.h"
#include "MRUIStyle.h"
#include "ImGuiHelpers.h"
#include "MRMesh/MRColor.h"
#include "MRPch/MRWasm.h"
#include <imgui.h>
#include <imgui_internal.h>

namespace
{
constexpr int cNotificationNumberLimit = 10;

constexpr std::array< std::pair<const char*, ImU32>, int( MR::NotificationType::Count )> notificationParams
{
    std::pair<const char*, ImU32> { "\xef\x81\xaa", ::MR::Color( 217, 0, 0 ).getUInt32() },
    std::pair<const char*, ImU32> { "\xef\x81\xb1", ::MR::Color( 255, 146, 0 ).getUInt32() },
    std::pair<const char*, ImU32> { "\xef\x83\xb3", ::MR::Color( 39, 119, 214 ).getUInt32() }
};

}

namespace MR
{

void RibbonNotifier::pushNotification( const RibbonNotification& notification )
{
    if ( notifications_.size() == cNotificationNumberLimit )
        notifications_.erase( notifications_.end() - 1 );
    notifications_.insert( notifications_.begin(), NotificationWithTimer{ notification } );
    requestClosestRedraw_();
}

void RibbonNotifier::drawNotifications( float scaling )
{
    Vector2f currentPos = Vector2f( getViewerInstance().framebufferSize );
    const Vector2f padding = Vector2f( 0.0f, 20.0f * scaling );
    const float width = 337.0f * scaling;
    currentPos.x -= padding.y;

    int numInvalid = -1;
    for ( int i = 0; i < notifications_.size(); ++i )
    {
        currentPos -= padding;
        auto& [notification, timer] = notifications_[i];

        ImGui::SetNextWindowPos( currentPos, ImGuiCond_Always, ImVec2( 1.0f, 1.0f ) );
        ImGui::SetNextWindowSize( ImVec2( width, -1 ), ImGuiCond_Always );
        ImGuiWindowFlags flags =
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoMove;
        std::string name = "##notification" + std::to_string( i );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 4.0f * scaling );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 10.0f * scaling, 12.0f * scaling ) );
        ImGui::PushStyleVar( ImGuiStyleVar_CellPadding, { 0, 0 } );
        ImGui::PushStyleColor( ImGuiCol_WindowBg, MR::ColorTheme::getRibbonColor( MR::ColorTheme::RibbonColorsType::FrameBackground ).getUInt32() );

        if ( i + 1 == cNotificationNumberLimit )
            ImGui::SetNextWindowBgAlpha( 0.5f );
        ImGui::Begin( name.c_str(), nullptr, flags );
        const int columnCount = notification.onButtonClick ? 3 : 2;
        const float firstColumnWidth = 28.0f * scaling;
        auto& style = ImGui::GetStyle();
        const float buttonWidth = notification.onButtonClick ?
            ImGui::CalcTextSize( notification.buttonName.c_str() ).x + 2.0f * style.FramePadding.x + 2.0f * style.WindowPadding.x : 0;

        ImGui::BeginTable( "##NotificationTable", columnCount, ImGuiTableFlags_SizingFixedFit );

        ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, firstColumnWidth );
        ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, width - firstColumnWidth - buttonWidth );
        if ( notification.onButtonClick )
            ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, buttonWidth );

        ImGui::TableNextColumn();
        auto iconsFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Icons );
        if ( iconsFont )
        {
            iconsFont->Scale = 0.7f;
            ImGui::PushFont( iconsFont );
        }

        ImGui::PushStyleColor( ImGuiCol_Text, notificationParams[int(notification.type)].second );
        if ( notification.onButtonClick )
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + style.FramePadding.y * 0.5f );
        ImGui::Text( "%s", notificationParams[int(notification.type)].first );
        ImGui::PopStyleColor();

        if ( iconsFont )
        {
            iconsFont->Scale = 1.0f;
            ImGui::PopFont();
        }

        ImGui::TableNextColumn();

        if ( !notification.header.empty() )
        {
            auto boldFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::BigSemiBold );
            if ( boldFont )
                ImGui::PushFont( boldFont );
            
            ImGui::SetCursorPosX( 40.0f * scaling );
            if ( notification.onButtonClick )
                ImGui::SetCursorPosY( ImGui::GetCursorPosY() + style.FramePadding.y * 0.5f );
            ImGui::TextWrapped( "%s", notification.header.c_str() );

            if ( boldFont )
                ImGui::PopFont();
        }

        auto bigFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Big );
        if ( bigFont )
            ImGui::PushFont( bigFont );
        
        ImGui::SetCursorPosX( 40.0f * scaling );
        if ( notification.onButtonClick )
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + style.FramePadding.y * 0.5f );
        ImGui::TextWrapped( "%s", notification.text.c_str() );
        
        if ( bigFont )
            ImGui::PopFont();
        ImGui::SameLine();

        if ( notification.onButtonClick )
        {
            ImGui::TableNextColumn();
            if ( UI::buttonCommonSize( notification.buttonName.c_str() ) )
                notification.onButtonClick();
        }
        ImGui::EndTable();       

        auto window = ImGui::GetCurrentContext()->CurrentWindow;
        if ( !ImGui::IsWindowHovered() )
            timer += ImGui::GetIO().DeltaTime;
        
        if ( notification.type == NotificationType::Error || notification.type == NotificationType::Warning )
        {
            auto drawList = window->DrawList;
            drawList->PushClipRectFullScreen();
            drawList->AddRect( window->Rect().Min, window->Rect().Max, notificationParams[int( notification.type )].second, 4.0f * scaling, 0, 2.0f * scaling );
            drawList->PopClipRect();
        }

        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar( 4 );
        currentPos.y -= window->Size.y;
    }
    filterInvalid_( numInvalid );
}

void RibbonNotifier::filterInvalid_( int numInvalid )
{
    bool changed = false;
    for ( int i = int( notifications_.size() ) - 1; i >= 0; --i )
    {
        if ( notifications_[i].notification.lifeTimeSec - notifications_[i].timer <= 0.0f || i == numInvalid )
        {
            changed = true;
            notifications_.erase( notifications_.begin() + i );
        }
    }
    if ( changed )
    {
#ifndef __EMSCRIPTEN__
        requestedTime_ = Time::max();
#endif
        requestClosestRedraw_();
    }
}

void RibbonNotifier::requestClosestRedraw_()
{
    float minTimeReq = FLT_MAX;
    for ( const auto& notification : notifications_ )
    {
        float neededTime = notification.notification.lifeTimeSec - notification.timer;
        if ( neededTime < minTimeReq )
            minTimeReq = neededTime;
    }
    if ( minTimeReq == FLT_MAX )
        return;
#ifndef __EMSCRIPTEN__
    Time neededTime = std::chrono::system_clock::now() + std::chrono::milliseconds( std::llround( minTimeReq * 1000 ) + 100 );
    if ( requestedTime_ < neededTime )
        return;
    requestedTime_ = neededTime;
    asyncRequest_.request( requestedTime_, [&] ()
    {
        CommandLoop::appendCommand( [&] ()
        {
            getViewerInstance().incrementForceRedrawFrames();
            requestedTime_ = Time::max();
        } );
    } );
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( postEmptyEvent( $0, 2 ), int( minTimeReq * 1000 ) + 100 );
#pragma clang diagnostic pop
#endif
}

}