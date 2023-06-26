// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2018 Jérémie Dumas <jeremie.dumas@ens-lyon.org>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
////////////////////////////////////////////////////////////////////////////////
#include "ImGuiMenu.h"
#include "MRMeshViewer.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "imgui_fonts_droid_sans.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRVisualObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRColorTheme.h"
#include "MRCommandLoop.h"
#include <GLFW/glfw3.h>
#include "MRShortcutManager.h"
#include "MRMesh/MRTimer.h"
#include "ImGuiHelpers.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRChangeNameAction.h"
#include "MRMesh/MRChangeSceneAction.h"
////////////////////////////////////////////////////////////////////////////////
#include "MRPch/MRWasm.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRRibbonButtonDrawer.h"
#include "MRMesh/MRObjectLabel.h"

#include "MRMesh/MRChangeXfAction.h"
#include "MRMeshModifier.h"
#include "MRPch/MRSpdlog.h"
#include "MRProgressBar.h"
#include "MRFileDialog.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRObjectLoad.h>
#include <MRMesh/MRObject.h>
#include <MRMesh/MRBox.h>
#include "MRMesh/MRBitSet.h"
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>

#include "MRMesh/MRVoxelsLoad.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRVoxelsSave.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRImageSave.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRChangeSceneAction.h"
#include "MRMesh/MRChangeNameAction.h"
#include "MRMesh/MRHistoryStore.h"
#include "ImGuiHelpers.h"
#include "MRAppendHistory.h"
#include "MRMesh/MRCombinedHistoryAction.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRChangeLabelAction.h"
#include "MRMesh/MRMatrix3Decompose.h"

#include "MRMesh/MRChangeXfAction.h"
#include "MRMesh/MRSceneSettings.h"
#include "imgui_internal.h"
#include "MRRibbonConstants.h"
#include "MRRibbonFontManager.h"
#include "MRUIStyle.h"

#ifndef __EMSCRIPTEN__
#include <fmt/chrono.h>
#endif

namespace
{
// translation multiplier that limits its maximum value depending on object size
constexpr float cMaxTranslationMultiplier = 0xC00;
}

namespace MR
{

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cGLPrimitivesCounterNames =
{
    "Point Array Size",
    "Line Array Size",
    "Triangle Array Size",
    "Point Elements Number",
    "Line Elements Number",
    "Triangle Elements Number"
};

constexpr std::array<const char*, size_t( MR::Viewer::EventType::Count )> cEventCounterNames =
{
    "Mouse Down",
    "Mouse Up",
    "Mouse Move",
    "Mouse Scroll",
    "Key Down",
    "Key Up",
    "Key Repeat",
    "Char Pressed"
};

bool objectHasRealChildren( const MR::Object& object )
{
    bool res = false;
    for ( const auto& child : object.children() )
    {
        if ( !child->isAncillary() )
            res = true;
        else
            res = objectHasRealChildren( *child );

        if ( res )
            break;
    }
    return res;
}

void selectRecursive( Object& obj )
{
    obj.select( true );
    for ( auto& child : obj.children() )
        if ( child )
            selectRecursive( *child );
}

void ImGuiMenu::init( MR::Viewer* _viewer )
{
    ViewerPlugin::init( _viewer );
    // Setup ImGui binding
    if ( _viewer )
    {
        IMGUI_CHECKVERSION();
        if ( !context_ )
        {
            // Single global context by default, but can be overridden by the user
            static ImGuiContext* __global_context = ImGui::CreateContext();
            context_ = __global_context;
        }
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.FrameRounding = 5.0f;
        reload_font();

        connect( _viewer, 0, boost::signals2::connect_position::at_front );
    }

    setupShortcuts_();
}

void ImGuiMenu::initBackend()
{
    if ( !viewer || !viewer->isGLInitialized() )
        return;

#ifdef __EMSCRIPTEN__
    const char* glsl_version = "#version 300 es";
#else
    const char* glsl_version = "#version 150";
#endif
    rescaleStyle_();
    ImGui_ImplGlfw_InitForOpenGL( viewer->window, false );
    ImGui_ImplOpenGL3_Init( glsl_version );
}

std::filesystem::path ImGuiMenu::getMenuFontPath() const
{
#ifdef _WIN32
    // get windows font
    wchar_t winDir[MAX_PATH];
    GetWindowsDirectoryW( winDir, MAX_PATH );
    std::filesystem::path winDirPath( winDir );
    winDirPath /= "Fonts";
    winDirPath /= "Consola.ttf";
    return winDirPath;
#else
    return {};
#endif
}

const ImVec4 undefined = ImVec4( 0.5f, 0.5f, 0.5f, 0.5f );



// at least one of selected is true - first,
// all selected are true - second
std::pair<bool, bool> getRealValue( const std::vector<std::shared_ptr<MR::VisualObject>>& selected,
                                    unsigned type, MR::ViewportMask viewportId, bool inverseInput = false )
{
    bool atLeastOneTrue = false;
    bool allTrue = true;
    for ( const auto& data : selected )
    {
        bool isThisTrue = data && data->getVisualizeProperty( type, viewportId );
        if ( inverseInput )
            isThisTrue = !isThisTrue;
        atLeastOneTrue = atLeastOneTrue || isThisTrue;
        allTrue = allTrue && isThisTrue;
    }
    allTrue = allTrue && atLeastOneTrue;
    return { atLeastOneTrue,allTrue };
}

void ImGuiMenu::addMenuFontRanges_( ImFontGlyphRangesBuilder& builder ) const
{
    builder.AddRanges( ImGui::GetIO().Fonts->GetGlyphRangesCyrillic() );
#ifndef __EMSCRIPTEN__
    builder.AddRanges( ImGui::GetIO().Fonts->GetGlyphRangesChineseSimplifiedCommon() );
#endif
}

void ImGuiMenu::load_font(int font_size)
{
#ifdef _WIN32
    if ( viewer->isGLInitialized() )
    {
        ImGuiIO& io = ImGui::GetIO();

        auto fontPath = getMenuFontPath();

        ImVector<ImWchar> ranges;
        ImFontGlyphRangesBuilder builder;
        addMenuFontRanges_( builder );
        builder.BuildRanges( &ranges );

        io.Fonts->AddFontFromFileTTF(
            utf8string( fontPath ).c_str(), font_size * menu_scaling(),
            nullptr, ranges.Data );
        io.Fonts->Build();
    }
    else
    {
        ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
      droid_sans_compressed_size, font_size * hidpi_scaling_ );
        ImGui::GetIO().Fonts[0].Build();
    }
#else
    ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF( droid_sans_compressed_data,
      droid_sans_compressed_size, font_size * hidpi_scaling_);
    //TODO: expand for non-Windows systems
#endif
  
}

void ImGuiMenu::reload_font(int font_size)
{
  hidpi_scaling_ = hidpi_scaling();
  pixel_ratio_ = pixel_ratio();
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  load_font(font_size);

  io.FontGlobalScale = 1.0f / pixel_ratio_;
}

void ImGuiMenu::shutdown()
{
    // Cleanup
    if ( viewer && viewer->isGLInitialized() )
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
    }

    disconnect();
    // User is responsible for destroying context if a custom context is given
    // ImGui::DestroyContext(*context_);
}

void ImGuiMenu::preDraw_()
{
  if( pollEventsInPreDraw )
  {
     glfwPollEvents();
  }

  if ( viewer->isGLInitialized() )
  {
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
  }
  else
  {
      // needed for dear ImGui
      ImGui::GetIO().DisplaySize = ImVec2( float( viewer->window_width ), float( viewer->window_height ) );
  }
  auto& style = ImGui::GetStyle();
  if ( storedError_.empty() )
      style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 0.0f, 0.0f, 0.0f, 0.8f );
  else
      style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4( 1.0f, 0.2f, 0.2f, 0.5f );
  ImGui::NewFrame();
}

void ImGuiMenu::postDraw_()
{
  draw_menu();
  if ( viewer->isGLInitialized() )
  {
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
  }
  else
  {
      ImGui::EndFrame();
  }
}

void ImGuiMenu::postResize_( int width, int height )
{
    if ( context_ )
    {
        ImGui::GetIO().DisplaySize.x = float( width );
        ImGui::GetIO().DisplaySize.y = float( height );
    }
}

void ImGuiMenu::postRescale_( float /*x*/, float /*y*/)
{
    reload_font();
    rescaleStyle_();
    ImGui_ImplOpenGL3_DestroyDeviceObjects();
}

bool ImGuiMenu::spaceMouseMove_( const Vector3f& /*translate*/, const Vector3f& /*rotate*/ )
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

bool ImGuiMenu::spaceMouseDown_( int /*key*/ )
{
    return ImGui::IsPopupOpen( "", ImGuiPopupFlags_AnyPopup );
}

void ImGuiMenu::rescaleStyle_()
{
    CommandLoop::appendCommand( [&] ()
    {
        ColorTheme::resetImGuiStyle(); // apply scaling inside
    } );
}

// Mouse IO
bool ImGuiMenu::onMouseDown_( Viewer::MouseButton button, int modifier)
{
    ImGui_ImplGlfw_MouseButtonCallback( viewer->window, int( button ), GLFW_PRESS, modifier );
    capturedMouse_ = ImGui::GetIO().WantCaptureMouse;
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiMenu::onMouseUp_( Viewer::MouseButton button, int modifier )
{
    ImGui_ImplGlfw_MouseButtonCallback( viewer->window, int( button ), GLFW_RELEASE, modifier );
    return capturedMouse_;
}

bool ImGuiMenu::onMouseMove_(int mouse_x, int mouse_y )
{
    ImGui_ImplGlfw_CursorPosCallback( viewer->window, double( mouse_x ), double( mouse_y ) );
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiMenu::onMouseScroll_(float delta_y)
{
    ImGui_ImplGlfw_ScrollCallback( viewer->window, 0.f, delta_y );
    // do extra frames to prevent imgui calculations ping
    if ( ImGui::GetIO().WantCaptureMouse )
    {
        viewer->incrementForceRedrawFrames( viewer->forceRedrawMinimumIncrementAfterEvents, viewer->swapOnLastPostEventsRedraw );
        return true;
    }
    return false;
}

void ImGuiMenu::cursorEntrance_( [[maybe_unused]] bool entered )
{
#ifdef __EMSCRIPTEN__
    if ( entered )
        ImGui::GetIO().ConfigFlags &= (~ImGuiConfigFlags_NoMouse);
    else
    {
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
        EM_ASM( postEmptyEvent( 100, 2 ) );
    }
#endif
}

// Keyboard IO
bool ImGuiMenu::onCharPressed_( unsigned  key, int /*modifiers*/ )
{
    ImGui_ImplGlfw_CharCallback( nullptr, key );
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiMenu::onKeyDown_( int key, int modifiers )
{
    ImGui_ImplGlfw_KeyCallback( viewer->window, key, 0, GLFW_PRESS, modifiers );
    if ( ImGui::GetIO().WantCaptureKeyboard )
        return true;

    if ( shortcutManager_ )
        return shortcutManager_->processShortcut( { key,modifiers } );

    return false;
}

bool ImGuiMenu::onKeyUp_( int key, int modifiers )
{
    ImGui_ImplGlfw_KeyCallback( viewer->window, key, 0, GLFW_RELEASE, modifiers );
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool ImGuiMenu::onKeyRepeat_( int key, int modifiers )
{
    ImGui_ImplGlfw_KeyCallback( viewer->window, key, 0, GLFW_REPEAT, modifiers );
    if ( ImGui::GetIO().WantCaptureKeyboard )
        return true;

    if ( shortcutManager_ )
        return shortcutManager_->processShortcut( { key, modifiers } );

    return false;
}

// Draw menu
void ImGuiMenu::draw_menu()
{
  // Text labels
  draw_labels_window();

  // Viewer settings
  if (callback_draw_viewer_window) { callback_draw_viewer_window(); }
  else { draw_viewer_window(); }

  // Other windows
  if (callback_draw_custom_window) { callback_draw_custom_window(); }
  else { draw_custom_window(); }
}

void ImGuiMenu::draw_viewer_window()
{
  float menu_width = 180.f * menu_scaling();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(0.0f, 0.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(menu_width, -1.0f), ImVec2(menu_width, -1.0f));
  ImGui::Begin(
      "Viewer", nullptr,
      ImGuiWindowFlags_NoSavedSettings
      | ImGuiWindowFlags_AlwaysAutoResize
  );
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.4f);
  if (callback_draw_viewer_menu) { callback_draw_viewer_menu(); }
  ImGui::PopItemWidth();
  ImGui::End();
}

void ImGuiMenu::draw_labels_window()
{
  // Text labels
  ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
  bool visible = true;
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0,0,0,0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::Begin("ViewerLabels", &visible,
      ImGuiWindowFlags_NoTitleBar
      | ImGuiWindowFlags_NoResize
      | ImGuiWindowFlags_NoMove
      | ImGuiWindowFlags_NoScrollbar
      | ImGuiWindowFlags_NoScrollWithMouse
      | ImGuiWindowFlags_NoCollapse
      | ImGuiWindowFlags_NoSavedSettings
      | ImGuiWindowFlags_NoInputs);
  for ( const auto& data : getAllObjectsInTree<const VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any ) )
  {
      draw_labels( *data );
  }

  // separate block for basis axes
  for ( const auto& viewport : viewer->viewport_list )
  {
      if ( !viewer->globalBasisAxes->isVisible( viewport.id ) )
          continue;
  }
  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar();
}

void ImGuiMenu::draw_labels( const VisualObject& obj )
{
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
    const auto& labels = obj.getLabels();

    for ( const auto& viewport : viewer->viewport_list )
    {
        if ( !obj.globalVisibilty( viewport.id ) )
            continue;
        AffineXf3f xf = obj.worldXf();
        bool clip = obj.getVisualizeProperty( VisualizeMaskType::CropLabelsByViewportRect, viewport.id );
        if ( obj.getVisualizeProperty( VisualizeMaskType::Labels, viewport.id ) )
            for ( int i = 0; i < labels.size(); ++i )
                draw_text(
                    viewport,
                    xf( labels[i].position ),
                    Vector3f( 0.0f, 0.0f, 0.0f ),
                    labels[i].text,
                    obj.getLabelsColor(),
                    clip );
        if ( obj.getVisualizeProperty( VisualizeMaskType::Name, viewport.id ) )
            draw_text(
                viewport,
                xf( obj.getBoundingBox().center() ),
                Vector3f( 0.0f, 0.0f, 0.0f ),
                obj.name(), 
                 obj.getLabelsColor(),
                clip );
    }
MR_SUPPRESS_WARNING_POP
}

void ImGuiMenu::draw_text(
    const Viewport& viewport,
    const Vector3f& posOriginal,
    const Vector3f& normal,
    const std::string& text,
    const Color& color,
    bool clipByViewport )
{
  Vector3f pos = posOriginal;
  pos += normal * 0.005f * viewport.getParameters().objectScale;
  const auto& viewportRect = viewport.getViewportRect();
  Vector3f coord = viewport.clipSpaceToViewportSpace( viewport.projectToClipSpace( pos ) );
  auto viewerCoord = viewer->viewportToScreen( coord, viewport.id );

  // Draw text labels slightly bigger than normal text
  ImDrawList* drawList = ImGui::GetWindowDrawList();
  ImVec4 clipRect( viewportRect.min.x, 
                   viewer->window_height - ( viewportRect.min.y + height( viewportRect ) ),
                   viewportRect.min.x + width( viewportRect ),
                   viewer->window_height - viewportRect.min.y );
  drawList->AddText( ImGui::GetFont(), ImGui::GetFontSize() * 1.2f,
                     ImVec2( viewerCoord.x / pixel_ratio_, viewerCoord.y / pixel_ratio_ ),
                     color.getUInt32(),
                     &text[0], &text[0] + text.size(), 0.0f,
                     clipByViewport ? &clipRect : nullptr );
}

float ImGuiMenu::pixel_ratio()
{
#if defined(__APPLE__)
    return 1.0f;
#endif
    // Computes pixel ratio for hidpi devices
    int buf_size[2];
    int win_size[2];
    GLFWwindow* window = glfwGetCurrentContext();
    if ( window )
    {
        glfwGetFramebufferSize( window, &buf_size[0], &buf_size[1] );
        glfwGetWindowSize( window, &win_size[0], &win_size[1] );
        return ( float )buf_size[0] / ( float )win_size[0];
    }
    return 1.0f;
}

float ImGuiMenu::hidpi_scaling()
{
#if defined(__APPLE__)
    return 1.0f;
#endif
    // Computes scaling factor for hidpi devices
    float xscale{ 1.0f }, yscale{ 1.0f };
#ifndef __EMSCRIPTEN__
    GLFWwindow* window = glfwGetCurrentContext();
    if ( window )
    {
        glfwGetWindowContentScale( window, &xscale, &yscale );
    }
#endif
    return 0.5f * ( xscale + yscale );
}

float ImGuiMenu::menu_scaling() const
{
#ifdef __EMSCRIPTEN__
    return float( emscripten_get_device_pixel_ratio() );
#else
    return hidpi_scaling_ / pixel_ratio_;
#endif
}

ImGuiContext* ImGuiMenu::getCurrentContext() const
{
    return context_;
}

void ImGuiMenu::draw_helpers()
{
    if ( showShortcuts_ )
    {
        drawShortcutsWindow_();
    }

    if ( showStatistics_ )
    {
        const auto style = ImGui::GetStyle();
        const float fpsWindowWidth = 300 * menu_scaling();
        int numLines = 5 + int( Viewer::EventType::Count ) + int( Viewer::GLPrimitivesType::Count ); // 5 - for: GL buffer size, prev frame time, swapped frames, total frames, fps;
        // TextHeight +1 for button, ItemSpacing +2 for separators
        const float fpsWindowHeight = ( style.WindowPadding.y * 2 +
                                        ImGui::GetTextLineHeight() * ( numLines + 2 ) +
                                        style.ItemSpacing.y * ( numLines + 3 ) +
                                        style.FramePadding.y * 4 );
        const float posX = Viewer::instanceRef().window_width - fpsWindowWidth;
        const float posY = Viewer::instanceRef().window_height - fpsWindowHeight;
        ImGui::SetNextWindowPos( ImVec2( posX, posY ), ImGuiCond_Appearing );
        ImGui::SetNextWindowSize( ImVec2( fpsWindowWidth, fpsWindowHeight ) );
        ImGui::Begin( "##FPS", nullptr, ImGuiWindowFlags_AlwaysAutoResize | //ImGuiWindowFlags_NoInputs | 
                      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing );
        for ( int i = 0; i<int( Viewer::GLPrimitivesType::Count ); ++i )
            ImGui::Text( "%s: %zu", cGLPrimitivesCounterNames[i], viewer->getLastFrameGLPrimitivesCount( Viewer::GLPrimitivesType( i ) ) );
        ImGui::Separator();
        for ( int i = 0; i<int( Viewer::EventType::Count ); ++i )
            ImGui::Text( "%s: %zu", cEventCounterNames[i], viewer->getEventsCount( Viewer::EventType( i ) ) );
        ImGui::Separator();
        auto glBufferSizeStr = bytesString( viewer->getStaticGLBufferSize() );
        ImGui::Text( "GL memory buffer: %s", glBufferSizeStr.c_str() );
        auto prevFrameTime = viewer->getPrevFrameDrawTimeMillisec();
        if ( prevFrameTime > frameTimeMillisecThreshold_ )
            ImGui::TextColored( ImVec4( 1.0f, 0.3f, 0.3f, 1.0f ), "Previous frame time: %.1f ms", prevFrameTime );
        else
            ImGui::Text( "Previous frame time: %.1f ms", prevFrameTime );
        ImGui::Text( "Total frames: %zu", viewer->getTotalFrames() );
        ImGui::Text( "Swapped frames: %zu", viewer->getSwappedFrames() );
        ImGui::Text( "FPS: %zu", viewer->getFPS() );

        if ( UI::buttonCommonSize( "Reset", Vector2f( -1, 0 ) ) )
        {
            viewer->resetAllCounters();
        }
        if ( UI::buttonCommonSize( "Print time to log", Vector2f( -1, 0 ) ) )
        {
            printTimingTreeAndStop();
        }
        ImGui::End();
    }

    if ( showRenameModal_ )
    {
        showRenameModal_ = false;
        ImGui::OpenPopup( "Rename object" );
        popUpRenameBuffer_ = renameBuffer_;
    }

    const auto menuScaling = menu_scaling();
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { 2.0f * cDefaultInnerSpacing * menuScaling, cDefaultInnerSpacing * menuScaling } );
    
    ImVec2 windowSize( cModalWindowWidth * menuScaling, 0.0f );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );

    if ( ImGui::BeginModalNoAnimation( "Rename object", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        ImGui::SetCursorPosX( ( windowSize.x - ImGui::CalcTextSize( "Rename Object" ).x ) * 0.5f );
        ImGui::Text( "Rename Object" );

        if ( headerFont )
            ImGui::PopFont();

        auto obj = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected ).front();
        if ( !obj )
        {
            ImGui::CloseCurrentPopup();
        }
        if ( ImGui::IsWindowAppearing() )
            ImGui::SetKeyboardFocusHere();

        const auto& style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cInputPadding * menuScaling } );
        ImGui::SetNextItemWidth( windowSize.x - 2 * style.WindowPadding.x - style.ItemInnerSpacing.x - ImGui::CalcTextSize( "Name" ).x );
        ImGui::InputText( "Name", popUpRenameBuffer_, ImGuiInputTextFlags_AutoSelectAll );
        ImGui::PopStyleVar();

        const float btnWidth = cModalButtonWidth * menuScaling;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        if ( UI::button( "Ok", Vector2f( btnWidth, 0 ), ImGuiKey_Enter ) )
        {
            AppendHistory( std::make_shared<ChangeNameAction>( "Rename object", obj ) );
            obj->setName( popUpRenameBuffer_ );
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX( windowSize.x - btnWidth - style.WindowPadding.x );
        if ( UI::button( "Cancel", Vector2f( btnWidth, 0 ), ImGuiKey_Escape ) )
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleVar();
        if ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) )
        {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 3 );

    ImGui::PushStyleColor( ImGuiCol_ModalWindowDimBg, ImVec4( 1, 0.125f, 0.125f, ImGui::GetStyle().Colors[ImGuiCol_ModalWindowDimBg].w ) );

    if ( !storedError_.empty() && !ImGui::IsPopupOpen( " Error##modal" ) )
    {        
        ImGui::OpenPopup( " Error##modal" );
    }

    const ImVec2 errorWindowSize{ MR::cModalWindowWidth * menuScaling, -1 };
    ImGui::SetNextWindowSize( errorWindowSize, ImGuiCond_Always );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
    if ( ImGui::BeginModalNoAnimation( " Error##modal", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        auto headerFont = RibbonFontManager::getFontByTypeStatic( RibbonFontManager::FontType::Headline );
        if ( headerFont )
            ImGui::PushFont( headerFont );

        const auto headerWidth = ImGui::CalcTextSize( "Error" ).x;

        ImGui::SetCursorPosX( ( errorWindowSize.x - headerWidth ) * 0.5f );
        ImGui::Text( "Error" );

        if ( headerFont )
            ImGui::PopFont();

        const float textWidth = ImGui::CalcTextSize( storedError_.c_str() ).x;

        if ( textWidth < errorWindowSize.x )
        {
            ImGui::SetCursorPosX( ( errorWindowSize.x - textWidth ) * 0.5f );
            ImGui::Text( "%s", storedError_.c_str() );
        }
        else
        {
            ImGui::TextWrapped( "%s", storedError_.c_str() );
        }
        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );
        if ( UI::button( "Okay", Vector2f( -1, 0 ) ) || ImGui::IsKeyPressed( ImGuiKey_Enter ) ||
           ( ImGui::IsMouseClicked( 0 ) && !( ImGui::IsAnyItemHovered() || ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) ) ) )
        {
            storedError_.clear();            
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 2 );
    ImGui::PopStyleColor();
}

void ImGuiMenu::setDrawTimeMillisecThreshold( long long maxGoodTimeMillisec )
{
    frameTimeMillisecThreshold_ = maxGoodTimeMillisec;
}

void ImGuiMenu::showErrorModal( const std::string& error )
{
    showRenameModal_ = false;
    ImGui::CloseCurrentPopup();
    storedError_ = error;
    // this is needed to correctly resize error window
    getViewerInstance().incrementForceRedrawFrames( 2, true );
}

void ImGuiMenu::setupShortcuts_()
{
}

void ImGuiMenu::draw_scene_list()
{
    const auto allObj = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selectable );
    auto selectedObjs = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    // Define next window position + size
    ImGui::SetNextWindowPos( ImVec2( 180 * menu_scaling(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 230 * menu_scaling(), 300 * menu_scaling() ), ImGuiCond_FirstUseEver );
    ImGui::Begin(
        "Scene", nullptr
    );
    draw_scene_list_content( selectedObjs, allObj );

    sceneWindowPos_ = ImGui::GetWindowPos();
    sceneWindowSize_ = ImGui::GetWindowSize();
    ImGui::End();

    draw_selection_properties( selectedObjs );
}

void ImGuiMenu::draw_scene_list_content( const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all )
{
    // mesh with index 0 is Ancillary, and cannot be removed
    // it can be cleaned but it is inconsistent, so this mesh is untouchable
    ImGui::BeginChild( "Meshes", ImVec2( -1, -1 ), true );
    updateSceneWindowScrollIfNeeded_();
    auto children = SceneRoot::get().children();
    for ( const auto& child : children )
        draw_object_recurse_( *child, selected, all );
    makeDragDropTarget_( SceneRoot::get(), false, true, "" );
    ImGui::EndChild();
    sceneOpenCommands_.clear();

    reorderSceneIfNeeded_();
}

void ImGuiMenu::draw_selection_properties( std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    if ( !selectedObjs.empty() )
    {
        // Define next window position + size
        ImGui::SetNextWindowPos( ImVec2( sceneWindowPos_.x, sceneWindowPos_.y + sceneWindowSize_.y ) );
        ImGui::SetNextWindowSize( ImVec2( sceneWindowSize_.x, -1 ) );
        ImGui::Begin(
            "Selection Properties", nullptr,
            ImGuiWindowFlags_NoMove
        );
        draw_selection_properties_content( selectedObjs );
        ImGui::End();
    }
}

void ImGuiMenu::draw_selection_properties_content( std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    drawSelectionInformation_();

    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    bool allHaveVisualisation = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        if ( !obj )
            return false;
        auto objMesh = obj->asType<ObjectMesh>();
        if ( objMesh && objMesh->mesh() )
            return true;
        auto objPoints = obj->asType<ObjectPoints>();
        if ( objPoints && objPoints->pointCloud() )
            return true;
        auto objLines = obj->asType<ObjectLines>();
        if ( objLines && objLines->polyline() )
            return true;
        return false;
    } );

    drawGeneralOptions_( selectedObjs );

    if ( allHaveVisualisation && drawCollapsingHeader_( "Draw Options" ) )
    {
        drawDrawOptionsCheckboxes_( selectedVisualObjs );
        drawDrawOptionsColors_( selectedVisualObjs );
        drawAdvancedOptions_( selectedVisualObjs );
    }

    draw_custom_selection_properties( selectedObjs );

    drawRemoveButton_( selectedObjs );


    drawTransform_();
}

void ImGuiMenu::makeDragDropSource_( const std::vector<std::shared_ptr<Object>>& payload )
{
    if ( !allowSceneReorder_ || payload.empty() )
        return;
    if ( ImGui::BeginDragDropSource( ImGuiDragDropFlags_AcceptNoDrawDefaultRect ) )
    {
        dragTrigger_ = true;

        std::vector<Object*> vectorObjPtr;
        for ( auto& ptr : payload )
            vectorObjPtr.push_back( ptr.get() );

        ImGui::SetDragDropPayload( "_TREENODE", vectorObjPtr.data(), sizeof( Object* ) * vectorObjPtr.size() );
        std::string allNames;
        allNames = payload[0]->name();
        for ( int i = 1; i < payload.size(); ++i )
            allNames += "\n" + payload[i]->name();
        ImGui::Text( "%s", allNames.c_str() );
        ImGui::EndDragDropSource();
    }

}

void ImGuiMenu::makeDragDropTarget_( Object& target, bool before, bool betweenLine, const std::string& uniqueStr )
{
    if ( !allowSceneReorder_ )
        return;
    const ImGuiPayload* payloadCheck = ImGui::GetDragDropPayload();
    ImVec2 curPos{};
    bool lineDrawed = false;
    if ( payloadCheck && std::string_view( payloadCheck->DataType ) == "_TREENODE" && betweenLine )
    {
        lineDrawed = true;
        curPos = ImGui::GetCursorPos();
        auto width = ImGui::GetContentRegionAvail().x;
        ImGui::ColorButton( ( "##InternalDragDropArea" + uniqueStr ).c_str(),
            ImVec4( 0, 0, 0, 0 ),
            0, ImVec2( width, 4 * menu_scaling() ) );
    }
    if ( ImGui::BeginDragDropTarget() )
    {
        if ( lineDrawed )
        {
            ImGui::SetCursorPos( curPos );
            auto width = ImGui::GetContentRegionAvail().x;
            ImGui::ColorButton( ( "##ColoredInternalDragDropArea" + uniqueStr ).c_str(),
                ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered],
                0, ImVec2( width, 4 * menu_scaling() ) );
        }
        if ( const ImGuiPayload* payload = ImGui::AcceptDragDropPayload( "_TREENODE" ) )
        {
            assert( payload->DataSize % sizeof( Object* ) == 0 );
            Object** objArray = ( Object** )payload->Data;
            const int size = payload->DataSize / sizeof( Object* );
            std::vector<Object*> vectorObj( size );
            for ( int i = 0; i < size; ++i )
                vectorObj[i] = objArray[i];
            sceneReorderCommand_ = { vectorObj, &target, before };
        }
        ImGui::EndDragDropTarget();
    }
}

void ImGuiMenu::draw_object_recurse_( Object& object, const std::vector<std::shared_ptr<Object>>& selected, const std::vector<std::shared_ptr<Object>>& all )
{
    std::string uniqueStr = std::to_string( intptr_t( &object ) );
    const bool isObjSelectable = !object.isAncillary();

    // has selectable children
    bool hasRealChildren = objectHasRealChildren( object );
    bool isOpen{ false };
    if ( ( hasRealChildren || isObjSelectable ) )
    {
        makeDragDropTarget_( object, true, true, uniqueStr );
        {
            // Visibility checkbox
            bool isVisible = object.isVisible( viewer->viewport().id );
            auto ctx = ImGui::GetCurrentContext();
            assert( ctx );
            auto window = ctx->CurrentWindow;
            assert( window );
            auto diff = ImGui::GetStyle().FramePadding.y - cCheckboxPadding * menu_scaling();
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + diff );
            if ( UI::checkbox( ( "##VisibilityCheckbox" + uniqueStr ).c_str(), &isVisible ) )
            {
                object.setVisible( isVisible, viewer->viewport().id );
                if ( deselectNewHiddenObjects_ && !object.isVisible( viewer->getPresentViewports() ) )
                    object.select( false );
            }
            window->DC.CursorPosPrevLine.y -= diff;
            ImGui::SameLine();
        }
        {
            // custom prefix
            drawCustomObjectPrefixInScene_( object );
        }

        const bool isSelected = object.isSelected();

        auto openCommandIt = sceneOpenCommands_.find( &object );
        if ( openCommandIt != sceneOpenCommands_.end() )
            ImGui::SetNextItemOpen( openCommandIt->second );

        if ( !isSelected )
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0, 0, 0, 0 ) );
        else
        {
            ImGui::PushStyleColor( ImGuiCol_Header, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectFrame ).getUInt32() );
            ImGui::PushStyleColor( ImGuiCol_Text, ColorTheme::getRibbonColor( ColorTheme::RibbonColorsType::SelectedObjectText ).getUInt32() );
        }

        ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );

        isOpen = drawCollapsingHeader_( ( object.name() + "##" + uniqueStr ).c_str(),
                                    ( hasRealChildren ? ImGuiTreeNodeFlags_DefaultOpen : 0 ) |
                                    ImGuiTreeNodeFlags_OpenOnArrow |
                                    ImGuiTreeNodeFlags_SpanAvailWidth |
                                    ImGuiTreeNodeFlags_Framed |
                                    ( isSelected ? ImGuiTreeNodeFlags_Selected : 0 ) );

        ImGui::PopStyleColor( isSelected ? 2 : 1 );
        ImGui::PopStyleVar();

        makeDragDropSource_( selected );
        makeDragDropTarget_( object, false, false, "0" );

        if ( isObjSelectable && ImGui::IsItemHovered() )
        {
            bool pressed = !isSelected && ( ImGui::IsMouseClicked( 0 ) || ImGui::IsMouseClicked( 1 ) );
            bool released = isSelected && !dragTrigger_ && !clickTrigger_ && ImGui::IsMouseReleased( 0 );

            if ( pressed )
                clickTrigger_ = true;
            if ( isSelected && clickTrigger_ && ImGui::IsMouseReleased( 0 ) )
                clickTrigger_ = false;

            if ( pressed || released )
            {

                auto newSelection = getPreSelection_( &object, ImGui::GetIO().KeyShift, ImGui::GetIO().KeyCtrl, selected, all );
                if ( ImGui::GetIO().KeyCtrl )
                {
                    for ( auto& sel : newSelection )
                    {
                        const bool select = ImGui::GetIO().KeyShift || !sel->isSelected();
                        sel->select( select );
                        if ( showNewSelectedObjects_ && select )
                            sel->setGlobalVisibilty( true );
                    }
                }
                else
                {
                    for ( const auto& data : selected )
                    {
                        auto inNewSelList = std::find( newSelection.begin(), newSelection.end(), data.get() );
                        if ( inNewSelList == newSelection.end() )
                            data->select( false );
                    }
                    for ( auto& sel : newSelection )
                    {
                        sel->select( true );
                        if ( showNewSelectedObjects_ )
                            sel->setGlobalVisibilty( true );
                    }
                }
            }

        }


        if ( isSelected )
            drawSceneContextMenu_( selected );
    }
    if ( isOpen )
    {
        draw_custom_tree_object_properties( object );
        bool infoOpen = false;
        auto lines = object.getInfoLines();
        if ( hasRealChildren && !lines.empty() )
        {
            auto infoId = std::string( "Info: ##" ) + uniqueStr;
            infoOpen = drawCollapsingHeader_( infoId.c_str(), ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed );
        }

        if ( infoOpen || !hasRealChildren )
        {
            auto itemSpacing = ImGui::GetStyle().ItemSpacing;
            auto framePadding = ImGui::GetStyle().FramePadding;
            auto scaling = menu_scaling();
            framePadding.y = 2.0f * scaling;
            itemSpacing.y = 2.0f * scaling;
            ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( 0, 0, 0, 0 ) );
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, framePadding );
            ImGui::PushStyleVar( ImGuiStyleVar_FrameBorderSize, 0.0f );
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, itemSpacing );
            ImGui::PushStyleVar( ImGuiStyleVar_IndentSpacing, cItemInfoIndent * scaling );
            ImGui::Indent();

            for ( const auto& str : lines )
            {
                ImGui::TreeNodeEx( str.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_Framed );
                ImGui::TreePop();
            }

            ImGui::Unindent();
            ImGui::PopStyleVar( 4 );
            ImGui::PopStyleColor();
        }

        if ( hasRealChildren )
        {
            auto children = object.children();
            ImGui::Indent();
            for ( const auto& child : children )
            {
                draw_object_recurse_( *child, selected, all );
            }
            makeDragDropTarget_( object, false, true, "0" );
            ImGui::Unindent();
        }
    }
}

float ImGuiMenu::drawSelectionInformation_()
{
    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );

    auto& style = ImGui::GetStyle();

    float resultHeight = ImGui::GetTextLineHeight() + style.FramePadding.y * 2 + style.ItemSpacing.y;
    if ( drawCollapsingHeader_( "Information", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_ScrollbarSize, 12.0f );
        int fieldCount = 6;

        size_t totalPoints = 0;
        size_t totalSelectedPoints = 0;
        for ( auto pObj : getAllObjectsInTree<ObjectPoints>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            totalPoints += pObj->numValidPoints();
            totalSelectedPoints += pObj->numSelectedPoints();
        }
        if ( totalPoints )
            ++fieldCount;

        selectionBbox_ = Box3f{};
        selectionWorldBox_ = {};
        for ( auto pObj : selectedVisualObjs )
        {
            selectionBbox_.include( pObj->getBoundingBox() );
            selectionWorldBox_.include( pObj->getWorldBox() );
        }
        
        Vector3f bsize;
        Vector3f wbsize;
        std::string bsizeStr;
        std::string wbsizeStr;        

        if ( selectionWorldBox_.valid() )
        {
            bsize = selectionBbox_.size();
            bsizeStr = fmt::format( "{:.3e} {:.3e} {:.3e}", bsize.x, bsize.y, bsize.z );
            wbsize = selectionWorldBox_.size();
            wbsizeStr = fmt::format( "{:.3e} {:.3e} {:.3e}", wbsize.x, wbsize.y, wbsize.z );
            if ( bsizeStr != wbsizeStr )
                ++fieldCount;
        }            

        const float smallItemSpacingY = 0.25f * cDefaultItemSpacing * menu_scaling();
        const float infoHeight = ImGui::GetTextLineHeight() * fieldCount +
            style.FramePadding.y * fieldCount * 2 +
            style.ItemSpacing.y + 
            smallItemSpacingY * ( fieldCount + 1 );
        resultHeight += infoHeight + style.ItemSpacing.y;

        ImGui::BeginChild( "SceneInformation", ImVec2( 0, infoHeight ), false, ImGuiWindowFlags_HorizontalScrollbar );
        // Compute total faces/verts in selected objects
        size_t totalFaces = 0;
        size_t totalSelectedFaces = 0;
        size_t totalVerts = 0;
        for ( auto pObj : getAllObjectsInTree<ObjectMesh>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            if ( auto mesh = pObj->mesh() )
            {
                totalFaces += mesh->topology.numValidFaces();
                totalSelectedFaces += pObj->numSelectedFaces();
                totalVerts += mesh->topology.numValidVerts();
            }
        }
        for ( auto pObj : getAllObjectsInTree<ObjectLines>( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            if ( auto polyline = pObj->polyline() )
            {
                totalVerts += polyline->topology.numValidVerts();
            }
        }        

        auto drawPrimitivesInfo = [this] ( std::string title, size_t value, size_t selected = 0 )
        {
            if ( value )
            {
                std::string valueStr;
                std::string labelStr;
                if ( selected )
                {
                    valueStr = std::to_string( selected ) + " / ";
                    labelStr = "Selected / ";
                }
                valueStr += std::to_string( value );
                labelStr += title;

                UI::inputTextCenteredReadOnly( labelStr.c_str(), valueStr, getSceneInfoItemWidth_( 3 ) * 2 + ImGui::GetStyle().ItemInnerSpacing.x * menu_scaling() );
            }
        };

        if ( selectedVisualObjs.size() > 1 )
        {
            drawPrimitivesInfo( "Objects", selectedVisualObjs.size() );
        }
        else if ( auto pObj = getDepthFirstObject( &SceneRoot::get(), ObjectSelectivityType::Selected ) )
        {
            auto lastRenameObj = lastRenameObj_.lock();
            if ( lastRenameObj != pObj )
            {
                renameBuffer_ = pObj->name();
                lastRenameObj_ = pObj;
            }
            if ( !UI::inputTextCentered( "Object Name", renameBuffer_, getSceneInfoItemWidth_(), ImGuiInputTextFlags_AutoSelectAll ) )
            {
                if ( renameBuffer_ == pObj->name() )
                {
                    // clear the pointer to reload the name on next frame (if it was changed from outside)
                    lastRenameObj_.reset();
                }
            }
            if ( ImGui::IsItemDeactivatedAfterEdit() )
            {
                AppendHistory( std::make_shared<ChangeNameAction>( "Rename object", pObj ) );
                pObj->setName( renameBuffer_ );
                lastRenameObj_.reset();
            }

            if ( auto pObjLabel = std::dynamic_pointer_cast<ObjectLabel>( pObj ) )
            {
                if ( pObjLabel != oldLabelParams_.obj )
                {
                    oldLabelParams_.obj = pObjLabel;
                    const auto& positionedText = pObjLabel->getLabel();
                    oldLabelParams_.lastLabel = positionedText.text;
                    oldLabelParams_.labelBuffer = oldLabelParams_.lastLabel;
                }

                if ( ImGui::InputText( "Label", oldLabelParams_.labelBuffer, ImGuiInputTextFlags_AutoSelectAll ) )
                    pObjLabel->setLabel( { oldLabelParams_.labelBuffer, pObjLabel->getLabel().position } );
                if ( ImGui::IsItemDeactivatedAfterEdit() && oldLabelParams_.labelBuffer != oldLabelParams_.lastLabel )
                {
                    pObjLabel->setLabel( { oldLabelParams_.lastLabel, pObjLabel->getLabel().position } );
                    AppendHistory( std::make_shared<ChangeLabelAction>( "Change label", pObjLabel ) );
                    pObjLabel->setLabel( { oldLabelParams_.labelBuffer, pObjLabel->getLabel().position } );
                    oldLabelParams_.lastLabel = oldLabelParams_.labelBuffer;
                } else if ( !ImGui::IsItemActive() )
                {
                    const auto& positionedText = pObjLabel->getLabel();
                    oldLabelParams_.lastLabel = positionedText.text;
                    oldLabelParams_.labelBuffer = oldLabelParams_.lastLabel;
                }
            }
            else if ( oldLabelParams_.obj )
            {
                oldLabelParams_.obj.reset();
            }
        }
        else
            lastRenameObj_.reset();

        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemSpacing.x, smallItemSpacingY } );

        drawPrimitivesInfo( "Faces", totalFaces, totalSelectedFaces );
        drawPrimitivesInfo( "Vertices", totalVerts );
        drawPrimitivesInfo( "Points", totalPoints, totalSelectedPoints );

        ImGui::SetCursorPosY( ImGui::GetCursorPosY() + 2 * smallItemSpacingY );

        if ( selectionBbox_.valid() )
        {
            auto drawVec3 = [&style] ( std::string title, Vector3f& value, float width )
            {
                UI::inputTextCenteredReadOnly( ( "##" + title + "_x" ).c_str(), fmt::format("{:.3f}", value.x), width);
                ImGui::SameLine();
                UI::inputTextCenteredReadOnly( ( "##" + title + "_y" ).c_str(), fmt::format( "{:.3f}", value.y ), width );
                ImGui::SameLine();
                UI::inputTextCenteredReadOnly( ( "##" + title + "_z" ).c_str(), fmt::format( "{:.3f}", value.z ), width );

                ImGui::SameLine( 0, style.ItemInnerSpacing.x );
                ImGui::Text( "%s", title.c_str() );
            };

            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { style.ItemInnerSpacing.x, style.ItemSpacing.y } );
            const float fieldWidth = getSceneInfoItemWidth_( 3 );
            drawVec3( "Box min", selectionBbox_.min, fieldWidth );
            drawVec3( "Box max", selectionBbox_.max, fieldWidth );
            drawVec3( "Box size", bsize, fieldWidth );

            if ( selectionWorldBox_.valid() && bsizeStr != wbsizeStr )
                drawVec3( "World box size", wbsize, fieldWidth );

            ImGui::PopStyleVar();
        }

        ImGui::PopStyleVar();
        ImGui::Dummy( ImVec2( 0, 0 ) );
        ImGui::EndChild();
        ImGui::PopStyleVar();
    }

    return resultHeight;
}

bool ImGuiMenu::drawGeneralOptions_( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    const auto selectedVisualObjs = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( !selectedVisualObjs.empty() )
    {
        const auto& viewportid = viewer->viewport().id;
        if ( make_visualize_checkbox( selectedVisualObjs, "Visibility", VisualizeMaskType::Visibility, viewportid ) )
        {
            someChanges = true;
            if ( deselectNewHiddenObjects_ )
                for ( const auto& visObj : selectedVisualObjs )
                    if ( !visObj->isVisible( viewer->getPresentViewports() ) )
                        visObj->select( false );
        }
    }

    bool hasLocked = false, hasUnlocked = false;
    for ( const auto& s : selectedObjs )
    {
        if ( s->isLocked() )
            hasLocked = true;
        else
            hasUnlocked = true;
    }
    const bool mixedLocking = hasLocked && hasUnlocked;
    bool checked = hasLocked;
    someChanges |= UI::checkboxMixed( "Lock Transform", &checked, mixedLocking );
    if ( checked != hasLocked )
        for ( const auto& s : selectedObjs )
            s->setLocked( checked );

    return someChanges;
}

bool ImGuiMenu::drawAdvancedOptions_( const std::vector<std::shared_ptr<VisualObject>>& selectedObjs )
{
    if ( selectedObjs.empty() )
        return false;
    auto currWindow = ImGui::GetCurrentContext()->CurrentWindow;
    if ( currWindow )
        currWindow->DrawList->PushClipRect( currWindow->OuterRectClipped.Min, currWindow->OuterRectClipped.Max );
    if ( !RibbonButtonDrawer::CustomCollapsingHeader( "Advanced" ) )
    {
        if ( currWindow )
            currWindow->DrawList->PopClipRect();
        return false;
    }
    if ( currWindow )
        currWindow->DrawList->PopClipRect();

    make_light_strength( selectedObjs, "Shininess", [&] ( const VisualObject* obj )
    {
        return obj->getShininess();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setShininess( value );
    } );

    make_light_strength( selectedObjs, "Ambient Strength", [&] ( const VisualObject* obj )
    {
        return obj->getAmbientStrength();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setAmbientStrength( value );
    } );

    make_light_strength( selectedObjs, "Specular Strength", [&] ( const VisualObject* obj )
    {
        return obj->getSpecularStrength();
    }, [&] ( VisualObject* obj, float value )
    {
        obj->setSpecularStrength( value );
    } );

    return false;
}

bool ImGuiMenu::drawRemoveButton_( const std::vector<std::shared_ptr<Object>>& selectedObjs )
{
    bool someChanges = false;
    auto backUpColorBtn = ImGui::GetStyle().Colors[ImGuiCol_Button];
    auto backUpColorBtnH = ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
    auto backUpColorBtnA = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];

    if ( !allowRemoval_ )
    {
        const auto& colorDis = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    bool clicked = allowRemoval_ ?
        UI::button( "Remove", Vector2f( -1, 0 ) ) :
        ImGui::Button( "Remove", ImVec2( -1, 0 ) );
    if ( clicked )
    {
        someChanges |= true;
        if ( allowRemoval_ )
        {
            SCOPED_HISTORY( "Remove objects" );
            for ( int i = ( int )selectedObjs.size() - 1; i >= 0; --i )
                if ( selectedObjs[i] )
                {
                    // for now do it by one object
                    AppendHistory<ChangeSceneAction>( "Remove object", selectedObjs[i], ChangeSceneAction::Type::RemoveObject );
                    selectedObjs[i]->detachFromParent();
                }
        }
    }
    if ( !allowRemoval_ )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }

    return someChanges;
}

bool ImGuiMenu::drawDrawOptionsCheckboxes_( const std::vector<std::shared_ptr<VisualObject>>& selectedVisualObjs )
{
    bool someChanges = false;
    if ( selectedVisualObjs.empty() )
        return someChanges;

    bool allIsObjMesh = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectMeshHolder>();
    } );
    bool allIsObjLines = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectLinesHolder>();
    } );
    bool allIsObjPoints = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectPointsHolder>();
    } );
    bool allIsObjLabels = !selectedVisualObjs.empty() &&
        std::all_of( selectedVisualObjs.cbegin(), selectedVisualObjs.cend(), [] ( const std::shared_ptr<VisualObject>& obj )
    {
        return obj && obj->asType<ObjectLabel>();
    } );

    const auto& viewportid = viewer->viewport().id;

    if ( allIsObjMesh )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Flat Shading", MeshVisualizePropertyType::FlatShading, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Edges", MeshVisualizePropertyType::Edges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Edges", MeshVisualizePropertyType::SelectedEdges, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Faces", MeshVisualizePropertyType::SelectedFaces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Borders", MeshVisualizePropertyType::BordersHighlight, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Faces", MeshVisualizePropertyType::Faces, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Only Odd Fragments", MeshVisualizePropertyType::OnlyOddFragments, viewportid );
        bool allHaveTexture = true;
        for ( const auto& visObj : selectedVisualObjs )
        {
            auto meshObj = visObj->asType<ObjectMeshHolder>();
            assert( meshObj );
            allHaveTexture = allHaveTexture && 
                ( !meshObj->getTexture().pixels.empty() && !meshObj->getUVCoords().empty() );
            if ( !allHaveTexture )
                break;
        }
        if ( allHaveTexture )
            someChanges |= make_visualize_checkbox( selectedVisualObjs, "Texture", MeshVisualizePropertyType::Texture, viewportid );
    }
    if ( allIsObjLines )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Points", LinesVisualizePropertyType::Points, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Smooth corners", LinesVisualizePropertyType::Smooth, viewportid );
        make_width<ObjectLinesHolder>( selectedVisualObjs, "Line width", [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getLineWidth();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setLineWidth( value );
        }, true );
        make_width<ObjectLinesHolder>( selectedVisualObjs, "Point size", [&] ( const ObjectLinesHolder* objLines )
        {
            return objLines->getPointSize();
        }, [&] ( ObjectLinesHolder* objLines, float value )
        {
            objLines->setPointSize( value );
        } );
    }
    if ( allIsObjPoints )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Selected Points", PointsVisualizePropertyType::SelectedVertices, viewportid );
        make_width<ObjectPointsHolder>( selectedVisualObjs, "Point size", [&] ( const ObjectPointsHolder* objPoints )
        {
            return objPoints->getPointSize();
        }, [&] ( ObjectPointsHolder* objPoints, float value )
        {
            objPoints->setPointSize( value );
        } );
    }
    if ( allIsObjLabels )
    {
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Always on top", VisualizeMaskType::DepthTest, viewportid, true );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Source point", LabelVisualizePropertyType::SourcePoint, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Background", LabelVisualizePropertyType::Background, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Contour", LabelVisualizePropertyType::Contour, viewportid );
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Leader line", LabelVisualizePropertyType::LeaderLine, viewportid );
    }
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Invert Normals", VisualizeMaskType::InvertedNormals, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Name", VisualizeMaskType::Name, viewportid );
    someChanges |= make_visualize_checkbox( selectedVisualObjs, "Labels", VisualizeMaskType::Labels, viewportid );
    if ( viewer->isDeveloperFeaturesEnabled() )
        someChanges |= make_visualize_checkbox( selectedVisualObjs, "Clipping", VisualizeMaskType::ClippedByPlane, viewportid );

    return someChanges;
}

bool ImGuiMenu::drawDrawOptionsColors_( const std::vector<std::shared_ptr<VisualObject>>& selectedVisualObjs )
{
    bool someChanges = false;
    const auto selectedMeshObjs = getAllObjectsInTree<ObjectMeshHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    const auto selectedPointsObjs = getAllObjectsInTree<ObjectPointsHolder>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    const auto selectedLabelObjs = getAllObjectsInTree<ObjectLabel>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selectedVisualObjs.empty() )
        return someChanges;

    if ( getViewerInstance().viewport_list.size() > 1 )
    {
        ImGui::SetNextItemWidth( 75.0f * menu_scaling() );

        if (ImGui::BeginCombo( "Viewport Id",
            selectedViewport_.value() == 0 ? "Default" :
            std::to_string( selectedViewport_.value() ).c_str() ) )
        {
            if ( ImGui::Selectable( "Default" ) )
                selectedViewport_ = ViewportId{ 0 };

            for ( const auto& viewport : getViewerInstance().viewport_list )
            {
                if ( ImGui::Selectable( std::to_string( viewport.id.value() ).c_str() ) )
                    selectedViewport_ = viewport.id;
            }

            ImGui::EndCombo();
        }
    }

    make_color_selector<VisualObject>( selectedVisualObjs, ("Selected color##" + std::to_string(selectedViewport_.value())).c_str(), [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor(true, selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), true, selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Unselected color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getFrontColor( false, selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setFrontColor( Color( color ), false, selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Back Faces color", [&] ( const VisualObject* data )
    {
        return Vector4f( data->getBackColor( selectedViewport_ ) );
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
        data->setBackColor( Color( color ), selectedViewport_ );
    } );
    make_color_selector<VisualObject>( selectedVisualObjs, "Labels color", [&] ( const VisualObject* data )
    {
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
        return Vector4f( data->getLabelsColor( selectedViewport_ ) );
MR_SUPPRESS_WARNING_POP
    }, [&] ( VisualObject* data, const Vector4f& color )
    {
MR_SUPPRESS_WARNING_PUSH( "-Wdeprecated-declarations", 4996 )
        data->setLabelsColor( Color( color ), selectedViewport_ );
MR_SUPPRESS_WARNING_POP
    } );

    if ( !selectedMeshObjs.empty() )
    {
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Edges color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getEdgesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setEdgesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Selected Faces color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedFacesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedFacesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Selected Edges color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getSelectedEdgesColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setSelectedEdgesColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectMeshHolder>( selectedMeshObjs, "Borders color", [&] ( const ObjectMeshHolder* data )
        {
            return Vector4f( data->getBordersColor( selectedViewport_ ) );
        }, [&] ( ObjectMeshHolder* data, const Vector4f& color )
        {
            data->setBordersColor( Color( color ), selectedViewport_ );
        } );
    }
    if ( !selectedPointsObjs.empty() )
    {
        make_color_selector<ObjectPointsHolder>( selectedPointsObjs, "Selected Points color", [&] ( const ObjectPointsHolder* data )
        {
            return Vector4f( data->getSelectedVerticesColor( selectedViewport_ ) );
        }, [&] ( ObjectPointsHolder* data, const Vector4f& color )
        {
            data->setSelectedVerticesColor( Color( color ), selectedViewport_ );
        } );
    }
    if ( !selectedLabelObjs.empty() )
    {
        make_color_selector<ObjectLabel>( selectedLabelObjs, "Source point color", [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getSourcePointColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setSourcePointColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectLabel>( selectedLabelObjs, "Leader line color", [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getLeaderLineColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setLeaderLineColor( Color( color ), selectedViewport_ );
        } );
        make_color_selector<ObjectLabel>( selectedLabelObjs, "Contour color", [&] ( const ObjectLabel* data )
        {
            return Vector4f( data->getContourColor( selectedViewport_ ) );
        }, [&] ( ObjectLabel* data, const Vector4f& color )
        {
            data->setContourColor( Color( color ), selectedViewport_ );
        } );
    }

    if ( !selectedVisualObjs.empty() )
    {
        make_uint8_slider( selectedVisualObjs, "Alpha", [&] ( const VisualObject* data )
        {
            return data->getGlobalAlpha( selectedViewport_ );
        }, [&] ( VisualObject* data, uint8_t alpha )
        {
            data->setGlobalAlpha( alpha, selectedViewport_ );
        } );
    }

    return someChanges;
}

void ImGuiMenu::draw_custom_selection_properties( const std::vector<std::shared_ptr<Object>>& )
{}

float ImGuiMenu::drawTransform_()
{
    auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );

    const auto scaling = menu_scaling();
    auto& style = ImGui::GetStyle();

    float resultHeight_ = 0.f;
    if ( selected.size() == 1 && !selected[0]->isLocked() )
    {
        if ( !selectionChangedToSingleObj_ )
        {
            selectionChangedToSingleObj_ = true;
            nextFrameFixScroll_ = true;
        }
        resultHeight_ = ImGui::GetTextLineHeight() + style.FramePadding.y * 2 + style.ItemSpacing.y;
        bool openedContext = false;
        if ( drawCollapsingHeader_( "Transform", ImGuiTreeNodeFlags_DefaultOpen ) )
        {
            openedContext = drawTransformContextMenu_( selected[0] );
            const float transformHeight = ( ImGui::GetTextLineHeight() + style.FramePadding.y * 2 ) * 3 + style.ItemSpacing.y * 2;
            resultHeight_ += transformHeight + style.ItemSpacing.y;
            ImGui::BeginChild( "SceneTransform", ImVec2( 0, transformHeight ) );
            auto& data = *selected.front();

            auto xf = data.xf();
            Matrix3f q, r;
            decomposeMatrix3( xf.A, q, r );

            auto euler = ( 180 / PI_F ) * q.toEulerAngles();
            Vector3f scale{ r.x.x, r.y.y, r.z.z };

            bool inputDeactivated = false;
            bool inputChanged = false;

            ImGui::PushItemWidth( getSceneInfoItemWidth_( 3 ) );
            if ( uniformScale_ )
            {
                float midScale = ( scale.x + scale.y + scale.z ) / 3.0f;
                ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
                inputChanged = ImGui::DragFloatValid( "##scaleX", &midScale, midScale * 0.01f, 1e-3f, 1e+6f, "%.3f" );
                if ( inputChanged )
                    scale.x = scale.y = scale.z = midScale;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine();
            }
            else
            {
                inputChanged = ImGui::DragFloatValid( "##scaleX", &scale.x, scale.x * 0.01f, 1e-3f, 1e+6f, "%.3f" );
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                inputChanged = ImGui::DragFloatValid( "##scaleY", &scale.y, scale.y * 0.01f, 1e-3f, 1e+6f, "%.3f" ) || inputChanged;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
                inputChanged = ImGui::DragFloatValid( "##scaleZ", &scale.z, scale.z * 0.01f, 1e-3f, 1e+6f, "%.3f" ) || inputChanged;
                inputDeactivated = inputDeactivated || ImGui::IsItemDeactivatedAfterEdit();
                ImGui::SameLine( 0, ImGui::GetStyle().ItemInnerSpacing.x );
            }

            auto ctx = ImGui::GetCurrentContext();
            assert( ctx );
            auto window = ctx->CurrentWindow;
            assert( window );
            auto diff = ImGui::GetStyle().FramePadding.y - cCheckboxPadding * menu_scaling();
            ImGui::SetCursorPosY( ImGui::GetCursorPosY() + diff );
            UI::checkbox( "Uni-scale", &uniformScale_ );
            window->DC.CursorPosPrevLine.y -= diff;
            UI::setTooltipIfHovered( "Selects between uniform scaling or separate scaling along each axis", scaling );
            ImGui::PopItemWidth();

            const char* tooltipsRotation[3] = {
                "Rotation around Ox-axis, degrees",
                "Rotation around Oy-axis, degrees",
                "Rotation around Oz-axis, degrees"
            };
            ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
            auto resultRotation = ImGui::DragFloatValid3( "Rotation XYZ", &euler.x, invertedRotation_ ? -0.1f : 0.1f, -360.f, 360.f, "%.1f", 0, &tooltipsRotation );
            inputChanged = inputChanged || resultRotation.valueChanged;
            inputDeactivated = inputDeactivated || resultRotation.itemDeactivatedAfterEdit;
            if ( ImGui::IsItemHovered() )
            {
                ImGui::BeginTooltip();
                ImGui::Text( "Sequential intrinsic rotations around Oz, Oy and Ox axes." ); // see more https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations
                ImGui::EndTooltip();
            }

            constexpr float cZenithEps = 0.01f;
            if ( resultRotation.valueChanged && ImGui::IsMouseDragging( ImGuiMouseButton_Left ) )
            {
                // resolve singularity
                if ( std::fabs( euler.y ) > 90.f - cZenithEps )
                {
                    euler.x = euler.x > 0.f ? euler.x - 180.f : euler.x + 180.f;
                    euler.z = euler.z > 0.f ? euler.z - 180.f : euler.z + 180.f;
                    invertedRotation_ = !invertedRotation_;
                    euler.y = euler.y > 0.f ? 90.f - cZenithEps : -90.f + cZenithEps;
                }
            }
            if ( resultRotation.itemDeactivatedAfterEdit )
            {
                invertedRotation_ = false;
            }
            euler.y = std::clamp( euler.y, -90.0f + 2.0f * cZenithEps, 90.0f - 2.0f * cZenithEps );
            if ( inputChanged )
                xf.A = Matrix3f::rotationFromEuler( ( PI_F / 180 ) * euler ) * Matrix3f::scale( scale );

            const char* tooltipsTranslation[3] = {
                "Translation along Ox-axis",
                "Translation along Oy-axis",
                "Translation along Oz-axis"
            };
            const auto trSpeed = ( selectionBbox_.valid() && selectionBbox_.diagonal() > std::numeric_limits<float>::epsilon() ) ? 0.003f * selectionBbox_.diagonal() : 0.003f;

            ImGui::SetNextItemWidth( getSceneInfoItemWidth_() );
            auto wbsize = selectionWorldBox_.valid() ? selectionWorldBox_.size() : Vector3f::diagonal( 1.f );
            auto minSizeDim = wbsize.length();
            if ( minSizeDim == 0 )
                minSizeDim = 1.f;
            auto translation = xf.b;
            auto resultTranslation = ImGui::DragFloatValid3( "Translation", &translation.x, trSpeed, -cMaxTranslationMultiplier * minSizeDim, +cMaxTranslationMultiplier * minSizeDim, "%.3f", 0, &tooltipsTranslation );
            inputDeactivated = inputDeactivated || resultTranslation.itemDeactivatedAfterEdit;

            if ( resultTranslation.valueChanged )
                xf.b = translation;

            if ( xfHistUpdated_ )
                xfHistUpdated_ = !inputDeactivated;

            if ( xf != data.xf() && !xfHistUpdated_ )
            {
                AppendHistory<ChangeXfAction>( "Change XF", selected[0] );
                xfHistUpdated_ = true;
            }
            data.setXf( xf );
            ImGui::EndChild();
            if ( !openedContext )
                openedContext = drawTransformContextMenu_( selected[0] );
        }
        if ( !openedContext )
            drawTransformContextMenu_( selected[0] );
    }
    else
    {
        if ( selectionChangedToSingleObj_ )
            selectionChangedToSingleObj_ = false;
    }

    return resultHeight_;
}

std::vector<Object*> ImGuiMenu::getPreSelection_( Object* meshclicked,
                                             bool isShift, bool isCtrl,
                                             const std::vector<std::shared_ptr<Object>>& selected,
                                             const std::vector<std::shared_ptr<Object>>& all_objects )
{
    if ( selected.empty() || !isShift )
        return { meshclicked };

    const auto& first = isCtrl ? selected.back().get() : selected.front().get();

    auto firstIt = std::find_if( all_objects.begin(), all_objects.end(), [first] ( const std::shared_ptr<Object>& obj )
    {
        return obj.get() == first;
    } );
    auto clickedIt = std::find_if( all_objects.begin(), all_objects.end(), [meshclicked] ( const std::shared_ptr<Object>& obj )
    {
        return obj.get() == meshclicked;
    } );

    size_t start{ 0 };
    std::vector<Object*> res;
    if ( firstIt < clickedIt )
    {
        start = std::distance( all_objects.begin(), firstIt );
        res.resize( std::distance( firstIt, clickedIt + 1 ) );
    }
    else
    {
        start = std::distance( all_objects.begin(), clickedIt );
        res.resize( std::distance( clickedIt, firstIt + 1 ) );
    }
    for ( int i = 0; i < res.size(); ++i )
    {
        res[i] = all_objects[start + i].get();
    }
    return res;
}

bool ImGuiMenu::drawCollapsingHeader_( const char* label, ImGuiTreeNodeFlags flags )
{
    return ImGui::CollapsingHeader( label, flags );
}

void ImGuiMenu::draw_custom_tree_object_properties( Object& )
{}

bool ImGuiMenu::make_visualize_checkbox( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, unsigned type, MR::ViewportMask viewportid, bool invert /*= false*/ )
{
    auto realRes = getRealValue( selectedVisualObjs, type, viewportid, invert );
    bool checked = realRes.first;
    const bool res = UI::checkboxMixed( label, &checked, !realRes.second && realRes.first );
    if ( checked != realRes.first )
    {
        if ( invert )
            checked = !checked;
        for ( const auto& data : selectedVisualObjs )
            if ( data )
                data->setVisualizeProperty( checked, type, viewportid );
    }

    return res;
}

template<typename ObjectT>
void ImGuiMenu::make_color_selector( std::vector<std::shared_ptr<ObjectT>> selectedVisualObjs, const char* label,
                                std::function<Vector4f( const ObjectT* )> getter,
                                std::function<void( ObjectT*, const Vector4f& )> setter )
{
    auto color = getter( selectedVisualObjs[0].get() );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != color )
        {
            isAllTheSame = false;
            break;
        }
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        color = Vector4f::diagonal( 0.0f ); color[3] = 1.0f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }

    std::string storedName = label;
    for ( const auto& obj : selectedVisualObjs )
        storedName += std::to_string( intptr_t( obj.get() ) );

    const auto colorConstForComparation = color;
    color = getStoredColor_( storedName, Color( color ) );
    ImGui::PushItemWidth( 40 * menu_scaling() );
    if ( ImGui::ColorEdit4( label, &color.x,
        ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
        storedColor_ = { storedName,color };
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( color != colorConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), color );
}

void ImGuiMenu::make_light_strength( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label,
    std::function<float( const VisualObject* )> getter,
    std::function<void( VisualObject*, const float& )> setter
)
{
    if ( selectedVisualObjs.empty() )
        return;

    auto obj = selectedVisualObjs[0];
    auto value = getter( obj.get() );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != value )
        {
            isAllTheSame = false;
            break;
        }

    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = 0.f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 50 * menu_scaling() );
    ImGui::DragFloatValid( label, &value, 0.01f, -99.0f, 99.0f, "%.3f" );

    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), value );
}


void ImGuiMenu::make_uint8_slider( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, 
    std::function<uint8_t( const VisualObject* )> getter, std::function<void( VisualObject*, uint8_t )> setter )
{
    if ( selectedVisualObjs.empty() )
        return;

    auto obj = selectedVisualObjs[0];
    auto value = int( getter( obj.get() ) );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i].get() ) != value )
        {
            isAllTheSame = false;
            break;
        }

    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = 255;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 100 * menu_scaling() );
    ImGui::SliderInt( label, &value, 0, 255, "%d", ImGuiSliderFlags_AlwaysClamp );

    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data.get(), uint8_t( value ) );
}

template<typename ObjType>
void ImGuiMenu::make_width( std::vector<std::shared_ptr<VisualObject>> selectedVisualObjs, const char* label, 
    std::function<float( const ObjType* )> getter,
    std::function<void( ObjType*, const float& )> setter,
    bool lineWidth )
{
    auto objLines = selectedVisualObjs[0]->asType<ObjType>();
    auto value = getter( objLines );
    bool isAllTheSame = true;
    for ( int i = 1; i < selectedVisualObjs.size(); ++i )
        if ( getter( selectedVisualObjs[i]->asType<ObjType>() ) != value )
        {
            isAllTheSame = false;
            break;
        }
    auto backUpTextColor = ImGui::GetStyle().Colors[ImGuiCol_Text];
    if ( !isAllTheSame )
    {
        value = 0.f;
        ImGui::GetStyle().Colors[ImGuiCol_Text] = undefined;
    }
    const auto valueConstForComparation = value;

    ImGui::PushItemWidth( 40 * menu_scaling() );
    if ( lineWidth )
        ImGui::DragFloatValidLineWidth( label, &value );
    else
        ImGui::DragFloatValid( label, &value, 0.02f, 1.0f, 10.0f, "%.1f");
    ImGui::GetStyle().Colors[ImGuiCol_Text] = backUpTextColor;
    ImGui::PopItemWidth();
    if ( value != valueConstForComparation )
        for ( const auto& data : selectedVisualObjs )
            setter( data->asType<ObjType>(), value );
}

void ImGuiMenu::reorderSceneIfNeeded_()
{
    if ( !allowSceneReorder_ )
        return;

    const bool filledReorderCommand = !sceneReorderCommand_.who.empty() && sceneReorderCommand_.to;
    const bool sourceNotTarget = std::all_of( sceneReorderCommand_.who.begin(), sceneReorderCommand_.who.end(), [target = sceneReorderCommand_.to]( auto it )
    {
        return it != target;
    } );
    const bool trueTarget = !sceneReorderCommand_.before || sceneReorderCommand_.to->parent();
    const bool trueSource = std::all_of( sceneReorderCommand_.who.begin(), sceneReorderCommand_.who.end(), [] ( auto it )
    {
        return bool( it->parent() );
    } );
    if ( !( filledReorderCommand && sourceNotTarget && trueSource && trueTarget ) )
    {
        sceneReorderCommand_ = {};
        return;
    }

    bool dragOrDropFailed = false;
    std::shared_ptr<Object> childTo = nullptr;
    if ( sceneReorderCommand_.before )
    {
        for ( auto childToItem : sceneReorderCommand_.to->parent()->children() )
            if ( childToItem.get() == sceneReorderCommand_.to )
            {
                childTo = childToItem;
                break;
            }
        assert( childTo );
    }

    struct MoveAction
    {
        std::shared_ptr<ChangeSceneAction> detachAction;
        std::shared_ptr<ChangeSceneAction> attachAction;
    };
    std::vector<MoveAction> actionList;
    for ( const auto& source : sceneReorderCommand_.who )
    {
        std::shared_ptr<Object> sourcePtr = nullptr;
        for ( auto child : source->parent()->children() )
            if ( child.get() == source )
            {
                sourcePtr = child;
                break;
            }
        assert( sourcePtr );

        auto detachAction = std::make_shared<ChangeSceneAction>( "Detach object", sourcePtr, ChangeSceneAction::Type::RemoveObject );
        bool detachSuccess = sourcePtr->detachFromParent();
        if ( !detachSuccess )
        {
            showErrorModal( "Cannot preform such reorder" );
            dragOrDropFailed = true;
            break;
        }

        auto attachAction = std::make_shared<ChangeSceneAction>( "Attach object", sourcePtr, ChangeSceneAction::Type::AddObject );
        bool attachSucess{ false };
        if ( !sceneReorderCommand_.before )
            attachSucess = sceneReorderCommand_.to->addChild( sourcePtr );
        else
            attachSucess = sceneReorderCommand_.to->parent()->addChildBefore( sourcePtr, childTo );
        if ( !attachSucess )
        {
            detachAction->action( HistoryAction::Type::Undo );
            showErrorModal( "Cannot preform such reorder" );
            dragOrDropFailed = true;
            break;
        }

        actionList.push_back( { detachAction, attachAction } );
    }

    if ( dragOrDropFailed )
    {
        for ( int i = int( actionList.size() ) - 1; i >= 0; --i )
        {
            actionList[i].attachAction->action( HistoryAction::Type::Undo );
            actionList[i].detachAction->action( HistoryAction::Type::Undo );
        }
    }
    else
    {
        SCOPED_HISTORY( "Reorder scene" );
        for ( const auto& moveAction : actionList )
        {
            AppendHistory( moveAction.detachAction );
            AppendHistory( moveAction.attachAction );
        }
    }
    sceneReorderCommand_ = {};
    dragTrigger_ = false;
}

Vector4f ImGuiMenu::getStoredColor_( const std::string& str, const Color& defaultColor ) const
{
    if ( !storedColor_ || storedColor_->first != str )
        return Vector4f( defaultColor );
    return storedColor_->second;
}

void ImGuiMenu::updateSceneWindowScrollIfNeeded_()
{
    auto window = ImGui::GetCurrentContext()->CurrentWindow;
    if ( !window )
        return;

    ScrollPositionPreservation scrollInfo;
    scrollInfo.relativeMousePos = ImGui::GetMousePos().y - window->Pos.y;
    scrollInfo.absLinePosRatio = window->ContentSize.y == 0.0f ? 0.0f : ( scrollInfo.relativeMousePos + window->Scroll.y ) / window->ContentSize.y;

    if ( nextFrameFixScroll_ )
    {
        nextFrameFixScroll_ = false;
        window->Scroll.y = std::clamp( prevScrollInfo_.absLinePosRatio * window->ContentSize.y - prevScrollInfo_.relativeMousePos, 0.0f, window->ScrollMax.y );
    }
    else if ( dragObjectsMode_ )
    {
        float relativeMousePosRatio = window->Size.y == 0.0f ? 0.0f : scrollInfo.relativeMousePos / window->Size.y;
        float shift = 0.0f;
        if ( relativeMousePosRatio < 0.05f )
            shift = ( relativeMousePosRatio - 0.05f ) * 25.0f - 1.0f;
        else if ( relativeMousePosRatio > 0.95f )
            shift = ( relativeMousePosRatio - 0.95f ) * 25.0f + 1.0f;

        auto newScroll = std::clamp( window->Scroll.y + shift, 0.0f, window->ScrollMax.y );
        if ( newScroll != window->Scroll.y )
        {
            window->Scroll.y = newScroll;
            getViewerInstance().incrementForceRedrawFrames();
        }
    }

    const ImGuiPayload* payloadCheck = ImGui::GetDragDropPayload();
    bool dragModeNow = payloadCheck && std::string_view( payloadCheck->DataType ) == "_TREENODE";
    if ( dragModeNow && !dragObjectsMode_ )
    {
        dragObjectsMode_ = true;
        nextFrameFixScroll_ = true;
        getViewerInstance().incrementForceRedrawFrames( 2, true );
    }
    else if ( !dragModeNow && dragObjectsMode_ )
    {
        dragObjectsMode_ = false;
        nextFrameFixScroll_ = true;
        getViewerInstance().incrementForceRedrawFrames( 2, true );
    }

    if ( !nextFrameFixScroll_ )
        prevScrollInfo_ = scrollInfo;
}

void ImGuiMenu::draw_custom_plugins()
{
    pluginsCache_.validate( viewer->plugins );
    StateBasePlugin* enabled = pluginsCache_.findEnabled();

    float availibleWidth = 200.0f * menu_scaling();

    auto selectedObjects = getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    auto selectedVisObjects = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );

    ImGui::SetNextWindowPos( ImVec2( 410.0f * menu_scaling(), 0 ), ImGuiCond_FirstUseEver );
    ImGui::SetNextWindowSize( ImVec2( 0.0f, 0.0f ), ImGuiCond_FirstUseEver );
    ImGui::Begin( "Plugins", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize );

    ImGui::SetCursorPosX( 570.f );
    if ( ImGui::InputText( "Search", searchPluginsString_ ) )
    {
        Viewer::instanceRef().incrementForceRedrawFrames( 2, true );
    }

    auto& colors = ImGui::GetStyle().Colors;
    auto backUpButtonColor = colors[ImGuiCol_Button];
    auto backUpTextColor = colors[ImGuiCol_Text];

    ImGui::BeginTabBar( "##CustomPluginsTabs", ImGuiTabBarFlags_TabListPopupButton );

    const int pluginsPerLine = 4;
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        StatePluginTabs tab = StatePluginTabs( t );
        const auto& plugins = pluginsCache_.getTabPlugins( tab );
        int counter = 0;
        for ( auto& plugin : plugins )
        {
            if ( !plugin->checkStringMask( searchPluginsString_ ) )
                continue;

            if ( counter == 0 )
            {
                if ( !ImGui::BeginTabItem( StateBasePlugin::getTabName( tab ) ) )
                    break;
            }

            std::string requirements = plugin->isAvailable( selectedObjects );
            bool canEnable = !enabled && requirements.empty();

            if ( plugin->isEnabled() )
            {
                colors[ImGuiCol_Button] = { 0.0f,0.8f,0.0f,1.0f };
                colors[ImGuiCol_Text] = { 0.0f,0.0f,0.0f,1.0f };
            }
            else if ( !canEnable )
            {
                colors[ImGuiCol_Button] = { 0.5f,0.5f,0.5f,1.0f };
                colors[ImGuiCol_Text] = { 1.0f,1.0f,1.0f,1.0f };
            }

            if ( counter % pluginsPerLine != 0 )
                ImGui::SameLine();

            if ( ImGui::Button( plugin->plugin_name.c_str(), ImVec2( availibleWidth, 0 ) ) )
            {
                if ( plugin->isEnabled() )
                    plugin->enable( false );
                else if ( canEnable )
                    plugin->enable( true );
            }
            ++counter;

            colors[ImGuiCol_Text] = { 1.0f,1.0f,1.0f,1.0f };
            const auto strTooltip = plugin->getTooltip();
            if ( ImGui::IsItemHovered() && ( !strTooltip.empty() || !requirements.empty() ) )
            {
                ImVec2 textSize;
                if ( requirements.empty() )
                    textSize = ImGui::CalcTextSize( strTooltip.c_str(), NULL, false, 400.f );
                else
                {
                    if ( strTooltip.empty() )
                        textSize = ImGui::CalcTextSize( requirements.c_str(), NULL, false, 400.f );
                    else
                        textSize = ImGui::CalcTextSize( ( strTooltip + "\n" + requirements ).c_str(), NULL, false, 400.f );
                }
                ImGui::SetNextWindowContentSize( textSize );
                ImGui::BeginTooltip();
                if ( !strTooltip.empty() )
                {
                    ImGui::TextWrapped( "%s", strTooltip.c_str() );
                }
                if ( !requirements.empty() )
                {
                    ImGui::PushStyleColor( ImGuiCol_Text, Color::red().getUInt32() );
                    ImGui::TextWrapped( "%s", requirements.c_str() );
                    ImGui::PopStyleColor();
                }
                ImGui::EndTooltip();
            }

            colors[ImGuiCol_Button] = backUpButtonColor;
            colors[ImGuiCol_Text] = backUpTextColor;
        }

        int counterModifier = 0;
        if ( !selectedVisObjects.empty() )
        {
            if ( counter != 0 )
                ImGui::Separator();

            for ( const auto& modifier : modifiers_ )
            {
                if ( tab != modifier->getTab() )
                    continue;

                if ( !modifier->checkStringMask( searchPluginsString_ ) )
                    continue;

                if ( counter + counterModifier == 0 )
                {
                    if ( !ImGui::BeginTabItem( StateBasePlugin::getTabName( tab ) ) )
                        break;
                }

                if ( counterModifier % pluginsPerLine != 0 )
                    ImGui::SameLine();

                if ( ImGui::Button( modifier->name().c_str(), ImVec2( availibleWidth, 0 ) ) )
                {
                    // Here should be popups
                    if ( modifier->modify( selectedVisObjects ) )
                        spdlog::info( "{}: success", modifier->name() );
                    else
                        spdlog::warn( "{}: failure", modifier->name() );
                }
                ++counterModifier;
            }
        }
        if ( ( counter + counterModifier ) != 0 )
        {
            ImGui::EndTabItem();
        }
    }

    ImGui::SetWindowSize( ImGui::GetWindowSize() );
    ImGui::EndTabBar();
    ImGui::End();

    if ( enabled && enabled->isEnabled() )
    {
        if ( allowRemoval_ )
            allowRemoval_ = false;
        enabled->drawDialog( menu_scaling(), ImGui::GetCurrentContext() );
        if ( !enabled->dialogIsOpen() )
            enabled->enable( false );
    }
    else
    {
        if ( !allowRemoval_ )
            allowRemoval_ = true;
    }
}

void ImGuiMenu::draw_mr_menu()
{
    // Mesh
    ProgressBar::setup( menu_scaling() );
    const auto& viewportParameters = viewer->viewport().getParameters();
    if ( drawCollapsingHeader_( "Main", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        draw_history_block_();
        float w = ImGui::GetContentRegionAvail().x;
        float p = ImGui::GetStyle().FramePadding.x;
        if ( ImGui::Button( "Load##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) )
        {
            auto filenames = openFilesDialog( { {},{},MeshLoad::getFilters() | PointsLoad::Filters | SceneFileFilters } );
            viewer->loadFiles( filenames );
        }
        ImGui::SameLine( 0, p );
        draw_open_recent_button_();
        ImGui::SameLine( 0, p );
        if ( ImGui::Button( "Load Dir##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto directory = openFolderDialog();
            if ( !directory.empty() )
            {
                auto container = makeObjectTreeFromFolder( directory );
                if ( container.has_value() && !container->children().empty() )
                {
                    auto obj = std::make_shared<Object>( std::move( container.value() ) );
                    obj->setName( utf8string( directory.stem() ) );
                    selectRecursive( *obj );
                    AppendHistory<ChangeSceneAction>( "Load Dir", obj, ChangeSceneAction::Type::AddObject );
                    SceneRoot::get().addChild( obj );
                    viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                }
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_DICOM) && !defined(MRMESH_NO_VOXEL)
                else
                {
                    ProgressBar::orderWithMainThreadPostProcessing( "Open directory", [directory, viewer = viewer] () -> std::function<void()>
                    {
                        ProgressBar::nextTask( "Load DICOM Folder" );
                        auto loadRes = VoxelsLoad::loadDCMFolder( directory, 4, ProgressBar::callBackSetProgress );
                        if ( loadRes.has_value() && !ProgressBar::isCanceled() )
                        {
                            std::shared_ptr<ObjectVoxels> voxelsObject = std::make_shared<ObjectVoxels>();
                            voxelsObject->setName( loadRes->name );
                            ProgressBar::setTaskCount( 2 );
                            ProgressBar::nextTask( "Construct ObjectVoxels" );
                            voxelsObject->construct( loadRes->vdbVolume, ProgressBar::callBackSetProgress );
                            auto bins = voxelsObject->histogram().getBins();
                            auto minMax = voxelsObject->histogram().getBinMinMax( bins.size() / 3 );

                            ProgressBar::nextTask( "Create ISO surface" );
                            voxelsObject->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                            voxelsObject->select( true );
                            return [viewer, voxelsObject] ()
                            {
                                AppendHistory<ChangeSceneAction>( "Load Voxels", voxelsObject, ChangeSceneAction::Type::AddObject );
                                SceneRoot::get().addChild( voxelsObject );
                                viewer->viewport().preciseFitDataToScreenBorder( { 0.9f } );
                            };
                        }
                        else
                            return [error = loadRes.error()] ()
                            {
                                auto menu = getViewerInstance().getMenuPlugin();
                                if ( menu )
                                    menu->showErrorModal( error );
                            };
                    }, 2 );
                }
#endif
            }
        }

        if ( ImGui::Button( "Save##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto filters = MeshSave::Filters | LinesSave::Filters | PointsSave::Filters;
#if !defined(__EMSCRIPTEN__) && !defined(MRMESH_NO_VOXEL)
            filters = filters | VoxelsSave::Filters;
#endif
            auto savePath = saveFileDialog( { {}, {}, filters } );
            if ( !savePath.empty() )
                viewer->saveToFile( savePath );
        }
        ImGui::SameLine( 0, p );

        if ( ImGui::Button( "Save Scene##Main", ImVec2( ( w - p ) / 2.f, 0 ) ) )
        {
            auto savePath = saveFileDialog( { {},{},SceneFileFilters } );

            ProgressBar::orderWithMainThreadPostProcessing( "Saving scene", [savePath, &root = SceneRoot::get(), viewer = this->viewer]()->std::function<void()>
            {
                auto res = serializeObjectTree( root, savePath, [] ( float progress )
                {
                    return ProgressBar::setProgress( progress );
                } );
                if ( !res.has_value() )
                    spdlog::error( res.error() );

                return[savePath, viewer, success = res.has_value()]()
                {
                    if ( success )
                        viewer->recentFilesStore.storeFile( savePath );
                };
            } );
        }

        if ( ImGui::Button( "New Issue##Main", ImVec2( w, 0 ) ) )
        {
            OpenLink( "https://meshinspector.github.io/ReportIssue" );
        }
        if ( ImGui::Button( "Capture Screen##Main", ImVec2( w, 0 ) ) )
        {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t( now );
            auto name = fmt::format( "Screenshot_{:%Y-%m-%d_%H-%M-%S}", fmt::localtime( t ) );

            auto savePath = saveFileDialog( { name,{},ImageSave::Filters } );
            if ( !savePath.empty() )
            {
                auto image = viewer->captureScreenShot();
                auto res = ImageSave::toAnySupportedFormat( image, savePath );

                if ( !res.has_value() )
                    spdlog::warn( "Error saving screenshot: {}", res.error() );
            }
        }
    }

    // Viewing options
    if ( drawCollapsingHeader_( "Viewing Options", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        ImGui::PushItemWidth( 80 * menu_scaling() );
        auto fov = viewportParameters.cameraViewAngle;
        ImGui::DragFloatValid( "Camera FOV", &fov, 0.001f, 0.01f, 179.99f );
        viewer->viewport().setCameraViewAngle( fov );

        bool showGlobalBasis = viewer->globalBasisAxes->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show Global Basis", &showGlobalBasis );
        viewer->viewport().showGlobalBasis( showGlobalBasis );

        bool showRotCenter = viewer->rotationSphere->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show rotation center", &showRotCenter );
        viewer->viewport().showRotationCenter( showRotCenter );

        // Orthographic view
        bool orth = viewportParameters.orthographic;
        ImGui::Checkbox( "Orthographic view", &orth );
        viewer->viewport().setOrthographic( orth );

        bool flatShading = SceneSettings::get( SceneSettings::Type::MeshFlatShading );
        bool flatShadingBackup = flatShading;
        ImGui::Checkbox( "Default shading flat", &flatShading );
        if ( flatShadingBackup != flatShading )
            SceneSettings::set( SceneSettings::Type::MeshFlatShading, flatShading );
        ImGui::PopItemWidth();

        bool showAxes = viewer->basisAxes->isVisible( viewer->viewport().id );
        ImGui::Checkbox( "Show axes", &showAxes );
        viewer->viewport().showAxes( showAxes );

        const std::string typeColorEditStr = "Background";
        auto backgroundColor = getStoredColor_( typeColorEditStr, viewportParameters.backgroundColor );
        if ( ImGui::ColorEdit4( typeColorEditStr.c_str(), &backgroundColor.x,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
            storedColor_ = { typeColorEditStr,backgroundColor };
        else if ( !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) && storedColor_ && storedColor_->first == typeColorEditStr )
            storedColor_ = {};
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );
    }

    if ( ImGui::Button( "Fit Data", ImVec2( -1, 0 ) ) )
    {
        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::Visible } );
    }
    if ( ImGui::Button( "Fit Selected", ImVec2( -1, 0 ) ) )
    {
        viewer->viewport().preciseFitDataToScreenBorder( { 0.9f, false, Viewport::FitMode::SelectedPrimitives } );
    }

    if ( viewer->isAlphaSortAvailable() )
    {
        bool alphaSortBackUp = viewer->isAlphaSortEnabled();
        bool alphaBoxVal = alphaSortBackUp;
        ImGui::Checkbox( "Alpha Sort", &alphaBoxVal );
        if ( alphaBoxVal != alphaSortBackUp )
            viewer->enableAlphaSort( alphaBoxVal );
    }

    if ( drawCollapsingHeader_( "Viewports" ) )
    {
        auto configBackup = viewportConfig_;
        ImGui::RadioButton( "Single", ( int* )&viewportConfig_, ViewportConfigurations::Single );
        ImGui::RadioButton( "Horizontal", ( int* )&viewportConfig_, ViewportConfigurations::Horizontal );
        ImGui::RadioButton( "Vertical", ( int* )&viewportConfig_, ViewportConfigurations::Vertical );
        ImGui::RadioButton( "Quad", ( int* )&viewportConfig_, ViewportConfigurations::Quad );
        if ( configBackup != viewportConfig_ )
        {
            for ( int i = int( viewer->viewport_list.size() ) - 1; i > 0; --i )
                viewer->erase_viewport( i );

            auto win = glfwGetCurrentContext();
            int window_width, window_height;
            glfwGetWindowSize( win, &window_width, &window_height );

            auto bounds = viewer->getViewportsBounds();

            float width = MR::width( bounds );
            float height = MR::height( bounds );

            ViewportRectangle rect;
            switch ( viewportConfig_ )
            {
            case Vertical:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height;
                viewer->viewport().setViewportRect( rect );

                rect.min.x = bounds.min.x + width * 0.5f;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height;
                viewer->append_viewport( rect );
                break;
            case Horizontal:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->viewport().setViewportRect( rect );

                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y + height * 0.5f;
                rect.max.x = rect.min.x + width;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );
                break;
            case Quad:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->viewport().setViewportRect( rect );

                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y + height * 0.5f;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );

                rect.min.x = bounds.min.x + width * 0.5f;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );

                rect.min.x = bounds.min.x + width * 0.5f;
                rect.min.y = bounds.min.y + height * 0.5f;
                rect.max.x = rect.min.x + width * 0.5f;
                rect.max.y = rect.min.y + height * 0.5f;
                viewer->append_viewport( rect );
                break;
            case Single:
            default:
                rect.min.x = bounds.min.x;
                rect.min.y = bounds.min.y;
                rect.max.x = rect.min.x + width;
                rect.max.y = rect.min.y + height;
                viewer->viewport().setViewportRect( rect );
                break;
            }
            postResize_( window_width, window_height );
        }
    }

    if ( drawCollapsingHeader_( "Clipping plane" ) )
    {
        auto plane = viewportParameters.clippingPlane;
        auto showPlane = viewer->clippingPlaneObject->isVisible( viewer->viewport().id );
        plane.n = plane.n.normalized();
        auto w = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth( w );
        ImGui::DragFloatValid3( "##ClippingPlaneNormal", &plane.n.x, 1e-3f );
        ImGui::SetNextItemWidth( w / 2.0f );
        ImGui::DragFloatValid( "##ClippingPlaneD", &plane.d, 1e-3f );
        ImGui::SameLine();
        ImGui::Checkbox( "Show##ClippingPlane", &showPlane );
        viewer->viewport().setClippingPlane( plane );
        viewer->viewport().showClippingPlane( showPlane );
    }
    ImGui::Text( "Current view: %d", viewer->viewport().id.value() );
    mainWindowPos_ = ImGui::GetWindowPos();
    mainWindowSize_ = ImGui::GetWindowSize();
}

void ImGuiMenu::draw_history_block_()
{
    auto historyStore = viewer->getGlobalHistoryStore();
    if ( !historyStore )
        return;
    auto backUpColorBtn = ImGui::GetStyle().Colors[ImGuiCol_Button];
    auto backUpColorBtnH = ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered];
    auto backUpColorBtnA = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];
    const auto& colorDis = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
    auto undos = historyStore->getNActions( 10u, HistoryAction::Type::Undo );
    auto redos = historyStore->getNActions( 10u, HistoryAction::Type::Redo );
    if ( undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if ( ImGui::Button( "Undo##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) && !undos.empty() )
    {
        historyStore->undo();
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::BeginCombo( "##UndoStack", "##UndoStack", ImGuiComboFlags_NoPreview ) )
    {
        if ( undos.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( int i = 0; i < undos.size(); ++i )
        {
            if ( ImGui::Selectable( ( undos[i] + "##" + std::to_string( i ) ).c_str() ) )
            {
                for ( int j = 0; j <= i; ++j )
                    historyStore->undo();
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
    if ( redos.empty() && !undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = colorDis;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = colorDis;
    }
    else if ( !redos.empty() && undos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::Button( "Redo##Main", ImVec2( ( w - p ) / 2.f - p - ImGui::GetFrameHeight(), 0 ) ) && !redos.empty() )
    {
        historyStore->redo();
    }
    ImGui::SameLine( 0, p );
    if ( ImGui::BeginCombo( "##RedoStack", "##RedoStack", ImGuiComboFlags_NoPreview ) )
    {
        if ( redos.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( int i = 0; i < redos.size(); ++i )
        {
            if ( ImGui::Selectable( ( redos[i] + "##" + std::to_string( i ) ).c_str() ) )
            {
                for ( int j = 0; j <= i; ++j )
                    historyStore->redo();
            }
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
    if ( redos.empty() )
    {
        ImGui::GetStyle().Colors[ImGuiCol_Button] = backUpColorBtn;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered] = backUpColorBtnH;
        ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = backUpColorBtnA;
    }
}

void ImGuiMenu::draw_open_recent_button_()
{
    if ( ImGui::BeginCombo( "##Recently Loaded", "##Recently Loaded", ImGuiComboFlags_NoPreview ) )
    {
        auto filenames = viewer->recentFilesStore.getStoredFiles();
        if ( filenames.empty() )
            ImGui::CloseCurrentPopup();
        const auto storedColor = ImGui::GetStyle().Colors[ImGuiCol_Header];
        ImGui::GetStyle().Colors[ImGuiCol_Header] = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
        for ( const auto& file : filenames )
        {
            if ( ImGui::Selectable( utf8string( file ).c_str() ) )
                viewer->loadFiles( std::vector<std::filesystem::path>( { file } ) );
        }
        ImGui::GetStyle().Colors[ImGuiCol_Header] = storedColor;
        ImGui::EndCombo();
    }
}

void ImGuiMenu::drawShortcutsWindow_()
{
    const auto& style = ImGui::GetStyle();
    const float hotkeysWindowWidth = 300 * menu_scaling();
    size_t numLines = 2;

    if ( shortcutManager_ )
        numLines += shortcutManager_->getShortcutList().size();

    const float hotkeysWindowHeight = ( style.WindowPadding.y * 2 + numLines * ( ImGui::GetTextLineHeight() + style.ItemSpacing.y ) );

    ImVec2 windowPos = ImGui::GetMousePos();
    windowPos.x = std::min( windowPos.x, Viewer::instanceRef().window_width - hotkeysWindowWidth );
    windowPos.y = std::min( windowPos.y, Viewer::instanceRef().window_height - hotkeysWindowHeight );

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Appearing );
    ImGui::SetNextWindowSize( ImVec2( hotkeysWindowWidth, hotkeysWindowHeight ) );
    ImGui::Begin( "HotKeys", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoFocusOnAppearing );

    ImFont font = *ImGui::GetFont();
    font.Scale = 1.2f;
    ImGui::PushFont( &font );
    ImGui::Text( "Hot Key List" );
    ImGui::PopFont();
    ImGui::NewLine();
    if ( shortcutManager_ )
    {
        const auto& shortcutsList = shortcutManager_->getShortcutList();
        for ( const auto& [key, category, name] : shortcutsList )
            ImGui::Text( "%s - %s", ShortcutManager::getKeyFullString( key ).c_str(), name.c_str() );
    }
    ImGui::End();
}

float ImGuiMenu::getSceneInfoItemWidth_(size_t itemCount)
{
    if ( itemCount == 0 )
        return 0;
    /// 100 is the widest label's size
    return ( ImGui::GetContentRegionAvail().x - 100.0f * menu_scaling() - ImGui::GetStyle().ItemInnerSpacing.x * ( itemCount - 1 ) ) / float ( itemCount );
}

void ImGuiMenu::add_modifier( std::shared_ptr<MeshModifier> modifier )
{
    if ( modifier )
        modifiers_.push_back( modifier );
}

void ImGuiMenu::allowSceneReorder( bool allow )
{
    allowSceneReorder_ = allow;
}

void ImGuiMenu::allowObjectsRemoval( bool allow )
{
    allowRemoval_ = allow;
}

void ImGuiMenu::tryRenameSelectedObject()
{
    const auto selected = getAllObjectsInTree( &SceneRoot::get(), ObjectSelectivityType::Selected );
    if ( selected.size() != 1 )
        return;
    renameBuffer_ = selected[0]->name();
    showRenameModal_ = true;
}

void ImGuiMenu::setObjectTreeState( const Object* obj, bool open )
{
    if ( obj )
        sceneOpenCommands_[obj] = open;
}

void ImGuiMenu::setShowShortcuts( bool val )
{
    showShortcuts_ = val;
}

bool ImGuiMenu::getShowShortcuts() const
{
    return showShortcuts_;
}

void ImGuiMenu::PluginsCache::validate( const std::vector<ViewerPlugin*>& viewerPlugins )
{
    // if same then cache is valid
    if ( viewerPlugins == allPlugins_ )
        return;

    allPlugins_ = viewerPlugins;

    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
        sortedCustomPlufins_[t] = {};
    for ( const auto& plugin : allPlugins_ )
    {
        StateBasePlugin* customPlugin = dynamic_cast< StateBasePlugin* >( plugin );
        if ( customPlugin )
            sortedCustomPlufins_[int( customPlugin->getTab() )].push_back( customPlugin );
    }
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        auto& tabPlugins = sortedCustomPlufins_[t];
        std::sort( tabPlugins.begin(), tabPlugins.end(), [] ( const auto& a, const auto& b )
        {
            return a->sortString() < b->sortString();
        } );
    }
}

StateBasePlugin* ImGuiMenu::PluginsCache::findEnabled() const
{
    for ( int t = 0; t < int( StatePluginTabs::Count ); ++t )
    {
        const auto& tabPlugins = sortedCustomPlufins_[t];
        for ( auto plug : tabPlugins )
            if ( plug->isEnabled() )
                return plug;
    }
    return nullptr;
}

const std::vector<StateBasePlugin*>& ImGuiMenu::PluginsCache::getTabPlugins( StatePluginTabs tab ) const
{
    return sortedCustomPlufins_[int( tab )];
}

} // end namespace
