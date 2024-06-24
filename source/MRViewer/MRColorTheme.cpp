#include "MRColorTheme.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectLabel.h"
#include "MRRibbonButtonDrawer.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRViewport.h"
#include "MRMesh/MRDirectory.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRPch/MRSuppressWarning.h"
#include "ImGuiMenu.h"
#include "MRUIStyle.h"
#include <imgui.h>
#include <assert.h>
#include <fstream>

#ifdef __EMSCRIPTEN__

extern "C" 
{
// 0 - dark
// 1 - light
EMSCRIPTEN_KEEPALIVE int emsChangeColorTheme( int theme )
{
    using namespace MR;
    if ( theme == 0 )
        ColorTheme::setupDefaultDark();
    else
        ColorTheme::setupDefaultLight();
    ColorTheme::apply();
    return 1;
}



EMSCRIPTEN_KEEPALIVE int emsGetColorTheme()
{
    using namespace MR;
    return int( ColorTheme::getPreset() == ColorTheme::Preset::Light );
}
}
#endif

namespace MR
{

ColorTheme& ColorTheme::instance()
{
    static ColorTheme instance;
    return instance;
}

void ColorTheme::setupFromJson( const Json::Value& root, Type type )
{
    auto& instance = ColorTheme::instance();

    bool valid = true, defined = true;

    Preset themePreset{};
    if ( root["ImGuiPreset"].isString() )
        themePreset = root["ImGuiPreset"].asString() == getPresetName( Preset::Light ) ? Preset::Light : Preset::Dark;
    else
        valid = false;
    // Start with fallback theme - newly introduced colors will be filled from built-in theme
    if ( type == Type::User )
    {
        std::string saveThemeName = instance.themeName_;
        if ( themePreset == Preset::Dark )
            setupDefaultDark();
        else
            setupDefaultLight();
        instance.themeName_ = saveThemeName;
    }
    instance.themePreset_ = themePreset;
    instance.type_ = type;

    if ( valid )
    {
        if ( instance.sceneColors_.size() < SceneColors::Count )
            instance.sceneColors_.resize( SceneColors::Count );

        for ( int i = 0; i < SceneColors::Count; ++i )
        {
            auto name = SceneColors::getName( SceneColors::Type( i ) );
            if ( root[name].isObject() )
                deserializeFromJson( root[name], instance.sceneColors_[i] );
            else
                defined = false;
        }
    }

    if ( valid )
    {
        if ( root["Ribbon Colors"].isObject() )
        {
            const auto& ribColors = root["Ribbon Colors"];
            for ( int i = 0; i < int( RibbonColorsType::Count ); ++i )
            {
                auto name = getRibbonColorTypeName( RibbonColorsType( i ) );
                if ( ribColors[name].isObject() )
                    deserializeFromJson( ribColors[name], instance.newUIColors_[i] );
                else
                    defined = false;
            }
        }
        else
            defined = false;
    }

    if ( valid )
    {
        if ( root["Viewport Colors"].isObject() )
        {
            const auto& viewportColors = root["Viewport Colors"];
            for ( int i = 0; i < int( ViewportColorsType::Count ); ++i )
            {
                auto name = getViewportColorTypeName( ViewportColorsType( i ) );
                if ( viewportColors[name].isObject() )
                    deserializeFromJson( viewportColors[name], instance.viewportColors_[i] );
                else
                    defined = false;
            }
        }
        else
            defined = false;
    }

    if ( !valid || ( type == Type::Default && !defined ) )
    {
        spdlog::error( "Color theme deserialization failed: invalid json schema." );
        instance.sceneColors_.clear();
    }
}

void ColorTheme::setupFromFile( const std::filesystem::path& path, Type type )
{
    auto res = deserializeJsonValue( path );
    if ( !res )
        spdlog::error( "Color theme deserialization failed: {}", res.error() );

    const auto root = res ? std::move( res.value() ) : Json::Value(); // Handle fail in `setupFromJson`
    return setupFromJson( root, type );
}

void ColorTheme::serializeCurrentToFile( const std::filesystem::path& path )
{
    Json::Value root;
    serializeCurrentToJson( root );

    // although json is a textual format, we open the file in binary mode to get exactly the same result on Windows and Linux
    std::ofstream ofs( path, std::ofstream::binary );
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer{ builder.newStreamWriter() };
    if ( !ofs || writer->write( root, &ofs ) != 0 )
    {
        spdlog::error( "Color theme serialization failed: cannot write file {}", utf8string( path ) );
    }

    ofs.close();
}

void ColorTheme::serializeCurrentToJson( Json::Value& root )
{
    assert( ImGui::GetCurrentContext() );
    auto& instance = ColorTheme::instance();

    if ( instance.sceneColors_.size() < SceneColors::Count )
        instance.sceneColors_.resize( SceneColors::Count );

    for ( int i = 0; i < SceneColors::Count; ++i )
        instance.sceneColors_[i] = SceneColors::get( SceneColors::Type( i ) );

    const auto& vpParams = Viewer::instanceRef().viewport().getParameters();
    setViewportColor( vpParams.backgroundColor, ViewportColorsType::Background );
    setViewportColor( vpParams.borderColor, ViewportColorsType::Borders );

    for ( int i = 0; i < SceneColors::Count; ++i )
        serializeToJson( instance.sceneColors_[i], root[SceneColors::getName( SceneColors::Type( i ) )] );

    root["ImGuiPreset"] = getPresetName( instance.themePreset_ );

    auto& ribbonColors = root["Ribbon Colors"];
    for ( int i = 0; i<int( RibbonColorsType::Count ); ++i )
        serializeToJson( instance.newUIColors_[i], ribbonColors[getRibbonColorTypeName( RibbonColorsType( i ) )] );

    auto& vieportColors = root["Viewport Colors"];
    for ( int i = 0; i<int( ViewportColorsType::Count ); ++i )
        serializeToJson( instance.viewportColors_[i], vieportColors[getViewportColorTypeName( ViewportColorsType( i ) )] );
}

void ColorTheme::apply()
{
    spdlog::info( "Apply color theme." );
    assert( ColorTheme::isInitialized() );

    const auto& instance = ColorTheme::instance();

    for ( int i = 0; i < SceneColors::Count; ++i )
        SceneColors::set( SceneColors::Type( i ), instance.sceneColors_[i] );

    RibbonButtonDrawer::InitGradientTexture();
    UI::init();

    CommandLoop::appendCommand( [&] ()
    {
        resetImGuiStyle();

        auto& viewer = getViewerInstance();
        for ( auto& vp : viewer.viewport_list )
        {
            auto params = vp.getParameters();
            params.backgroundColor = getViewportColor( ViewportColorsType::Background );
            params.borderColor = getViewportColor( ViewportColorsType::Borders );
            vp.setParameters( params );
        }
        instance.colorThemeChangedSignal();
    } );
}

bool ColorTheme::isInitialized()
{
    return !ColorTheme::instance().sceneColors_.empty();
}

void ColorTheme::setRibbonColor( const Color& color, RibbonColorsType type )
{
    auto& instance = ColorTheme::instance();
    instance.newUIColors_[int( type )] = color;
}

const Color& ColorTheme::getRibbonColor( RibbonColorsType type )
{
    const auto& instance = ColorTheme::instance();
    return instance.newUIColors_[int( type )];
}

ColorTheme::Preset ColorTheme::getPreset()
{
    return instance().themePreset_;
}

const char* ColorTheme::getPresetName( Preset type )
{
    constexpr std::array<const char*, size_t( Preset::Count )> presetNames =
    {
        "Dark",
        "Light"
    };
    return presetNames[int( type )];
}

ColorTheme::Type ColorTheme::getThemeType()
{
    return instance().type_;
}

const std::string& ColorTheme::getThemeName()
{
    return instance().themeName_;
}

void ColorTheme::setupByTypeName( Type type, const std::string& name )
{
    if ( type == Type::User )
        setupUserTheme( name );
    else if ( name == getPresetName( Preset::Light ) )
        setupDefaultLight();
    else
        setupDefaultDark();
}

void ColorTheme::setupDefaultDark()
{
    spdlog::info( "Setup dark color theme." );
    instance().themeName_ = getPresetName( Preset::Dark );
    setupFromFile( MR::GetResourcesDirectory() / "MRDarkTheme.json", Type::Default );
}

void ColorTheme::setupDefaultLight()
{
    spdlog::info( "Setup light color theme." );
    instance().themeName_ = getPresetName( Preset::Light );
    setupFromFile( MR::GetResourcesDirectory() / "MRLightTheme.json", Type::Default );
}

void ColorTheme::setupUserTheme( const std::string& themeName )
{
    spdlog::info( "Setup user color theme: {}", themeName );
    instance().themeName_ = themeName;
    setupFromFile( getUserThemesDirectory() / ( asU8String( themeName ) + u8".json" ), Type::User );
}

const char* ColorTheme::getRibbonColorTypeName( RibbonColorsType type )
{
    constexpr std::array<const char*, size_t( RibbonColorsType::Count )> colorNames
    {
        "Background",
        "HeaderBackground",
        "HeaderSeparator",
        "TopPanelBackground",
        "QuickAccessBackground",
        "Borders",

        "TabHovered",
        "TabClicked",
        "TabActive",
        "TabActiveHovered",
        "TabActiveClicked",
        "TabText",
        "TabActiveText",

        "DialogTab",
        "DialogTabHovered",
        "DialogTabActive",
        "DialogTabActiveHovered",
        "DialogTabText",
        "DialogTabActiveText",

        "ToolbarHovered",
        "ToolbarClicked",

        "ModalBackground",

        "Text",
        "TextEnabled",
        "TextDisabled",
        "TextSelectedBg",

        "RibbonButtonHovered",
        "RibbonButtonClicked",
        "RibbonButtonActive",
        "RibbonButtonActiveHovered",
        "RibbonButtonActiveClicked",

        "ThirdpartyBackground",
        "ProgressBarBackground",
        "FrameBackground",
        "CollapseHeaderBackground",
        "SelectedObjectText",
        "SelectedObjectFrame",
        "GradientStart",
        "GradientEnd",

        "TextContrastBackground",

        "GradBtnStart",
        "GradBtnHoverStart",
        "GradBtnActiveStart",
        "GradBtnDisableStart",
        "GradBtnEnd",
        "GradBtnHoverEnd",
        "GradBtnActiveEnd",
        "GradBtnDisableEnd",
        "GradBtnText"
    };
    return colorNames[int( type )];
}

void ColorTheme::setViewportColor( const Color& color, ViewportColorsType type )
{
    auto& instance = ColorTheme::instance();
    instance.viewportColors_[int( type )] = color;
}

const Color& ColorTheme::getViewportColor( ViewportColorsType type )
{
    const auto& instance = ColorTheme::instance();
    return instance.viewportColors_[int( type )];
}

const char* ColorTheme::getViewportColorTypeName( ViewportColorsType type )
{
    constexpr std::array<const char*, size_t( ViewportColorsType::Count )> colorNames
    {
        "Background",
        "Borders"
    };
    return colorNames[int( type )];
}

std::filesystem::path ColorTheme::getUserThemesDirectory()
{
    auto configDir = getUserConfigDir();
    configDir /= "UserThemes";
    return configDir;
}

void ColorTheme::resetImGuiStyle()
{
    const auto& instance = ColorTheme::instance();

    auto& style = ImGui::GetStyle();
    style = ImGuiStyle();

    switch ( instance.themePreset_ )
    {
        case MR::ColorTheme::Preset::Light:
            ImGui::StyleColorsLight();
            break;
        case MR::ColorTheme::Preset::Dark:
        default:
            ImGui::StyleColorsDark();
            break;
    }

    Vector4f bg = Vector4f( getRibbonColor( RibbonColorsType::Background ) );
    Vector4f text = Vector4f( getRibbonColor( RibbonColorsType::Text ) );
    Vector4f border = Vector4f( getRibbonColor( RibbonColorsType::Borders ) );
    Vector4f frameBg = Vector4f( getRibbonColor( RibbonColorsType::FrameBackground ) );
    Vector4f headerBg = Vector4f( getRibbonColor( RibbonColorsType::CollapseHeaderBackground ) );
    Vector4f textSelBg = Vector4f( getRibbonColor( RibbonColorsType::TextSelectedBg ) );
    Vector4f popupBg = Vector4f( getRibbonColor( RibbonColorsType::ModalBackground ) );
    Vector4f tabBg = Vector4f( getRibbonColor( RibbonColorsType::DialogTab ) );
    Vector4f tabBgActive = Vector4f( getRibbonColor( RibbonColorsType::DialogTabActive ) );
    Vector4f tabBgHovered = Vector4f( getRibbonColor( RibbonColorsType::DialogTabActiveHovered ) );

    style.Colors[ImGuiCol_WindowBg] = ImVec4( bg.x, bg.y, bg.z, bg.w );
    style.Colors[ImGuiCol_Text] = ImVec4( text.x, text.y, text.z, text.w );
    style.Colors[ImGuiCol_Border] = ImVec4( border.x, border.y, border.z, border.w );
    style.Colors[ImGuiCol_FrameBg] = ImVec4( frameBg.x, frameBg.y, frameBg.z, frameBg.w );
    style.Colors[ImGuiCol_Header] = ImVec4( headerBg.x, headerBg.y, headerBg.z, headerBg.w );
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4( textSelBg.x, textSelBg.y, textSelBg.z, textSelBg.w );
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4( 0, 0, 0, 0 );
    style.Colors[ImGuiCol_PopupBg] = ImVec4( popupBg.x, popupBg.y, popupBg.z, popupBg.w );
    style.Colors[ImGuiCol_Tab] = ImVec4( tabBg.x, tabBg.y, tabBg.z, tabBg.w );
    style.Colors[ImGuiCol_TabActive] = ImVec4( tabBgActive.x, tabBgActive.y, tabBgActive.z, tabBgActive.w );
    style.Colors[ImGuiCol_TabHovered] = ImVec4( tabBgHovered.x, tabBgHovered.y, tabBgHovered.z, tabBgHovered.w );

    style.ScrollbarRounding = 4.0f;
    style.FrameRounding = 5.0f;
    style.GrabRounding = 3.0f;
    style.GrabMinSize = 16.0f;
    style.FramePadding.y = 5.0f;
    style.ItemSpacing.y = 6.0f;

    style.FrameBorderSize = 1.0f;
    style.AntiAliasedLines = false;

    style.WindowBorderSize = 1.0f;
    
    if ( auto menu = getViewerInstance().getMenuPlugin() )
    {
        auto scaling = menu->menu_scaling();
        ImGui::GetStyle().ScaleAllSizes( scaling );
        style.ScrollbarSize = 4.0f * scaling + 6.0f; // 6 - is scroll background area, independent of scaling
    }
}

void ColorTheme::updateUserThemesList()
{
    auto& instance = ColorTheme::instance();
    instance.foundUserThemes_.clear();

    auto userThemesDir = getUserThemesDirectory();
    std::error_code ec;
    if ( !std::filesystem::is_directory( userThemesDir, ec ) )
        return;

    for ( auto entry : Directory{ userThemesDir, ec } )
    {
        if ( !entry.is_regular_file( ec ) )
            continue;

        auto ext = entry.path().extension().u8string();
        for ( auto& c : ext )
            c = (char)tolower( c );

        if ( ext != u8".json" )
            continue;

        instance.foundUserThemes_.emplace_back( utf8string( entry.path().stem() ) );
    }
}

std::vector<std::string> ColorTheme::foundUserThemes()
{
    const auto& instance = ColorTheme::instance();
    return instance.foundUserThemes_;
}

}