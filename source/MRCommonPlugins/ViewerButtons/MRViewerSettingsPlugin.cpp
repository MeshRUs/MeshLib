#include "MRViewerSettingsPlugin.h"
#include "MRViewer/MRRibbonMenu.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRColorTheme.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRMouseController.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRViewer/MRViewerSettingsManager.h"
#include "MRViewer/MRGLMacro.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/MRViewer.h"
#include "MRViewer/MRImGuiVectorOperators.h"
#include "MRMesh/MRSystem.h"
#include "MRViewer/MRSpaceMouseHandlerHidapi.h"
#include "MRMesh/MRLog.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewer/MRUIStyle.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRMesh/MRDirectory.h"
#include <MRMesh/MRSceneRoot.h>
#include <MRViewer/MRFileDialog.h>
#include "MRMesh/MRObjectMesh.h"
#include "MRViewer/MRRibbonSceneObjectsListDrawer.h"

namespace
{
const char* getViewerSettingTabName( MR::ViewerSettingsPlugin::TabType tab )
{
    constexpr std::array<const char*, size_t( MR::ViewerSettingsPlugin::TabType::Count )> tabNames{
        "Quick",
        "Application",
        "Control",
        "3D View",
        "Units",
        "Features",
    };
    return tabNames[int( tab )];
}
}

namespace MR
{

ViewerSettingsPlugin::ViewerSettingsPlugin() :
    StatePlugin( "Viewer settings" )
{
    shadowGl_ = std::make_unique<ShadowsGL>();
    CommandLoop::appendCommand( [maxSamples = &maxSamples_, curSamples = &curSamples_, storedSamples = &storedSamples_] ()
    {
        if ( getViewerInstance().isGLInitialized() && loadGL() )
        {
            GL_EXEC( glGetIntegerv( GL_MAX_SAMPLES, maxSamples ) );
            GL_EXEC( glGetIntegerv( GL_SAMPLES, curSamples ) );
            *maxSamples = std::max( std::min( *maxSamples, 16 ), *curSamples ); // there are some known issues with 32 MSAA
            *storedSamples = *curSamples;
        }
    } );
#ifndef __EMSCRIPTEN__
    CommandLoop::appendCommand( [&] ()
    {
        auto& viewer = getViewerInstance();
        int samples = 0;
        if ( auto& settingsManager = viewer.getViewportSettingsManager() )
            samples = settingsManager->loadInt( "multisampleAntiAliasing", 8 );
        if ( viewer.isGLInitialized() && loadGL() )
        {
            int realSamples;
            GL_EXEC( glGetIntegerv( GL_SAMPLES, &realSamples ) );
            gpuOverridesMSAA_ = ( realSamples != samples );
        }
    }, CommandLoop::StartPosition::AfterWindowAppear );
#endif
}

const std::string& ViewerSettingsPlugin::uiName() const
{
    static std::string name = std::string( "Settings" ) + UINameSuffix();
    return name;
}

void ViewerSettingsPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 500.0f * menuScaling;

    ImVec2 position{ ( viewer->framebufferSize.x - menuWidth ) / 2, viewer->framebufferSize.y / 6.0f };
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .position = &position, .menuScaling = menuScaling } ) )
        return;

    if ( UI::beginTabBar( "##MainTabs" ) )
    {
        for ( int i = 0; i<int( TabType::Count ); ++i )
        {
            if ( i == int( TabType::Features ) && !viewer->experimentalFeatures )
            {
                // Check if features are considered experimental
                const auto& tabs = RibbonSchemaHolder::schema().tabsOrder;
                auto itFeatures = std::find_if( tabs.begin(), tabs.end(),
                    [] ( const RibbonTab& tab ) { return tab.name == "Features"; } );
                if ( itFeatures != tabs.end() && itFeatures->experimental )
                    continue;
            }
            auto tab = TabType( i );
            if ( UI::beginTabItem( getViewerSettingTabName( tab ) ) )
            {
                activeTab_ = tab;
                drawTab_( tab, menuWidth, menuScaling );
                drawCustomSettings_( tab, menuScaling );
                UI::endTabItem();
            }
        }
        UI::endTabBar();
    }

    ImGui::EndCustomStatePlugin();
}

void ViewerSettingsPlugin::updateThemes()
{
    selectedUserPreset_ = -1;
    userThemesPresets_.clear();
    userThemesPresets_.push_back( "Dark" );
    userThemesPresets_.push_back( "Light" );
    auto colorThemeType = ColorTheme::getThemeType();
    auto colorThemeName = ColorTheme::getThemeName();
    if ( colorThemeType == ColorTheme::Type::Default )
    {
        if ( colorThemeName == ColorTheme::getPresetName( ColorTheme::Preset::Light ) )
            selectedUserPreset_ = 1;
        else
            selectedUserPreset_ = 0;
    }

    auto userThemesDir = ColorTheme::getUserThemesDirectory();
    int i = int( userThemesPresets_.size() );
    std::error_code ec;
    if ( std::filesystem::is_directory( userThemesDir, ec ) )
    {
        for ( auto entry : Directory{ userThemesDir, ec } )
        {
            if ( entry.is_regular_file( ec ) )
            {
                auto ext = entry.path().extension().u8string();
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext != u8".json" )
                    break;
                std::string themeName = utf8string( entry.path().stem() );
                userThemesPresets_.push_back( themeName );
                if ( selectedUserPreset_ == -1 && themeName == ColorTheme::getThemeName() )
                    selectedUserPreset_ = i;
                ++i;
            }
        }
    }
}

void ViewerSettingsPlugin::addComboSettings( const TabType tab, std::shared_ptr<ExternalSettings> settings )
{
    comboSettings_[size_t( tab )].push_back( settings );
}

bool ViewerSettingsPlugin::onEnable_()
{
    backgroundColor_.w = -1.0f;

    ribbonMenu_ = getViewerInstance().getMenuPluginAs<RibbonMenu>().get();

    updateDialog_();

    return true;
}

bool ViewerSettingsPlugin::onDisable_()
{
    userThemesPresets_.clear();
    ribbonMenu_ = nullptr;
    return true;
}

void ViewerSettingsPlugin::drawTab_( TabType tab, float menuWidth, float menuScaling )
{
    switch ( tab )
    {
    case MR::ViewerSettingsPlugin::TabType::Quick:
        drawQuickTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Application:
        drawApplicationTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Control:
        drawControlTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Viewport:
        drawViewportTab_( menuWidth, menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::MeasurementUnits:
        drawMeasurementUnitsTab_( menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Features:
        drawFeaturesTab_( menuScaling );
        break;
    case MR::ViewerSettingsPlugin::TabType::Count:
    default:
        break;
    }
}

void ViewerSettingsPlugin::drawQuickTab_( float menuWidth, float menuScaling )
{
    UI::separator( menuScaling, "General" );

    drawThemeSelector_( menuWidth, menuScaling );
    drawProjectionModeSelector_( menuScaling );
    drawShadingModeCombo_( false, menuScaling );
    drawBackgroundButton_( true );

    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;
    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->openToolbarCustomize();
    ImGui::SameLine();
    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->setShowShortcuts( true );

    drawMouseSceneControlsSettings_( menuWidth, menuScaling );
}

void ViewerSettingsPlugin::drawApplicationTab_( float menuWidth, float menuScaling )
{
    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;

    UI::separator( menuScaling, "Interface" );

    drawThemeSelector_( menuWidth, menuScaling );

    bool savedDialogsBackUp = viewer->getMenuPlugin()->isSavedDialogPositionsEnabled();
    bool savedDialogsVal = savedDialogsBackUp;
    UI::checkbox( "Save Tool Window Positions", &savedDialogsVal );
    UI::setTooltipIfHovered( "If checked then enables using of saved positions of tool windows in the config file", menuScaling );
    if ( savedDialogsVal != savedDialogsBackUp )
        viewer->getMenuPlugin()->enableSavedDialogPositions( savedDialogsVal );

    if ( UI::button( "Toolbar Customize", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->openToolbarCustomize();

    UI::separator( menuScaling, "Behavior" );

    ImGui::SetNextItemWidth( menuWidth * 0.5f );
    if ( ribbonMenu_ )
    {
        auto sceneObjectsListDrawer = ribbonMenu_->getSceneObjectsList();
        if ( sceneObjectsListDrawer )
        {
            UI::checkbox( "Make Visible on Select",
                                                    std::bind( &SceneObjectsListDrawer::getShowNewSelectedObjects, sceneObjectsListDrawer ),
                                                    std::bind( &SceneObjectsListDrawer::setShowNewSelectedObjects, sceneObjectsListDrawer, std::placeholders::_1 ) );
            UI::checkbox( "Deselect on Hide",
                                                    std::bind( &SceneObjectsListDrawer::getDeselectNewHiddenObjects, sceneObjectsListDrawer ),
                                                    std::bind( &SceneObjectsListDrawer::setDeselectNewHiddenObjects, sceneObjectsListDrawer, std::placeholders::_1 ) );
            UI::checkbox( "Show Info in Object Tree",
                                                    std::bind( &SceneObjectsListDrawer::getShowInfoInObjectTree, sceneObjectsListDrawer ),
                                                    std::bind( &SceneObjectsListDrawer::setShowInfoInObjectTree, sceneObjectsListDrawer, std::placeholders::_1 ) );
            UI::setTooltipIfHovered( "Show detailed information in the object tree", menuScaling );

            std::shared_ptr<RibbonSceneObjectsListDrawer> ribbonSceneObjectsListDrawer = std::dynamic_pointer_cast< RibbonSceneObjectsListDrawer >( sceneObjectsListDrawer );
            if ( ribbonSceneObjectsListDrawer )
            {
                UI::checkbox( "Close Context Menu on Click",
                                                        std::bind( &RibbonSceneObjectsListDrawer::getCloseContextOnChange, ribbonSceneObjectsListDrawer ),
                                                        std::bind( &RibbonSceneObjectsListDrawer::setCloseContextOnChange, ribbonSceneObjectsListDrawer, std::placeholders::_1 ) );
                UI::setTooltipIfHovered( "Close scene context menu on any change or click outside", menuScaling );
            }
        }

        UI::checkbox( "Auto Close Previous Tool",
                                                std::bind( &RibbonMenu::getAutoCloseBlockingPlugins, ribbonMenu_ ),
                                                std::bind( &RibbonMenu::setAutoCloseBlockingPlugins, ribbonMenu_, std::placeholders::_1 ) );
        UI::setTooltipIfHovered( "Automatically close blocking tool when another blocking tool is activated", menuScaling );

        UI::checkbox( "Show Experimental Features", &viewer->experimentalFeatures );
        UI::setTooltipIfHovered( "Show experimental or diagnostic tools and controls", menuScaling );
    }

    UI::separator( menuScaling, "Global" );

    bool resetClicked = UI::button( "Reset Settings", Vector2f( btnHalfSizeX, 0 ) );
    drawResetDialog_( resetClicked, menuScaling );

#if 0 // Hide unimplemented settings
    if ( !viewer->experimentalFeatures )
        return; // TODO

    UI::separator( menuScaling, "Notifications" );

    static bool newVersion, importWarnings; // TODO
    UI::checkbox( "New application version", &newVersion );
    UI::setTooltipIfHovered( "Show when a new version of MeshInspector is available.", menuScaling );
    UI::checkbox( "Import warnings", &importWarnings );
    UI::setTooltipIfHovered( "Non-fatal warnings when importing a file", menuScaling );

#ifndef __EMSCRIPTEN__
    UI::separator( menuScaling, "Files and Folders" );
    // TODO
    static std::string logFolderPath = Logger::instance().getLogFileName().parent_path().string();
    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { 1.5f * cButtonPadding * menuScaling, cButtonPadding * menuScaling } );
    ImGui::InputText( "##LogFolderPath", logFolderPath, 0 );
    ImGui::SameLine( 0, 1.5f * style.ItemInnerSpacing.x );
    if ( ImGui::Link( "Logs folder") )
        OpenDocument( logFolderPath );
    ImGui::PopStyleVar();
    ImGui::SameLine( 0.0f, 0.0f );
    ImGui::SameLine( 0.0f, -30.0f * menuScaling );
    if ( UI::button( "...", ImVec2( 24.0f * menuScaling, 24.0f * menuScaling ) ) )
    {
        std::filesystem::path newPath = openFolderDialog( logFolderPath );
        if ( !newPath.empty() )
            logFolderPath = newPath.string();
    }
#endif
#endif
}

void ViewerSettingsPlugin::drawControlTab_( float menuWidth, float menuScaling )
{
    UI::separator( menuScaling, "Keyboard" );

    auto& style = ImGui::GetStyle();
    const float btnHalfSizeX = ( menuWidth - style.WindowPadding.x * 2 - style.ItemSpacing.x ) / 2.f;
    if ( UI::button( "Show Hotkeys", Vector2f( btnHalfSizeX, 0 ) ) && ribbonMenu_ )
        ribbonMenu_->setShowShortcuts( true );

    drawMouseSceneControlsSettings_( menuWidth, menuScaling );
    drawTouchpadSettings_( menuScaling );
    drawSpaceMouseSettings_( menuWidth, menuScaling );
}

void ViewerSettingsPlugin::drawViewportTab_( float menuWidth, float menuScaling )
{
    const auto& viewportParameters = viewer->viewport().getParameters();

    UI::separator( menuScaling, "Viewport" );
    if ( viewer->viewport_list.size() > 1 )
        ImGui::Text( "Current viewport: %d", viewer->viewport().id.value() );

    ImGui::SetNextItemWidth( 140.0f * menuScaling );
    auto rotMode = viewportParameters.rotationMode;
    UI::combo( "Rotation Mode", ( int* )&rotMode, { "Scene Center", "Pick / Scene Center", "Pick" } );
    viewer->viewport().rotationCenterMode( rotMode );

    ImGui::PushItemWidth( 80 * menuScaling );

    bool showAxes = viewer->basisAxes->isVisible( viewer->viewport().id );
    UI::checkbox( "Show Axes", &showAxes );
    viewer->viewport().showAxes( showAxes );

    bool showGlobalBasis = viewer->globalBasisAxes->isVisible( viewer->viewport().id );
    UI::checkbox( "Show Global Basis", &showGlobalBasis );
    viewer->viewport().showGlobalBasis( showGlobalBasis );

    bool showRotCenter = viewer->rotationSphere->isVisible( viewer->viewport().id );
    UI::checkbox( "Show Rotation Center", &showRotCenter );
    viewer->viewport().showRotationCenter( showRotCenter );

    ImGui::PopItemWidth();

    drawProjectionModeSelector_( menuScaling );

    drawBackgroundButton_( false );

    auto coef = SceneSettings::get( SceneSettings::FloatType::AmbientCoefSelectedObj );
    ImGui::SetNextItemWidth( 140.0f * menuScaling );
    if ( UI::drag<NoUnit>( "Selected Highlight Modifier", coef, 0.01f, 1.0f, 10.0f ) )
    {
        SceneSettings::set( SceneSettings::FloatType::AmbientCoefSelectedObj, coef );
    }

    UI::setTooltipIfHovered( "Ambient light brightness multiplication factor for selected objects", menuScaling );

    if ( viewer->experimentalFeatures &&
        RibbonButtonDrawer::CustomCollapsingHeader( "Clipping Plane" ) )
    {
        auto plane = viewportParameters.clippingPlane;
        auto showPlane = viewer->clippingPlaneObject->isVisible( viewer->viewport().id );
        plane.n = plane.n.normalized();
        auto w = ImGui::GetContentRegionAvail().x;
        ImGui::SetNextItemWidth( w );
        UI::drag<NoUnit>( "##ClippingPlaneNormal", plane.n, 1e-3f );
        ImGui::SetNextItemWidth( w / 2.0f );
        UI::drag<NoUnit>( "##ClippingPlaneD", plane.d, 1e-3f );
        ImGui::SameLine();
        UI::checkbox( "Show##ClippingPlane", &showPlane );
        viewer->viewport().setClippingPlane( plane );
        viewer->viewport().showClippingPlane( showPlane );
    }

    UI::separator( menuScaling, "Options" );

    ImGui::SetNextItemWidth( 100.0f * menuScaling );
    int pickRadius = int( getViewerInstance().glPickRadius );
    UI::drag<PixelSizeUnit>( "Picker Radius", pickRadius, 1, 0, 10 );
    getViewerInstance().glPickRadius = uint16_t( pickRadius );
    UI::setTooltipIfHovered( "Radius of area under cursor to pick objects in scene.", menuScaling );

    UI::separator( menuScaling, "Defaults" );

    drawShadingModeCombo_( true, menuScaling );
    drawUpDirectionSelector_();

    UI::separator( menuScaling, "Render" );

    drawRenderOptions_( menuScaling );
    drawShadowsOptions_( menuWidth, menuScaling );
}

void ViewerSettingsPlugin::drawMeasurementUnitsTab_( float menuScaling )
{
    (void)menuScaling;

    auto paramsNoUnit = getDefaultUnitParams<NoUnit>();
    auto paramsLen = getDefaultUnitParams<LengthUnit>();
    auto paramsArea = getDefaultUnitParams<AreaUnit>();
    auto paramsVol = getDefaultUnitParams<VolumeUnit>();
    auto paramsMoveSpeed = getDefaultUnitParams<MovementSpeedUnit>();
    auto paramsAngle = getDefaultUnitParams<AngleUnit>();
    auto paramsTime = getDefaultUnitParams<TimeUnit>();
    auto paramsRatio = getDefaultUnitParams<RatioUnit>();
    auto paramsPixelSize = getDefaultUnitParams<PixelSizeUnit>();

    auto forAllLengthParams = [&]( auto&& func )
    {
        func( paramsLen );
        func( paramsArea );
        func( paramsVol );
        func( paramsMoveSpeed );
    };
    auto forAllParams = [&]( auto&& func )
    {
        func( paramsNoUnit );
        forAllLengthParams( func );
        func( paramsAngle );
        func( paramsTime );
        func( paramsRatio );
        func( paramsPixelSize );
    };

    auto applyParams = [&]
    {
        forAllParams( []( const auto& params ){ setDefaultUnitParams( params ); } );
    };


    { // Common.
        UI::separator( menuScaling, "Common" );

        // --- Leading zero

        if ( UI::checkbox( "Leading zero", &paramsLen.leadingZero ) )
        {
            forAllParams( [&]( auto& params ){ params.leadingZero = paramsLen.leadingZero; } );
            applyParams();
        }
        ImGui::SetItemTooltip( "If disabled, remove the lone zeroes before the decimal point." );

        // --- Thousands separator

        auto thousandsSeparator = [&]( bool frac )
        {
            ImGui::PushItemWidth( std::floor( ImGui::CalcItemWidth() / 2 ) );
            MR_FINALLY{ ImGui::PopItemWidth(); };

            char thouSep[2] = { frac ? paramsNoUnit.thousandsSeparatorFrac : paramsNoUnit.thousandsSeparator, '\0' };
            ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( std::floor( ( ImGui::CalcItemWidth() - ImGui::CalcTextSize( thouSep ).x ) / 2 ), ImGui::GetStyle().FramePadding.y ) );
            MR_FINALLY{ ImGui::PopStyleVar(); };
            if ( ImGui::InputText( frac ? "Thousands separator (fractional part)" : "Thousands separator", thouSep, sizeof thouSep, ImGuiInputTextFlags_AutoSelectAll ) )
            {
                forAllParams( [&]( auto& params )
                {
                    if ( frac )
                        params.thousandsSeparatorFrac = thouSep[0];
                    else
                        params.thousandsSeparator = thouSep[0];
                } );
                applyParams();
            }
            // If the separator is empty or a space, display a string explaining that on top of the textbox.
            if ( !ImGui::IsItemActive() )
            {
                const char* label = nullptr;
                if ( thouSep[0] == 0 )
                    label = "None";
                else if ( thouSep[0] == ' ' )
                    label = "Space";

                if ( label )
                {
                    ImVec2 textSize = ImGui::CalcTextSize( label );
                    ImGui::GetWindowDrawList()->AddText( ImGui::GetItemRectMin() + ( ImVec2( ImGui::CalcItemWidth(), ImGui::GetItemRectSize().y ) - textSize ) / 2, ImGui::GetColorU32( ImGuiCol_TextDisabled ), label );
                }
            }
        };
        thousandsSeparator( false ); // Integral part.
        // thousandsSeparator( true ); // Fractional part.
    }


    { // Length.
        UI::separator( menuScaling, "Linear" );

        ImGui::PushID( "length" );
        MR_FINALLY{ ImGui::PopID(); };

        // --- Units

        static const std::vector<std::string> optionNames = []{
            std::vector<std::string> ret;
            ret.reserve( std::size_t( LengthUnit::_count ) );
            for ( std::size_t i = 0; i < std::size_t( LengthUnit::_count ); i++ )
                ret.push_back( std::string( getUnitInfo( LengthUnit( i ) ).prettyName ) );
            return ret;
        }();

        int option = int( paramsLen.targetUnit );
        if ( UI::combo( "Unit##length", &option, optionNames ) )
        {
            paramsLen.targetUnit = LengthUnit( option );

            switch ( paramsLen.targetUnit )
            {
            case LengthUnit::mm:
                forAllParams( [&]( auto&& params ){ params.leadingZero = true; } );
                paramsArea.targetUnit = AreaUnit::mm2;
                paramsVol.targetUnit = VolumeUnit::mm3;
                paramsMoveSpeed.targetUnit = MovementSpeedUnit::mmPerSecond;
                break;
            case LengthUnit::inches:
                forAllParams( [&]( auto&& params ){ params.leadingZero = false; } );
                paramsArea.targetUnit = AreaUnit::inches2;
                paramsVol.targetUnit = VolumeUnit::inches3;
                paramsMoveSpeed.targetUnit = MovementSpeedUnit::inchesPerSecond;
                break;
            case LengthUnit::_count:; // MSVC warns otherwise.
                break;
            }

            applyParams();
        }

        // --- Precision

        if ( UI::drag<NoUnit>( "Precision##length", paramsLen.precision, 1, 0, 12 ) )
        {
            forAllLengthParams( [&]( auto& params ){ params.precision = paramsLen.precision; } );
            applyParams();
        }
    }

    { // Angle.
        UI::separator( menuScaling, "Angular" );

        static const std::vector<std::string> flavorOptions = { "Degrees", "Degrees, minutes", "Degrees, minutes, seconds" };
        static_assert( int( DegreesMode::degrees ) == 0 );
        static_assert( int( DegreesMode::degreesMinutes ) == 1 );
        static_assert( int( DegreesMode::degreesMinutesSeconds ) == 2 );

        int flavorOption = int( paramsAngle.degreesMode );

        // Degree mode.

        if ( UI::combo( "Unit##angle", &flavorOption, flavorOptions ) )
        {
            DegreesMode newMode = DegreesMode( flavorOption );

            if ( ( paramsAngle.degreesMode == DegreesMode::degrees ) != ( newMode == DegreesMode::degrees ) )
            {
                if ( newMode == DegreesMode::degrees )
                {
                    paramsAngle.style = NumberStyle::normal;
                    paramsAngle.precision = 1;
                }
                else
                {
                    paramsAngle.style = NumberStyle::normal;
                    paramsAngle.precision = 0;
                }
            }

            paramsAngle.degreesMode = newMode;

            applyParams();
        }

        // Degree-mode-specific options.

        if ( paramsAngle.degreesMode == DegreesMode::degrees )
        {
            // --- Precision

            if ( UI::drag<NoUnit>( "Precision##angle", paramsAngle.precision, 1, 0, 12 ) )
                applyParams();
        }
    }
}

void ViewerSettingsPlugin::drawFeaturesTab_( float menuScaling )
{
    (void)menuScaling;

    UI::separator( menuScaling, "Visuals" );
    float value = 0;

    value = SceneSettings::get( SceneSettings::FloatType::FeatureMeshAlpha );
    if ( UI::slider<NoUnit>( "Surface opacity", value, 0.f, 1.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureMeshAlpha, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeaturePointSize );
    if ( UI::slider<PixelSizeUnit>( "Point size", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeaturePointSize, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureSubPointSize );
    if ( UI::slider<PixelSizeUnit>( "Point size (subfeatures)", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureSubPointSize, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureLineWidth );
    if ( UI::slider<PixelSizeUnit>( "Line width", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureLineWidth, value );

    value = SceneSettings::get( SceneSettings::FloatType::FeatureSubLineWidth );
    if ( UI::slider<PixelSizeUnit>( "Line width (subfeatures)", value, 1.f, 20.f ) )
        SceneSettings::set( SceneSettings::FloatType::FeatureSubLineWidth, value );
}

void ViewerSettingsPlugin::drawRenderOptions_( float menuScaling )
{

    if ( viewer->isAlphaSortAvailable() )
    {
        bool alphaSortBackUp = viewer->isAlphaSortEnabled();
        bool alphaBoxVal = alphaSortBackUp;
        UI::checkbox( "Alpha Sort", &alphaBoxVal );
        if ( alphaBoxVal != alphaSortBackUp )
            viewer->enableAlphaSort( alphaBoxVal );
    }

    if ( viewer->isGLInitialized() )
    {
        if ( maxSamples_ > 1 )
        {
#ifdef __EMSCRIPTEN__
            (void)menuScaling;
            ImGui::Text( "Multisample anti-aliasing (MSAA): x%d", curSamples_ );
#else
            auto& style = ImGui::GetStyle();
            auto backUpSamples = storedSamples_;
            ImGui::Text( "Multisample anti-aliasing (MSAA):" );
            UI::setTooltipIfHovered( "The number of samples per pixel: more samples - better render quality but worse performance.", menuScaling );
            int counter = 0;
            for ( int i = 0; i <= maxSamples_; i <<= 1 )
            {
                if ( i == 0 )
                {
                    UI::radioButton( "Off", &storedSamples_, i );
                    ++i;
                }
                else
                {
                    std::string label = 'x' + std::to_string( i );
                    UI::radioButton( label.c_str(), &storedSamples_, i );
                }
                if ( i << 1 <= maxSamples_ )
                    ImGui::SameLine( ( ( ++counter ) * 70.f + style.WindowPadding.x ) * menuScaling );
            }
            if ( backUpSamples != storedSamples_ )
            {
                if ( auto& settingsManager = viewer->getViewportSettingsManager() )
                    settingsManager->saveInt( "multisampleAntiAliasing", storedSamples_ );

                needReset_ = storedSamples_ != curSamples_;
            }
            if ( gpuOverridesMSAA_ )
                UI::transparentTextWrapped( "GPU multisampling settings override application value." );
            if ( needReset_ )
                UI::transparentTextWrapped( "Application requires restart to apply this change" );
#endif
        }
    }
}

void ViewerSettingsPlugin::drawShadowsOptions_( float menuWidth, float menuScaling )
{
    if ( shadowGl_ && RibbonButtonDrawer::CustomCollapsingHeader( "Shadows" ) )
    {
        auto& style = ImGui::GetStyle();
        bool isEnableShadows = shadowGl_->isEnabled();
        UI::checkbox( "Enabled", &isEnableShadows );
        if ( isEnableShadows != shadowGl_->isEnabled() )
        {
            CommandLoop::appendCommand( [shadowGl = shadowGl_.get(), isEnableShadows] ()
            {
                shadowGl->enable( isEnableShadows );
            } );
        }
        ImGui::SameLine( menuWidth * 0.25f + style.WindowPadding.x + 2 * menuScaling );
        auto color = shadowGl_->getShadowColor();
        UI::colorEdit4( "Shadow Color", color,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel );
        shadowGl_->setShadowColor( color );

        ImGui::PushItemWidth( menuWidth * 0.5f );
        auto shift = shadowGl_->getShadowShift();
        auto radius = shadowGl_->getBlurRadius();
        auto quality = shadowGl_->getQuality();
        UI::drag<PixelSizeUnit>( "Shift", shift, 0.4f, -200.0f, 200.0f );
        ImGui::SetItemTooltip( "X = shift to the left, Y = shift upwards" );
        UI::drag<PixelSizeUnit>( "Blur Radius", radius, 0.2f, 0.f, 200.f );
        UI::drag<NoUnit>( "Quality", quality, 0.001f, 0.0625f, 1.0f );
        ImGui::PopItemWidth();
        UI::setTooltipIfHovered( "Blur texture downscaling coefficient", menuScaling );
        shadowGl_->setShadowShift( shift );
        shadowGl_->setBlurRadius( radius );
        shadowGl_->setQuality( quality );
    }
}

void ViewerSettingsPlugin::drawThemeSelector_( float menuWidth, float menuScaling )
{
    ImGui::SetNextItemWidth( menuWidth * 0.5f );
    int selectedUserIdxBackup = selectedUserPreset_;
    UI::combo( "Color theme", &selectedUserPreset_, userThemesPresets_ );
    if ( selectedUserPreset_ != selectedUserIdxBackup )
    {
        if ( selectedUserPreset_ == 0 )
            ColorTheme::setupDefaultDark();
        else if ( selectedUserPreset_ == 1 )
            ColorTheme::setupDefaultLight();
        else
            ColorTheme::setupUserTheme( userThemesPresets_[selectedUserPreset_] );
        if ( !ColorTheme::isInitialized() )
        {
            if ( selectedUserIdxBackup == 0 )
                ColorTheme::setupDefaultDark();
            else if ( selectedUserIdxBackup == 1 )
                ColorTheme::setupDefaultLight();
            else
                ColorTheme::setupUserTheme( userThemesPresets_[selectedUserIdxBackup] );
            selectedUserPreset_ = selectedUserIdxBackup;

            showError( "This theme is not valid." );
        }
        backgroundColor_ = Vector4f( ColorTheme::getViewportColor( ColorTheme::ViewportColorsType::Background ) );
        ColorTheme::apply();
    }
    auto item = RibbonSchemaHolder::schema().items.find( "Add custom theme" );
    if ( item != RibbonSchemaHolder::schema().items.end() )
    {
        ImGui::SameLine( menuWidth * 0.75f );
        if ( UI::button( "Add",
            item->second.item->isAvailable( getAllObjectsInTree<const Object>( &SceneRoot::get(), ObjectSelectivityType::Selected ) ).empty(),
            Vector2f( menuWidth * 0.20f, 0 ) ) )
        {
            item->second.item->action();
        }
        UI::setTooltipIfHovered( item->second.tooltip, menuScaling );
    }
}

void ViewerSettingsPlugin::drawResetDialog_( bool activated, float menuScaling )
{
    if ( activated )
        ImGui::OpenPopup( "Settings reset" );
    const ImVec2 windowSize{ cModalWindowWidth * menuScaling, -1 };
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, { cModalWindowPaddingX * menuScaling, cModalWindowPaddingY * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { 2.0f * cDefaultItemSpacing * menuScaling, 3.0f * cDefaultItemSpacing * menuScaling } );
    if ( ImGui::BeginModalNoAnimation( "Settings reset", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar ) )
    {
        std::string text = "Reset all application settings?";
        const float textWidth = ImGui::CalcTextSize( text.c_str() ).x;
        ImGui::SetCursorPosX( ( windowSize.x - textWidth ) * 0.5f );
        ImGui::Text( "%s", text.c_str() );

        const auto style = ImGui::GetStyle();
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { style.FramePadding.x, cButtonPadding * menuScaling } );

        const float p = ImGui::GetStyle().ItemSpacing.x;
        const Vector2f btnSize{ ( ImGui::GetContentRegionAvail().x - p ) / 2.f, 0 };

        if ( UI::buttonCommonSize( "Reset", btnSize, ImGuiKey_Enter ) )
        {
            resetSettings_();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine( 0, p );
        if ( UI::buttonCommonSize( "Cancel", btnSize, ImGuiKey_Escape ) )
            ImGui::CloseCurrentPopup();

        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar( 2 );
}

void ViewerSettingsPlugin::drawShadingModeCombo_( bool inGroup, float menuScaling )
{
    static std::vector<std::string> shadingModes = { "Auto Detect", "Smooth", "Flat" };
    SceneSettings::ShadingMode shadingMode = SceneSettings::getDefaultShadingMode();
    ImGui::SetNextItemWidth( 140.0f * menuScaling );
    UI::combo( inGroup ? "Shading Mode" : "Default Shading Mode", ( int* )&shadingMode, shadingModes);
    UI::setTooltipIfHovered( "Shading mode for mesh objects imported from files\n"
        "Detection depends on source format and mesh shape\n"
        "This setting also affects some tools", menuScaling );
    if ( shadingMode != SceneSettings::getDefaultShadingMode() )
        SceneSettings::setDefaultShadingMode( shadingMode );
}

void ViewerSettingsPlugin::drawProjectionModeSelector_( float menuScaling )
{
    ImGui::SetNextItemWidth( 140.0f * menuScaling );
    static std::vector<std::string> projectionModes = { "Orthographic", "Perspective" };
    int projectionMode = viewer->viewport().getParameters().orthographic ? 0 : 1;
    if ( UI::combo( "Projection Mode", &projectionMode, projectionModes) )
        viewer->viewport().setOrthographic( projectionMode == 0 );
}

void ViewerSettingsPlugin::drawUpDirectionSelector_()
{
#if 0 // Hide unimplemented settings
    // TODO
    if ( !viewer->experimentalFeatures )
        return;
    ImGui::Text( "Up Direction" );
    static int axis = 2; // Z
    ImGui::SameLine();
    UI::radioButton( "Y", &axis, 1 );
    ImGui::SameLine();
    UI::radioButton( "Z", &axis, 2 );
#endif
}

void ViewerSettingsPlugin::drawBackgroundButton_( bool allViewports )
{
    bool needUpdateBackup = backgroundColor_.w == -1.0f;
    if ( needUpdateBackup )
        backgroundColor_ = Vector4f( viewer->viewport().getParameters().backgroundColor );

    auto backgroundColor = backgroundColor_;

    if ( UI::colorEdit4( "Background Color", backgroundColor,
        ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel ) )
        backgroundColor_ = backgroundColor;
    else if ( ImGui::IsWindowFocused() || !ImGui::IsWindowFocused( ImGuiFocusedFlags_ChildWindows ) )
        backgroundColor_.w = -1.0f;
    if ( allViewports )
        for ( ViewportId vid : viewer->getPresentViewports() )
            viewer->viewport( vid ).setBackgroundColor( Color( backgroundColor ) );
    else
        viewer->viewport().setBackgroundColor( Color( backgroundColor ) );
}

void ViewerSettingsPlugin::drawMouseSceneControlsSettings_( float menuWidth, float menuScaling )
{
    UI::separator( menuScaling, "Mouse" );

    const float buttonHeight = cGradientButtonFramePadding * menuScaling + ImGui::CalcTextSize( "Set other" ).y;
    for ( int i = 0; i < int( MouseMode::Count ); ++i )
    {
        MouseMode mode = MouseMode( i );
        if ( mode == MouseMode::None )
            continue;
        std::string modeName = getMouseModeString( mode );
        std::string ctrlStr = "None";
        auto ctrl = viewer->mouseController().findControlByMode( mode );
        if ( ctrl )
            ctrlStr = MouseController::getControlString( *ctrl );

        const float posY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( posY + cGradientButtonFramePadding * menuScaling / 2.f );
        ImGui::Text( "%s", modeName.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( menuWidth * 0.5f - 50.0f * menuScaling );
        ImGui::Text( "%s", ctrlStr.c_str() );
        ImGui::SameLine();
        ImGui::SetCursorPosX( menuWidth - 150.0f * menuScaling );

		ImGui::SetCursorPosY( posY );
        UI::button( fmt::format( "Set other##{}", i ).c_str(), Vector2f( -1, buttonHeight ) );
        if ( ImGui::IsItemHovered() )
        {
            ImGui::BeginTooltip();
            ImGui::Text( "Click here with preferred mouse button \nwith/without modifier (alt/ctrl/shift)" );
            ImGui::EndTooltip();

            if ( ImGui::GetIO().MouseClicked[0] || ImGui::GetIO().MouseClicked[1] || ImGui::GetIO().MouseClicked[2] )
            {
                MouseButton clikedBtn = MouseButton::Left;
                if ( ImGui::GetIO().MouseClicked[1] )
                    clikedBtn = MouseButton::Right;
                else if ( ImGui::GetIO().MouseClicked[2] )
                    clikedBtn = MouseButton::Middle;

                int modifier = 0;
                if ( ImGui::GetIO().KeyCtrl )
                    modifier |= GLFW_MOD_CONTROL;
                if ( ImGui::GetIO().KeyAlt )
                    modifier |= GLFW_MOD_ALT;
                if ( ImGui::GetIO().KeyShift )
                    modifier |= GLFW_MOD_SHIFT;

                viewer->mouseController().setMouseControl( { clikedBtn,modifier }, mode );
            }
        }
    }

    ImGui::SetNextItemWidth( 100 * menuScaling );
    UI::drag<NoUnit>( "Zoom Gain", viewer->scrollForce, 0.01f, 0.2f, 3.0f );
    UI::setTooltipIfHovered( "Sensitivity for mouse wheel rotation affecting the speed of zooming.", menuScaling );
}

void ViewerSettingsPlugin::drawSpaceMouseSettings_( float menuWidth, float menuScaling )
{
    UI::separator( menuScaling, "Spacemouse" );

    bool anyChanged = false;
    auto drawSlider = [&anyChanged, menuWidth] ( const char* label, float& value )
    {
        int valueAbs = int( std::fabs( value ) );
        bool inverse = value < 0.f;
        ImGui::SetNextItemWidth( menuWidth * 0.6f );
        bool changed = UI::slider<NoUnit>( label, valueAbs, 1, 100 );
        ImGui::SameLine( menuWidth * 0.78f );
        const float cursorPosY = ImGui::GetCursorPosY();
        ImGui::SetCursorPosY( cursorPosY + 3 );
        changed = UI::checkbox( ( std::string( "Inverse##" ) + label ).c_str(), &inverse ) || changed;
        if ( changed )
            value = valueAbs * ( inverse ? -1.f : 1.f );
        anyChanged = anyChanged || changed;
    };

    ImGui::Text( "%s", "Translation Sensitivity" );
    drawSlider( "X##translate", spaceMouseParams_.translateScale[0] );
    drawSlider( "Y##translate", spaceMouseParams_.translateScale[2] );
    drawSlider( "Zoom##translate", spaceMouseParams_.translateScale[1] );

    ImGui::Text( "%s", "Rotation Sensitivity" );
    drawSlider( "Ox##rotate", spaceMouseParams_.rotateScale[0] );
    drawSlider( "Oy##rotate", spaceMouseParams_.rotateScale[1] );
    drawSlider( "Oz##rotate", spaceMouseParams_.rotateScale[2] );
#if defined(_WIN32) || defined(__APPLE__)
    if ( UI::checkbox( "Zoom by mouse wheel", &activeMouseScrollZoom_ ) )
    {
        if ( auto spaceMouseHandler = getViewerInstance().getSpaceMouseHandler() )
        {
            auto hidapiHandler = std::dynamic_pointer_cast< SpaceMouseHandlerHidapi >( spaceMouseHandler );
            if ( hidapiHandler )
            {
                hidapiHandler->activateMouseScrollZoom( activeMouseScrollZoom_ );
            }
        }
    }
    UI::setTooltipIfHovered( "This mode is NOT recommended if you have 3Dconnexion driver installed, which sends mouse wheel fake events resulting in double reaction on SpaceMouse movement and camera tremble.", menuScaling );
#else
    (void)menuScaling;
#endif
    if ( anyChanged )
        getViewerInstance().setSpaceMouseParameters( spaceMouseParams_ );
}

void ViewerSettingsPlugin::drawTouchpadSettings_( float menuScaling )
{
    UI::separator( menuScaling, "Touchpad" );

    const std::vector<std::string> swipeModeList = { "Swipe Rotates Camera", "Swipe Moves Camera" };
    assert( swipeModeList.size() == (size_t)TouchpadParameters::SwipeMode::Count );

    bool updateSettings = false;
    if ( UI::checkbox( "Ignore Kinetic Movements", &touchpadParameters_.ignoreKineticMoves ) )
        updateSettings = true;
    if ( UI::checkbox( "Allow System to Interrupt Gestures", &touchpadParameters_.cancellable ) )
        updateSettings = true;
    if ( UI::combo( "Swipe Mode", (int*)&touchpadParameters_.swipeMode, swipeModeList ) )
        updateSettings = true;
    if ( updateSettings )
        viewer->setTouchpadParameters( touchpadParameters_ );
}

void ViewerSettingsPlugin::drawCustomSettings_( TabType tabType, float menuScaling )
{
    if ( comboSettings_[size_t( tabType )].empty() )
        return;
    UI::separator( menuScaling, "Tools" );
    for ( auto& settings : comboSettings_[size_t( tabType )] )
    {
        settings->draw( menuScaling );
    }
}

void ViewerSettingsPlugin::updateDialog_()
{
    updateThemes();

    spaceMouseParams_ = viewer->getSpaceMouseParameters();
    touchpadParameters_ = viewer->getTouchpadParameters();
#if defined(_WIN32) || defined(__APPLE__)
    if ( auto spaceMouseHandler = viewer->getSpaceMouseHandler() )
    {
        auto hidapiHandler = std::dynamic_pointer_cast< MR::SpaceMouseHandlerHidapi >( spaceMouseHandler );
        if ( hidapiHandler )
            activeMouseScrollZoom_ = hidapiHandler->isMouseScrollZoomActive();
    }
#endif
}

void ViewerSettingsPlugin::resetSettings_()
{
    viewer->getViewportSettingsManager()->resetSettings( *viewer );

    for ( size_t tabType = size_t( 0 ); tabType < size_t( TabType::Count ); tabType++ )
        for ( auto& settings : comboSettings_[ tabType ] )
            settings->reset();

    CommandLoop::appendCommand( [shadowGl = shadowGl_.get()] ()
    {
        shadowGl->enable( false );
    } );

    storedSamples_ = 8;
    if ( auto& settingsManager = viewer->getViewportSettingsManager() )
        settingsManager->saveInt( "multisampleAntiAliasing", storedSamples_ );
    needReset_ = storedSamples_ != curSamples_;

#if defined(_WIN32) || defined(__APPLE__)
    if ( auto spaceMouseHandler = viewer->getSpaceMouseHandler() )
    {
        auto hidapiHandler = std::dynamic_pointer_cast< MR::SpaceMouseHandlerHidapi >( spaceMouseHandler );
        if ( hidapiHandler )
            hidapiHandler->activateMouseScrollZoom( false );
    }
#endif

    updateDialog_();
}

MR_REGISTER_RIBBON_ITEM( ViewerSettingsPlugin )

}
