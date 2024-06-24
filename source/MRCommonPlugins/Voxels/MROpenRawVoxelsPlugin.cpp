#include "MROpenRawVoxelsPlugin.h"
#ifndef MRMESH_NO_OPENVDB
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/ImGuiMenu.h"
#include "MRViewer/MRRibbonConstants.h"
#include "MRViewer/ImGuiHelpers.h"
#include "MRViewer/MRFileDialog.h"
#include "MRViewer/MRProgressBar.h"
#include "MRViewer/MRViewport.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer/MRAppendHistory.h"
#include "MRMesh/MRChangeSceneAction.h"
#include <MRMesh/MRSceneRoot.h>
#include "MRViewer/MRUIStyle.h"
#include "MRViewer/MRViewer.h"

namespace
{
const std::vector<std::string> cScalarTypeNames =
{
    "UInt8",
    "Int8",
    "UInt16",
    "Int16",
    "UInt32",
    "Int32",
    "UInt64",
    "Int64",
    "Float32",
    "Float64",
    "4 x Float32"
};
}

namespace MR
{

OpenRawVoxelsPlugin::OpenRawVoxelsPlugin():
    StatePlugin( "Open RAW Voxels" )
{
    parameters_.dimensions = { 256, 256, 256 };
    parameters_.voxelSize = { 1.0f, 1.0f, 1.0f };
}

void OpenRawVoxelsPlugin::drawDialog( float menuScaling, ImGuiContext* )
{
    auto menuWidth = 350.0f * menuScaling;
    if ( !ImGuiBeginWindow_( { .width = menuWidth, .menuScaling = menuScaling } ) )
        return;
    ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );
    ImGui::PushStyleVar( ImGuiStyleVar_ItemInnerSpacing, { cDefaultItemSpacing * menuScaling, cDefaultItemSpacing * menuScaling } );

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { cCheckboxPadding * menuScaling, cCheckboxPadding * menuScaling } );
    UI::checkbox( "Auto parameters", &autoMode_ );
    ImGui::PopStyleVar();
    UI::setTooltipIfHovered( "Use this flag to parse RAW parameters from filename.", menuScaling );
    ImGui::Separator();

    if ( !autoMode_ )
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, { ImGui::GetStyle().FramePadding.x, cInputPadding * menuScaling } );
        ImGui::PushItemWidth( menuScaling * 200.0f );
        UI::drag<NoUnit>( "Dimensions", parameters_.dimensions, 1, 0, std::numeric_limits<int>::max() );
        UI::drag<LengthUnit>( "Voxel size", parameters_.voxelSize, 1e-3f, 0.0f );
        ImGui::PopItemWidth();
        ImGui::Separator();
        ImGui::PopStyleVar();
        UI::combo( "Scalar Type", ( int* )&parameters_.scalarType, cScalarTypeNames );
    }
    if ( UI::button( "Open file", Vector2f( -1, 0 ) ) )
    {
        const auto cb = [this] ( const std::filesystem::path& path )
        {
            if ( path.empty() )
                return;

            ProgressBar::orderWithMainThreadPostProcessing( "Load voxels", [params = parameters_, path, autoMode = autoMode_] ()->std::function<void()>
            {
                ProgressBar::nextTask( "Load file" );
                Expected<VdbVolume, std::string> res;
                auto error = std::make_shared<std::string>();

                const auto showError = [error] () -> void
                {
                    if ( error )
                        MR::showError( *error );
                };


                if ( autoMode )
                    res = VoxelsLoad::fromRaw( path, ProgressBar::callBackSetProgress );
                else
                    res = VoxelsLoad::fromRaw( path, params, ProgressBar::callBackSetProgress );

                if ( ProgressBar::isCanceled() )
                {
                    *error = getCancelMessage( path );
                    return showError;
                }

                if ( res.has_value() )
                {
                    ProgressBar::nextTask( "Create object" );
                    std::shared_ptr<ObjectVoxels> object = std::make_shared<ObjectVoxels>();
                    object->setName( utf8string( path.stem() ) );
                    object->construct( res->data, res->voxelSize, ProgressBar::callBackSetProgress );
                    auto bins = object->histogram().getBins();
                    auto minMax = object->histogram().getBinMinMax( bins.size() / 3 );

                    if ( ProgressBar::isCanceled() )
                    {
                        *error = getCancelMessage( path );
                        return showError;
                    }

                    ProgressBar::nextTask( "Create ISO surface" );
                    object->setIsoValue( minMax.first, ProgressBar::callBackSetProgress );
                    object->select( true );

                    if ( ProgressBar::isCanceled() )
                    {
                        *error = getCancelMessage( path );
                        return showError;
                    }

                    return [object, path] ()
                    {
                        AppendHistory<ChangeSceneAction>( "Open Voxels", object, ChangeSceneAction::Type::AddObject );
                        SceneRoot::get().addChild( object );
                        std::filesystem::path scenePath = path;
                        getViewerInstance().onSceneSaved( scenePath, false );
                        getViewerInstance().viewport().preciseFitDataToScreenBorder( { 0.9f } );
                    };
                }
                else
                {
                    *error = res.error();
                    return showError;
                }
            }, 3 );
            dialogIsOpen_ = false;
        };
        openFileDialogAsync( cb, {
            .filters = {
                { "RAW File", "*.raw;*.bin" },
            },
        } );
    }
    ImGui::PopStyleVar( 2 );
    ImGui::EndCustomStatePlugin();
}

bool OpenRawVoxelsPlugin::onEnable_()
{
    return true;
}

bool OpenRawVoxelsPlugin::onDisable_()
{
    return true;
}

MR_REGISTER_RIBBON_ITEM( OpenRawVoxelsPlugin )

}
#endif
#if defined( MRMESH_NO_OPENVDB ) && defined( __EMSCRIPTEN__ )
#include "MRCommonPlugins/Basic/MRWasmUnavailablePlugin.h"
MR_REGISTER_WASM_UNAVAILABLE_ITEM( OpenRawVoxelsPlugin, "Open RAW Voxels" )
#endif
