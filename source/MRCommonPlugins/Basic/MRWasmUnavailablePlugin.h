#pragma once
#ifdef __EMSCRIPTEN__
#include "MRCommonPlugins/exports.h"
#include "MRViewer/MRStatePlugin.h"
#include "MRViewer/MRRibbonSchema.h"
#include "MRViewer/ImGuiMenu.h"
#include "MRViewer/MRColorTheme.h"
#include "MRPch/MRWasm.h"

namespace MR
{

class WasmUnavailableObjectVoxels
{
public:
    constexpr static const char* TypeName() noexcept { return "ObjectVoxels"; }
};

class MRCOMMONPLUGINS_CLASS WasmUnavailableItem : public RibbonMenuItem
{
public:
    WasmUnavailableItem( const std::string& name ) :
        RibbonMenuItem( name ) {}

    virtual bool action() override
    {
        showDownloadWindow_();
        return false;
    }

    EMSCRIPTEN_KEEPALIVE void showDownloadWindow_()
    {
        bool isLightThemeEnabled = ( ColorTheme::getPreset() == ColorTheme::Preset::Light );
        #pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
            EM_ASM( showDownloadWindow( $0 ), isLightThemeEnabled );
#pragma clang diagnostic pop
    }
};

}

#define MR_REGISTER_WASM_UNAVAILABLE_ITEM( pluginType, name )\
    static MR::RibbonMenuItemAdder<MR::WasmUnavailableItem> ribbonMenuItemAdder##pluginType##_(name);

#define MR_REGISTER_WASM_UNAVAILABLE_ITEM_CHECK( pluginType, availCheckType, name )\
    static MR::RibbonMenuItemAdder<MR::WasmUnavailablePluginWithSceneCheck<availCheckType>> ribbonMenuItemAdder##pluginType##_(name);

#endif
