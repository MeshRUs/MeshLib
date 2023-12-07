#pragma once

#include "exports.h"
#include <functional>

namespace MR
{

enum class MouseButton;
enum class MouseMode;

class AlphaSortGL;
class ColorTheme;
class ImGuiImage;
class ImGuiMenu;
class IViewerSettingsManager;
class MarkedVoxelSlice;
class Palette;
class RecentFilesStore;
class ScopeHistory;
class SelectScreenLasso;
class SceneTextureGL;
class SpaceMouseHandlerHidapi;
class SplashWindow;
class StateBasePlugin;
class ViewerPlugin;
class ViewerSettingsManager;
class ViewerSetup;
class Viewer;
class ViewerEventQueue;
class Viewport;
class RibbonMenu;
class RibbonMenuItem;

class TouchpadController;
struct TouchpadParameters;

template<typename ...Connectables>
class StateListenerPlugin;
using StatePlugin = StateListenerPlugin<>;

using ViewerEventCallback = std::function<void()>;

// this is needed as far as MAKE_SLOT cannot be used with movable classes
#define MR_DELETE_MOVE(ClassName)\
ClassName(ClassName&&)noexcept = delete;\
ClassName& operator=(ClassName&&)noexcept = delete

#define MR_ADD_CTOR_DELETE_MOVE(ClassName)\
ClassName()=default;\
ClassName(ClassName&&)noexcept = delete;\
ClassName& operator=(ClassName&&)noexcept = delete

} //namespace MR
