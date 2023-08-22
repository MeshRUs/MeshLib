#pragma once

#include "MRViewport.h"

#include <functional>
#include <memory>

struct GLFWwindow;

namespace MR
{

/// Class for touchpad gesture processing
///
/// The supported gestures and their default actions are:
///  - pinch: zooms the camera's angle
///  - rotate: rotates the camera around the scene's center along the Z axis
///  - swipe: rotates the camera around the world's center along all axes (by default) or moves the camera
/// The actions are mapped by calling the `connect` method; you can define your own actions by connecting to the
/// `MR::Viewer::touchpad*` signals.
///
/// NOTE: on some platforms the default GLFW mouse scroll is being disabled by the touchpad event handlers because
/// mouse scrolls and touchpad swipe gestures are a single event type there and have to be processed by a single handler.
/// Consider it if you use this class and define your own scroll callback with `glfwSetScrollCallback`.
class TouchpadController : public MultiListener<
    TouchpadRotateGestureBeginListener, TouchpadRotateGestureUpdateListener, TouchpadRotateGestureEndListener,
    TouchpadSwipeGestureBeginListener, TouchpadSwipeGestureUpdateListener, TouchpadSwipeGestureEndListener,
    TouchpadZoomGestureBeginListener, TouchpadZoomGestureUpdateListener, TouchpadZoomGestureEndListener
>
{
public:
    MR_ADD_CTOR_DELETE_MOVE( TouchpadController );

    /// initialize listening to touchpad events
    MRVIEWER_API void initialize( GLFWwindow* window );

    struct Parameters
    {
        /// most touchpads implement kinetic (or inertial) scrolling, this option disables handling of these events
        bool ignoreKineticMoves = false;
        /// enable gesture's cancellability, i.e. revert its changes in case of external interruption
        bool cancellable = false;
        /// swipe processing mode
        enum SwipeMode {
            SwipeRotatesCamera = 0,
            SwipeMovesCamera = 1,
            SwipeModeCount,
        } swipeMode = SwipeRotatesCamera;
    };
    [[nodiscard]] MRVIEWER_API const Parameters& getParameters() const;
    MRVIEWER_API void setParameters( const Parameters& parameters );

    /// Base class for platform-dependent code handling touchpad events
    ///
    /// If you want to add touchpad gesture support on your platform, inherit this class, extend
    /// the `TouchpadController::initialize` method and call the corresponding methods on the touchpad events.
    class Handler
    {
    public:
        virtual ~Handler() = default;

        /// gesture state
        enum class GestureState
        {
            /// gesture has started
            Begin,
            /// gesture data has updated
            Update,
            /// gesture has ended
            End,
        };

        /// not a touchpad gesture but actually a mouse scroll; call it if mouse scrolls and touchpad swipe gestures are
        /// a single event on your platform (and when the event is actually is a mouse scroll)
        void mouseScroll( float dx, float dy, bool kinetic );
        /// rotate gesture
        void rotate( float angle, GestureState state );
        /// swipe gesture; `kinetic` flag is set when the event is produced not by a user action but by hardware 'kinetic' scrolling
        void swipe( float dx, float dy, bool kinetic, GestureState state );
        /// pitch ('zoom') gesture
        void zoom( float scale, bool kinetic, GestureState state );
    };

private:
    std::unique_ptr<Handler> handler_;
    Parameters parameters_;

    Viewport::Parameters initRotateParams_;
    virtual void touchpadRotateGestureBegin_() override;
    virtual void touchpadRotateGestureUpdate_( float angle ) override;
    virtual void touchpadRotateGestureEnd_() override;

    virtual void touchpadSwipeGestureBegin_() override;
    virtual void touchpadSwipeGestureUpdate_( float deltaX, float deltaY, bool kinetic ) override;
    virtual void touchpadSwipeGestureEnd_() override;

    Viewport::Parameters initZoomParams_;
    virtual void touchpadZoomGestureBegin_() override;
    virtual void touchpadZoomGestureUpdate_( float scale, bool kinetic ) override;
    virtual void touchpadZoomGestureEnd_() override;
};

} // namespace MR