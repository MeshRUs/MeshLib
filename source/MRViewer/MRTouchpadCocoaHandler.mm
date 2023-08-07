#ifdef __APPLE__

#include "MRTouchpadCocoaHandler.h"

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <objc/objc-runtime.h>

#include <map>

namespace
{

class TouchpadCocoaHandlerRegistry
{
public:
    static TouchpadCocoaHandlerRegistry& instance()
    {
        static TouchpadCocoaHandlerRegistry instance;
        return instance;
    }

    void add( NSView* view, MR::TouchpadCocoaHandler* handler )
    {
        registry_.emplace( view, handler );
    }

    void remove( NSView* view )
    {
        registry_.erase( view );
    }

    [[nodiscard]] MR::TouchpadCocoaHandler* find( NSView* view ) const
    {
        const auto it = registry_.find( view );
        if ( it != registry_.end() )
            return it->second;
        else
            return nullptr;
    }

private:
    std::map<NSView*, MR::TouchpadCocoaHandler*> registry_;
};

std::optional<MR::TouchpadController::Impl::GestureState> convert( NSGestureRecognizerState state )
{
    using GS = MR::TouchpadController::Impl::GestureState;
    switch ( state )
    {
        case NSGestureRecognizerStateBegan:
            return GS::Begin;
        case NSGestureRecognizerStateChanged:
            return GS::Change;
        case NSGestureRecognizerStateEnded:
            return GS::End;
        case NSGestureRecognizerStateCancelled:
            return GS::Cancel;
        default:
            return std::nullopt;
    }
}

}

namespace MR
{

TouchpadCocoaHandler::TouchpadCocoaHandler( GLFWwindow* window )
    : view_( ( (NSWindow*)glfwGetCocoaWindow( window ) ).contentView )
{
    Class cls = [view_ class];

    magnificationGestureRecognizer_ = [[NSMagnificationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleMagnificationGesture:)];
    if ( !class_respondsToSelector( cls, @selector(handleMagnificationGesture:) ) )
        class_addMethod( cls, @selector(handleMagnificationGesture:), (IMP)TouchpadCocoaHandler::onMagnificationGestureEvent, "v@:@" );
    [view_ addGestureRecognizer:magnificationGestureRecognizer_];

    rotationGestureRecognizer_ = [[NSRotationGestureRecognizer alloc] initWithTarget:view_ action:@selector(handleRotationGesture:)];
    if ( !class_respondsToSelector( cls, @selector(handleRotationGesture:) ) )
        class_addMethod( cls, @selector(handleRotationGesture:), (IMP)TouchpadCocoaHandler::onRotationGestureEvent, "v@:@" );
    [view_ addGestureRecognizer:rotationGestureRecognizer_];

    // NOTE: GLFW scroll handler is replaced here
    if ( !class_respondsToSelector( cls, @selector(scrollWheel:) ) )
    {
        previousScrollWheelMethod_ = nil;
        class_addMethod( cls, @selector(scrollWheel:), (IMP)TouchpadCocoaHandler::onScrollEvent, "v@:@" );
    }
    else
    {
        previousScrollWheelMethod_ = (IMP)[view_ methodForSelector:@selector(scrollWheel:)];
        class_replaceMethod( cls, @selector(scrollWheel:), (IMP)TouchpadCocoaHandler::onScrollEvent, "v@:@" );
    }

    TouchpadCocoaHandlerRegistry::instance().add( view_, this );
}

TouchpadCocoaHandler::~TouchpadCocoaHandler()
{
    if ( previousScrollWheelMethod_ != nil )
    {
        Class cls = [view_ class];
        class_replaceMethod( cls, @selector(scrollWheel:), (IMP)previousScrollWheelMethod_, "v@:@" );
    }
    [rotationGestureRecognizer_ release];
    [magnificationGestureRecognizer_ release];
}

void TouchpadCocoaHandler::onMagnificationGestureEvent( NSView* view, SEL cmd, NSMagnificationGestureRecognizer* recognizer )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    const auto state = convert( recognizer.state );
    if ( state )
        handler->zoom( std::exp( -recognizer.magnification ), *state );
}

void TouchpadCocoaHandler::onRotationGestureEvent( NSView* view, SEL cmd, NSRotationGestureRecognizer* recognizer )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( ! handler )
        return;

    const auto state = convert( recognizer.state );
    if ( state )
        handler->rotate( recognizer.rotation, *state );
}

void TouchpadCocoaHandler::onScrollEvent( NSView* view, SEL cmd, NSEvent* event )
{
    auto* handler = TouchpadCocoaHandlerRegistry::instance().find( view );
    if ( !handler )
        return;

    auto deltaX = [event scrollingDeltaX];
    auto deltaY = [event scrollingDeltaY];
    if ( [event hasPreciseScrollingDeltas] )
    {
        deltaX *= 0.1;
        deltaY *= 0.1;
    }
    if ( deltaX == 0.0 && deltaY == 0.0 )
        return;

    if ( [event subtype] == NSEventSubtypeMouseEvent )
        handler->mouseScroll( deltaX, deltaY, [event momentumPhase] != NSEventPhaseNone );
    else
        handler->swipe( deltaX, deltaY, [event momentumPhase] != NSEventPhaseNone );
}

}

#endif