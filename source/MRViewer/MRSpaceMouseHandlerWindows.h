#pragma once
#ifdef _WIN32
#include "MRSpaceMouseHandler.h"
#include "MRViewerEventsListener.h"

namespace MR
{

class SpaceMouseHandlerWindows : public SpaceMouseHandler, public MultiListener<PostFocusListener>
{
public:
    SpaceMouseHandlerWindows();
    ~SpaceMouseHandlerWindows();

    virtual void initialize() override;
    virtual void handle() override;
    virtual void updateConnected( int jid, int event );

    // set state of zoom by mouse scroll (to fix scroll signal from spacemouse driver)
    MRVIEWER_API void activateMouseScrollZoom( bool activeMouseScrollZoom );
    // get state of zoom by mouse scroll
    bool isMouseScrollZoomActive() { return activeMouseScrollZoom_; }
private:
    bool active_{ true };
    bool initialized_{ false };
    std::array<unsigned char, SMB_BUTTON_COUNT> buttons_{};
    int joystickIndex_{ -1 };
    const int* mapButtons_{ nullptr };
    int buttonsCount_{ 0 };
    float handleTime_{ 0.f };

    std::thread updateThread_;
    std::atomic_bool updateThreadActive_{ true };
    std::atomic<std::array<float, 6>> axesDiff_;
    std::array<float, 6> axesOld_{};

    bool activeMouseScrollZoom_{ false };

    //hotfix TODO rework
    bool isUniversalReceiver_{ false };

    virtual void postFocus_( bool focused ) override;

    void updateConnected_();
    void startUpdateThread_();
};

}

#endif
