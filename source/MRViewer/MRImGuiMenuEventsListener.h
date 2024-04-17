#pragma once

#include "exports.h"
#include "MRViewer/ImGuiMenu.h"
#include "MRViewer/MRViewerEventsListener.h"

namespace MR
{

// A helper base class to subscribe to `ImGuiMenu::manuallySelectObjectSignal`.
struct MRVIEWER_CLASS NameTagClickListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( NameTagClickListener );
    virtual ~NameTagClickListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual bool onNameTagClicked_( Object& object, ImGuiMenu::NameTagSelectionMode mode ) = 0;
};

// A helper base class to subscribe to `ImGuiMenu::drawUiSignal`.
// This is useful for widgets that need to draw their own UI, like plugins do in their `drawDialog_()`.
struct MRVIEWER_CLASS DrawUiListener : ConnectionHolder
{
    MR_ADD_CTOR_DELETE_MOVE( DrawUiListener );
    virtual ~DrawUiListener() = default;
    MRVIEWER_API virtual void connect( Viewer* viewer, int group, boost::signals2::connect_position pos ) override;
protected:
    virtual void drawUi_( float menuScaling ) = 0;
};

}
