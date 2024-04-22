#pragma once

#ifdef __APPLE__
#include "MRSpaceMouseHandler.h"
#include "MRViewerEventsListener.h"

namespace MR
{

class SpaceMouseHandler3dxMacDriver : public SpaceMouseHandler, public PostFocusListener
{
public:
    SpaceMouseHandler3dxMacDriver();
    ~SpaceMouseHandler3dxMacDriver() override;

    void setClientName( const char* name, size_t len = 0 );

public:
    // SpaceMouseHandler
    bool initialize() override;
    void handle() override;

private:
    // PostFocusListener
    void postFocus_( bool focused ) override;

private:
    struct LibHandle;
    std::unique_ptr<LibHandle> lib_;

    std::unique_ptr<uint8_t[]> clientName_;
    uint16_t clientId_{ 0 };
};

} // namespace MR
#endif
