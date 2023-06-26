#ifdef _WIN32
#include "MRSpaceMouseHandlerWindows.h"
#include "MRPch/MRSpdlog.h"
#include "MRViewerInstance.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include <windows.h>
#include <GLFW/glfw3.h>
#include <functional>

namespace MR
{

constexpr float cAxesScale = 93.62f; // experemental coefficient to scale raw axes data to range [-1 ; 1]
constexpr float cAxesThreshold = 1.e-2f / cAxesScale; // axis threshold to send signals spacemouse move

struct DeviceInfo
{
    DWORD vendorId{ 0 };
    DWORD deviceId{ 0 };
    std::string deviceName;
};

constexpr DWORD logitechId = 0x46D;
constexpr DWORD connexionId = 0x256F;

const DeviceInfo connexionDevices[] = {
    { connexionId, 0xC62E, "SpaceMouse Wireless (cabled)" },
    { connexionId, 0xC62F, "SpaceMouse Wireless Receiver" },
    { connexionId, 0xC631, "SpaceMouse Pro Wireless (cabled)" },
    { connexionId, 0xC632, "SpaceMouse Pro Wireless Receiver" },
    { connexionId, 0xC633, "SpaceMouse Enterprise" },
    { connexionId, 0xC635, "SpaceMouse Compact" },
//     { connexionId, 0xC651, "CadMouse Wireless" },
    { connexionId, 0xC652, "Universal Receiver" },
//     { connexionId, 0xC654, "CadMouse Pro Wireless" },
//     { connexionId, 0xC657, "CadMouse Pro Wireless Left" }
};

constexpr int mapButtonsCompact[2] = {
    SMB_CUSTOM_1, SMB_CUSTOM_2
};
constexpr int mapButtonsPro[15] = {
    SMB_MENU, SMB_FIT,
    SMB_TOP, SMB_RIGHT, SMB_FRONT, SMB_ROLL_CW,
    SMB_CUSTOM_1, SMB_CUSTOM_2, SMB_CUSTOM_3, SMB_CUSTOM_4,
    SMB_ESC, SMB_ALT, SMB_SHIFT, SMB_CTRL,
    SMB_LOCK_ROT
};
// TODO !!! NOT TESTED !!!
constexpr int mapButtonsEnterprise[31] = {
    SMB_MENU, SMB_FIT,
    SMB_TOP, SMB_RIGHT, SMB_FRONT, SMB_ROLL_CW, SMB_LOCK_ROT,
    SMB_ISO1, SMB_BTN_V1, SMB_BTN_V2, SMB_BTN_V3,
    SMB_CUSTOM_1, SMB_CUSTOM_2, SMB_CUSTOM_3, SMB_CUSTOM_4, SMB_CUSTOM_5, SMB_CUSTOM_6,
    SMB_CUSTOM_7, SMB_CUSTOM_8, SMB_CUSTOM_9, SMB_CUSTOM_10, SMB_CUSTOM_11, SMB_CUSTOM_12,
    SMB_ESC, SMB_ENTER, SMB_ALT, SMB_SHIFT, SMB_CTRL, SMB_TAB, SMB_SPACE, SMB_DELETE
};

// Array of input device examples to register
RAWINPUTDEVICE inputDevices[] = {
    {0x01, 0x08, 0x00, 0x00}, // Usage Page = 0x01 Generic Desktop Page, Usage Id = 0x08 Multi-axis Controller
    {0x01, 0x05, 0x00, 0x00}, // Game Pad
    {0x01, 0x04, 0x00, 0x00} // Joystick
};
constexpr int inputDevicesCount = sizeof( inputDevices ) / sizeof( inputDevices[0] );

bool isSpaceMouseAttached()
{
    unsigned int devicesCount = 0;

    if ( GetRawInputDeviceList( NULL, &devicesCount, sizeof( RAWINPUTDEVICELIST ) ) != 0 )
        return false;

    if ( devicesCount == 0 )
        return false;

    std::vector<RAWINPUTDEVICELIST> rawInputDeviceList( devicesCount );
    if ( GetRawInputDeviceList( rawInputDeviceList.data(), &devicesCount, sizeof( RAWINPUTDEVICELIST ) ) == unsigned int( -1 ) )
        return false;

    for ( unsigned int i = 0; i < devicesCount; ++i )
    {
        RID_DEVICE_INFO rdi = { sizeof( rdi ) };
        unsigned int cbSize = sizeof( rdi );

        if ( GetRawInputDeviceInfo( rawInputDeviceList[i].hDevice, RIDI_DEVICEINFO, &rdi, &cbSize ) > 0 )
        {
            //skip non HID and non logitec (3DConnexion) devices
            if ( !( rdi.dwType == RIM_TYPEHID && ( rdi.hid.dwVendorId == logitechId || rdi.hid.dwVendorId == connexionId ) ) )
                continue;

            //check if devices matches Multi-axis Controller
            for ( unsigned int j = 0; j < inputDevicesCount; ++j )
            {
                if ( inputDevices[j].usUsage == rdi.hid.usUsage && inputDevices[j].usUsagePage == rdi.hid.usUsagePage )
                {
                    return true;
                }
            }
        }
    }
    return false;
}

bool InitializeRawInput()
{
    unsigned int cbSize = sizeof( inputDevices[0] );
    for ( size_t i = 0; i < inputDevicesCount; i++ )
    {
        // Set the target window to use
        //inputDevices[i].hwndTarget = hwndTarget;

        // enable receiving the WM_INPUT_DEVICE_CHANGE message.
        inputDevices[i].dwFlags |= RIDEV_DEVNOTIFY;
    }
    return ( RegisterRawInputDevices( inputDevices, inputDevicesCount, cbSize ) != FALSE );
}

SpaceMouseHandlerWindows::SpaceMouseHandlerWindows()
{
    axesDiff_ = { 0, 0, 0, 0, 0, 0 };
    connect( &getViewerInstance() );
}

SpaceMouseHandlerWindows::~SpaceMouseHandlerWindows()
{
    updateThreadActive_ = false;
    if ( updateThread_.joinable() )
        updateThread_.join();
}

void SpaceMouseHandlerWindows::initialize()
{
    bool spaceMouseAttached = isSpaceMouseAttached();
    if ( spaceMouseAttached )
        spdlog::info( "Found attached SpaceMouse" );
    else
    {
        spdlog::info( "Not found any attached SpaceMouse" );
        return;
    }

    initialized_ = InitializeRawInput();
    spdlog::info( "Initialize SpaceMouse {}", initialized_ ? "success" : "failed" );

    updateConnected_();
}

void SpaceMouseHandlerWindows::handle()
{
    if ( !initialized_ || joystickIndex_ == -1 || !active_ )
        return;

    auto& viewer = getViewerInstance();
    int count;
    std::array<float, 6> axes = {0, 0, 0, 0, 0, 0};
    if ( isUniversalReceiver_ )
    {
        const float* axesNew = glfwGetJoystickAxes( joystickIndex_, &count );
        if ( count == 6 )
            for ( int i = 0; i < 6; ++i )
                axes[i] = axesNew[i];
    }
    else
    {
        axes = axesDiff_;
        for ( auto& v : axes )
            v *= cAxesScale;
    }
    if ( std::any_of( axes.begin(), axes.end(), [] ( const float& v ) { return std::fabs( v ) > cAxesThreshold; } ) )
    {
        axesDiff_ = { 0, 0, 0, 0, 0, 0 };
        Vector3f translate( axes[0], axes[1], axes[2] );
        Vector3f rotate( axes[3], axes[4], axes[5] );

        float newHandleTime = float( glfwGetTime() );
        if ( handleTime_ == 0.f )
            handleTime_ = newHandleTime;
        else
        {
            float timeScale = std::clamp( ( newHandleTime - handleTime_ ), 0.f, 0.5f ) * 60.f;
            handleTime_ = newHandleTime;

            translate *= timeScale;
            rotate *= timeScale;

            if ( active_ )
                viewer.spaceMouseMove( translate, rotate );
        }
    }

    const unsigned char* buttons = glfwGetJoystickButtons( joystickIndex_, &count );
    // SpaceMouse Compact have 2 btns, Pro - 15, Enterprise - 31
    if ( count == 2 || count == 15 || count == 31 )
    {
        buttonsCount_ = count;
        if ( active_ )
        {
            mapButtons_ = mapButtonsPro;
            if ( count == 2 )
                mapButtons_ = mapButtonsCompact;
            else if ( count == 15 )
                mapButtons_ = mapButtonsPro;
            else
                mapButtons_ = mapButtonsEnterprise;
            for ( int i = 0; i < count; ++i )
            {
                int button = mapButtons_[i];
                if ( !buttons_[i] && buttons[i] ) // button down
                    viewer.spaceMouseDown( button );
                else if ( buttons_[i] && !buttons[i] ) // button up
                    viewer.spaceMouseUp( button );
//                 else if ( buttons_[i] && buttons[i] &&  ) // button repeat
//                     viewer.spaceMouseRepeat( button );
            }
        }
        std::copy( buttons, buttons + count, buttons_.begin() );
    }
}

void SpaceMouseHandlerWindows::updateConnected( int /*jid*/, int /*event*/ )
{
    if ( initialized_ )
        updateConnected_();
    else
        initialize();
}

void SpaceMouseHandlerWindows::activateMouseScrollZoom( bool activeMouseScrollZoom )
{
    activeMouseScrollZoom_ = activeMouseScrollZoom;
    getViewerInstance().mouseController.setMouseScroll( joystickIndex_ == -1 || activeMouseScrollZoom_ );
}

void SpaceMouseHandlerWindows::postFocus_( bool focused )
{
    active_ = focused;
}

void SpaceMouseHandlerWindows::updateConnected_()
{
    isUniversalReceiver_ = false;
    int newJoystickIndex = -1;
    for ( int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_LAST; ++i )
    {
        int present = glfwJoystickPresent( i );
        if ( !present )
            continue;
        const char* name = glfwGetJoystickName( i );
        std::string_view str( name );
        auto findRes = str.find( "SpaceMouse" );
        if ( findRes != std::string_view::npos )
        {
            newJoystickIndex = i;
            break;
        }

        findRes = str.find( "3Dconnexion Universal Receiver" );
        if ( findRes != std::string_view::npos )
        {
            isUniversalReceiver_ = true;
            newJoystickIndex = i;
            break;
        }
    }

    if ( newJoystickIndex == joystickIndex_ )
        return;

    auto& viewer = getViewerInstance();
    if ( joystickIndex_ != -1 )
    {
        for ( int i = 0; i < buttonsCount_; ++i )
        {
            if ( buttons_[i] ) // button up
                viewer.spaceMouseUp( mapButtons_[i] );
        }
        buttons_ = {};

        updateThreadActive_ = false;
        if ( updateThread_.joinable() )
            updateThread_.join();
    }

    joystickIndex_ = newJoystickIndex;

    if ( joystickIndex_ != -1 && !isUniversalReceiver_ )
    {
        int count;
        const float* axesNew = glfwGetJoystickAxes( joystickIndex_, &count );
        std::copy( axesNew, axesNew + 6, axesOld_.begin() );

        startUpdateThread_();
    }

    getViewerInstance().mouseController.setMouseScroll( joystickIndex_ == -1 || activeMouseScrollZoom_ );
}

void SpaceMouseHandlerWindows::startUpdateThread_()
{
    if ( joystickIndex_ == -1 )
        return;

    updateThreadActive_ = true;
    axesDiff_ = { 0, 0, 0, 0, 0, 0 };
    // additional thread needed to avoid double changing spacemouse axes values
    // we assume that the mouse has its own refresh rate
    // if we receive data less frequently, we can accept several changes as one
    // to avoid this, we poll the mouse more often, but we can't reduce frequency main thread
    // so we create another thread
    updateThread_ = std::thread( [&, joystickIndex = joystickIndex_]
    {
        std::array<float, 6> axesDiff{};
        int count = 0;
        do
        {
            const float* axesNew = glfwGetJoystickAxes( joystickIndex, &count );
            if ( count == 6 )
            {
                axesDiff = axesDiff_;
                for ( int i = 0; i < 6; ++i )
                {
                    float newDiff = axesNew[i] - axesOld_[i];
                    // updating axis differences
                    // if in the last cycle we got a non-zero diff and in this cycle we got a non-zero diff but of a different sign, we remember the last diff
                    // if in the past and this cycle we got a non-zero diff of one sign, we remember a larger diff
                    if ( newDiff * axesDiff[i] < 0 )
                        axesDiff[i] = newDiff;
                    else if ( std::fabs( axesDiff[i] ) < std::fabs( newDiff ) )
                        axesDiff[i] = newDiff;
                }
                axesDiff_ = axesDiff;
                std::copy( axesNew, axesNew + 6, axesOld_.begin() );
            }
            std::this_thread::sleep_for( std::chrono::microseconds( 1000 ) );
        } while ( updateThreadActive_ );
    } );
}

}

#endif
