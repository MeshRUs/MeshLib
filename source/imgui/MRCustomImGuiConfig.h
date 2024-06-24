#pragma once

#include <MRMesh/MRVector2.h>
#include <MRMesh/MRColor.h>

#ifdef _WIN32
#   ifdef MRIMGUI_EXPORT
#       define IMGUI_API __declspec(dllexport)
#   else
#       define IMGUI_API __declspec(dllimport)
#   endif
#else
#       define IMGUI_API __attribute__((visibility("default")))
#endif

struct ImGuiContext;

IMGUI_API ImGuiContext*& MyImGuiTLS();
#define GImGui MyImGuiTLS()

#define IM_VEC2_CLASS_EXTRA \
    constexpr ImVec2( const MR::Vector2f & v ) noexcept : x( v.x ), y( v.y ) {} \
    constexpr operator MR::Vector2f() const noexcept { return { x, y }; }

#define IM_VEC4_CLASS_EXTRA \
    constexpr ImVec4( const MR::Vector4f & v ) noexcept : x( v.x ), y( v.y ), z( v.z ), w( v.w ) {} \
    constexpr ImVec4( const MR::Color & c ) noexcept : ImVec4( MR::Vector4f( c ) ) {} \
    constexpr operator MR::Vector4f() const noexcept { return { x, y, z, w }; } \
    constexpr operator MR::Color() const noexcept { return MR::Color( operator MR::Vector4f() ); }

#ifndef MR_FILTER_IMGUI_INPUT_CHARS
// This block is needed to prevent filtering alphabet characters from input
// as far as the function lay under ImGui namespace it is used before no namespace function is found
typedef int ImGuiInputTextFlags;
typedef int ImGuiDataType;
namespace ImGui
{
static inline ImGuiInputTextFlags InputScalar_DefaultCharsFilter( ImGuiDataType, const char* )
{
    return 0;
}
}
#endif

// declare function here to have access to it
// duplicate of ImGui_ImplGlfw_KeyToImGuiKey
// please review this function if update ImGui
IMGUI_API int GlfwToImGuiKey_Duplicate( int key );
