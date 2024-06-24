#pragma once
#include "MRRenderHelpers.h"
#include "MRMesh/MRLog.h"
#include <array>

namespace MR
{
// This class holds all shaders that are used in the program
// creates shader on access if it is needed
// holds shared memory buffer for loading to GPU
class MRVIEWER_CLASS GLStaticHolder
{
public:
    typedef unsigned int GLuint;
    enum ShaderType
    {
        Mesh,
        Picker,
        MeshDesktopPicker, // only for non corner-based mode
        TransparentMesh,

        DrawPoints,

        Lines,
        LinesJoint,
        LinesPicker,
        LinesJointPicker,

        TransparentLines,

        Labels,

        AdditionalLines,
        AdditionalPoints,
        AdditionalPointsNoOffset, // special shader for old intel gpu (Intel HD 4000)
        AdditionalQuad,
        ViewportBorder,
        TransparencyOverlayQuad,
        ShadowOverlayQuad,
        SimpleOverlayQuad,

        Volume,
        VolumePicker,
        Count
    };

    // Creates shader if it is not and return valid id
    MRVIEWER_API static GLuint getShaderId( ShaderType type );
    // Free shader from GL
    MRVIEWER_API static void freeShader( ShaderType type );
    // Free all shaders from GL
    MRVIEWER_API static void freeAllShaders();
    // Memory buffer for objects that about to be loaded to GPU, shared among different data types
    MRVIEWER_API static RenderObjectBuffer& getStaticGLBuffer();
private:
    GLStaticHolder();
    ~GLStaticHolder();

    static GLStaticHolder& instance_();

    void createShader_( ShaderType type );

    std::array<GLuint, size_t( Count )> shadersIds_;

    // it is stored here to prolong its life till this destructor
    std::shared_ptr<spdlog::logger> logger_;

    RenderObjectBuffer glBuffer_;
};
}