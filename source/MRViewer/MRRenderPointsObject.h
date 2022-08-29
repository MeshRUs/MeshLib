#pragma once

#include "MRMesh/MRIRenderObject.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class RenderPointsObject : public IRenderObject
{
public:
    RenderPointsObject( const VisualObject& visObj );
    ~RenderPointsObject();

    virtual void render( const RenderParams& params ) const override;
    virtual void renderPicker( const BaseRenderParams& params, unsigned geomId ) const override;
    virtual size_t heapBytes() const override;

private:
    const ObjectPointsHolder* objPoints_;

    // memory buffer for objects that about to be loaded to GPU, shared among different data types
    mutable RenderObjectBuffer bufferObj_;
    mutable int validIndicesSize_{ 0 };
    mutable int vertSelectionTextureSize_{ 0 };

    RenderBufferRef<VertId> loadValidIndicesBuffer_() const;
    RenderBufferRef<unsigned> loadVertSelectionTextureBuffer_() const;

    typedef unsigned int GLuint;
    GLuint pointsArrayObjId_{ 0 };
    GLuint pointsPickerArrayObjId_{ 0 };

    mutable GlBuffer vertPosBuffer_;
    mutable GlBuffer vertNormalsBuffer_;
    mutable GlBuffer vertColorsBuffer_;

    mutable GlBuffer validIndicesBuffer_;

    GLuint vertSelectionTex_{ 0 };

    void bindPoints_() const;
    void bindPointsPicker_() const;

    // Create a new set of OpenGL buffer objects
    void initBuffers_();

    // Release the OpenGL buffer objects
    void freeBuffers_();

    void update_() const;

    // Marks dirty buffers that need to be uploaded to OpenGL
    mutable uint32_t dirty_;
};

}