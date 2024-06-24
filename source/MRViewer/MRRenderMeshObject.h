#pragma once
#include "exports.h"
#include "MRMesh/MRIRenderObject.h"
#include "MRMesh/MRMeshTexture.h"
#include "MRMesh/MRBuffer.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"

namespace MR
{
class MRVIEWER_CLASS RenderMeshObject : public virtual IRenderObject
{
public:
    MRVIEWER_API RenderMeshObject( const VisualObject& visObj );
    MRVIEWER_API virtual ~RenderMeshObject();

    MRVIEWER_API virtual bool render( const ModelRenderParams& params ) override;
    MRVIEWER_API virtual void renderPicker( const ModelBaseRenderParams& params, unsigned geomId ) override;
    MRVIEWER_API virtual size_t heapBytes() const override;
    MRVIEWER_API virtual size_t glBytes() const override;
    MRVIEWER_API virtual void forceBindAll() override;
protected:
    const ObjectMeshHolder* objMesh_;

    int vertPosSize_{ 0 };
    int vertNormalsSize_{ 0 };
    int vertColorsSize_{ 0 };
    int vertUVSize_{ 0 };
    int faceIndicesSize_{ 0 };
    int edgeSize_{ 0 };
    int selEdgeSize_{ 0 };
    int bordersSize_{ 0 };
    Vector2i faceSelectionTextureSize_;
    Vector2i faceNormalsTextureSize_;

    MRVIEWER_API RenderBufferRef<Vector3f> loadVertPosBuffer_();
    MRVIEWER_API RenderBufferRef<Vector3f> loadVertNormalsBuffer_();
    MRVIEWER_API RenderBufferRef<Color> loadVertColorsBuffer_();
    MRVIEWER_API RenderBufferRef<UVCoord> loadVertUVBuffer_();
    MRVIEWER_API RenderBufferRef<Vector3i> loadFaceIndicesBuffer_();
    MRVIEWER_API RenderBufferRef<unsigned> loadFaceSelectionTextureBuffer_();
    MRVIEWER_API RenderBufferRef<Vector4f> loadFaceNormalsTextureBuffer_();

    typedef unsigned int GLuint;

    GLuint edgesArrayObjId_{ 0 };
    GLuint borderArrayObjId_{ 0 };
    GlBuffer borderBuffer_;

    GLuint selectedEdgesArrayObjId_{ 0 };
    GlBuffer selectedEdgesBuffer_;

    GLuint meshArrayObjId_{ 0 };
    GLuint meshPickerArrayObjId_{ 0 };

    GlBuffer vertPosBuffer_;
    GlBuffer vertUVBuffer_;
    GlBuffer vertNormalsBuffer_;
    GlBuffer vertColorsBuffer_;

    GlBuffer facesIndicesBuffer_;

    GlTexture2 texture_;
    GlTexture2 faceSelectionTex_;
    GlTexture2 faceColorsTex_;
    GlTexture2 facesNormalsTex_;

    GlTexture2 edgesTexture_;
    GlTexture2 selEdgesTexture_;
    GlTexture2 borderTexture_;
    GlTexture2 emptyVertsColorTexture_;
    GlTexture2 emptyLinesColorTexture_;

    int maxTexSize_{ 0 };

    MRVIEWER_API virtual void renderEdges_( const ModelRenderParams& parameters, bool alphaSort, GLuint vao, const Color& color, uint32_t dirtyFlag );

    MRVIEWER_API virtual void renderMeshEdges_( const ModelRenderParams& parameters, bool alphaSort );

    MRVIEWER_API virtual void bindMesh_( bool alphaSort );

    MRVIEWER_API virtual void bindMeshPicker_();

    MRVIEWER_API virtual void bindEdges_();
    MRVIEWER_API virtual void bindBorders_();
    MRVIEWER_API virtual void bindSelectedEdges_();
    MRVIEWER_API virtual void bindEmptyTextures_( GLuint shaderId );

    MRVIEWER_API virtual void drawMesh_( bool solid, ViewportId viewportId, bool picker = false ) const;

    // Create a new set of OpenGL buffer objects
    MRVIEWER_API virtual void initBuffers_();

    // Release the OpenGL buffer objects
    MRVIEWER_API virtual void freeBuffers_();

    MRVIEWER_API virtual void update_( ViewportMask mask );

    // Marks dirty buffers that need to be uploaded to OpenGL
    uint32_t dirty_{ 0 };
    // ...
    bool dirtyEdges_{ false };

#ifdef __EMSCRIPTEN__
    bool cornerMode = true;
#else
    bool cornerMode = false;
#endif
};

}
