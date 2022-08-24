#pragma once

#include "MRGladGlfw.h"
#include "MRGLMacro.h"
#include "exports.h"
#include "MRMesh/MRColor.h"
#include <cassert>

namespace MR
{

// represents OpenGL buffer owner, and allows uploading data in it remembering buffer size
class GlBuffer
{
    constexpr static GLuint NO_BUF = 0;
public:
    GlBuffer() = default;
    GlBuffer( const GlBuffer & ) = delete;
    GlBuffer( GlBuffer && r ) : bufferID_( r.bufferID_ ), size_( r.size_ ) { r.detach_(); }
    ~GlBuffer() { del(); }

    GlBuffer& operator =( const GlBuffer & ) = delete;
    GlBuffer& operator =( GlBuffer && r ) { del(); bufferID_ = r.bufferID_; size_ = r.size_; r.detach_(); return * this; }

    bool valid() const { return bufferID_ != NO_BUF; }
    size_t size() const { return size_; }

    // generates new buffer
    MRVIEWER_API void gen();

    // deletes the buffer
    MRVIEWER_API void del();

    // binds current buffer to OpenGL context
    MRVIEWER_API void bind();

    // creates GL data buffer using given data
    MRVIEWER_API void loadData( const char * arr, size_t arrSize );
    template<typename T>
    void loadData( const T * arr, size_t arrSize ) { loadData( (const char *)arr, sizeof( T ) * arrSize ); }

private:
    /// another object takes control over the GL buffer
    void detach_() { bufferID_ = NO_BUF; size_ = 0; }

private:
    GLuint bufferID_ = NO_BUF;
    size_t size_ = 0;
};

template<typename T, template<typename, typename...> class C, typename... args>
GLint bindVertexAttribArray(
    const GLuint program_shader,
    const std::string& name,
    GlBuffer & buf,
    const C<T, args...>& V,
    int baseTypeElementsNumber,
    bool refresh,
    bool forceUse = false )
{
    GL_EXEC( GLint id = glGetAttribLocation( program_shader, name.c_str() ) );
    if ( id < 0 )
        return id;
    if ( V.size() == 0 && !forceUse )
    {
        GL_EXEC( glDisableVertexAttribArray( id ) );
        buf.del();
        return id;
    }

    if ( refresh )
        buf.loadData( V.data(), V.size() );
    else
        buf.bind();

    // GL_FLOAT is left here consciously 
    if constexpr ( std::is_same_v<Color, T> )
    {
        GL_EXEC( glVertexAttribPointer( id, baseTypeElementsNumber, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0 ) );
    }
    else
    {
        GL_EXEC( glVertexAttribPointer( id, baseTypeElementsNumber, GL_FLOAT, GL_FALSE, 0, 0 ) );
    }

    GL_EXEC( glEnableVertexAttribArray( id ) );
    return id;
}
}