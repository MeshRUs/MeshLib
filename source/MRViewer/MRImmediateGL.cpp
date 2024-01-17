#include "MRImmediateGL.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRMatrix4.h"

namespace MR
{

namespace ImmediateGL
{

void drawTris( const std::vector<Triangle3f>& tris, const std::vector<TriCornerColors>& colors, const ImmediateGL::TriRenderParams& params )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
        return;
    // set GL_DEPTH_TEST specified for points 
    GLuint quadVAO;
    GL_EXEC( glGenVertexArrays( 1, &quadVAO ) );
    GlBuffer quadBuffer, quadColorBuffer, quadNormalBuffer;

    // set GL_DEPTH_TEST specified for lines
    if ( params.depthTest )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glViewport( ( GLsizei )params.viewport.x, ( GLsizei )params.viewport.y,
        ( GLsizei )params.viewport.z, ( GLsizei )params.viewport.w ) );
    // Send lines data to GL, install lines properties 
    GL_EXEC( glBindVertexArray( quadVAO ) );

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::AdditionalQuad );
    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, params.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, params.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, params.projMatrix.data() ) );
    auto normM = ( params.viewMatrix * params.modelMatrix ).inverse().transposed();
    if ( normM.det() == 0 )
    {
        auto norm = normM.norm();
        if ( std::isnormal( norm ) )
        {
            normM /= norm;
            normM.w = { 0, 0, 0, 1 };
        }
        else
        {
            spdlog::warn( "Object transform is degenerate" );
        }
    }
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, normM.data() ) );

    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &params.lightPos.x ) );

    GL_EXEC( GLint colorsId = glGetAttribLocation( shader, "color" ) );
    quadColorBuffer.loadData( GL_ARRAY_BUFFER, colors );
    GL_EXEC( glVertexAttribPointer( colorsId, 4, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( colorsId ) );

    GL_EXEC( GLint normalId = glGetAttribLocation( shader, "normal" ) );
    Buffer<Vector3f> normals( tris.size() * 3 );
    for ( int i = 0; i < tris.size(); ++i )
    {
        auto* norm = &normals[i * 3];
        norm[0] = norm[1] = norm[2] = cross( tris[i][2] - tris[i][0], tris[i][1] - tris[i][0] ).normalized();
    }
    quadNormalBuffer.loadData( GL_ARRAY_BUFFER, normals );
    GL_EXEC( glVertexAttribPointer( normalId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( normalId ) );

    GL_EXEC( GLint positionId = glGetAttribLocation( shader, "position" ) );
    quadBuffer.loadData( GL_ARRAY_BUFFER, tris );
    GL_EXEC( glVertexAttribPointer( positionId, 3, GL_FLOAT, GL_FALSE, 0, 0 ) );
    GL_EXEC( glEnableVertexAttribArray( positionId ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, tris.size() );

    GL_EXEC( glBindVertexArray( quadVAO ) );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, 3 * int( tris.size() ) ) );

    GL_EXEC( glDeleteVertexArrays( 1, &quadVAO ) );
}

} //namespace ImmediateGL

} //namespace MR
