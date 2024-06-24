#include "MRRenderMeshObject.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMeshNormals.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRPch/MRTBB.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRViewer/MRRenderDefaultObjects.h"

namespace MR
{

RenderMeshObject::RenderMeshObject( const VisualObject& visObj )
{
    objMesh_ = dynamic_cast< const ObjectMeshHolder* >( &visObj );
    assert( objMesh_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
}

RenderMeshObject::~RenderMeshObject()
{
    freeBuffers_();
}

bool RenderMeshObject::render( const ModelRenderParams& renderParams )
{
    RenderModelPassMask desiredPass =
        !objMesh_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) ? RenderModelPassMask::NoDepthTest :
        ( objMesh_->getGlobalAlpha( renderParams.viewportId ) < 255 || objMesh_->getFrontColor( objMesh_->isSelected(), renderParams.viewportId ).a < 255 || objMesh_->getBackColor( renderParams.viewportId ).a < 255 ) ? RenderModelPassMask::Transparent :
        RenderModelPassMask::Opaque;
    if ( !bool( renderParams.passMask & desiredPass ) )
        return false; // Nothing to draw in this pass.

    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objMesh_->resetDirty();
        return false;
    }
    update_( renderParams.viewportId );

    if ( renderParams.allowAlphaSort && desiredPass == RenderModelPassMask::Transparent )
    {
        GL_EXEC( glDepthMask( GL_FALSE ) );
        GL_EXEC( glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE ) );
#ifndef __EMSCRIPTEN__
        GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif
    }
    else
    {
        GL_EXEC( glDepthMask( GL_TRUE ) );
        GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );
#ifndef __EMSCRIPTEN__
        GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
    }

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
        ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );

    if ( objMesh_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    const bool useAlphaSort = renderParams.allowAlphaSort && desiredPass == RenderModelPassMask::Transparent;
    bindMesh_( useAlphaSort );

    auto shader = useAlphaSort ? GLStaticHolder::getShaderId( GLStaticHolder::TransparentMesh ) : GLStaticHolder::getShaderId( GLStaticHolder::Mesh );
    // Send transformations to the GPU
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );
    if ( renderParams.normMatrixPtr )
    {
        GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr->data() ) );
    }

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "onlyOddFragments" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::OnlyOddFragments, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "invertNormals" ), objMesh_->getVisualizeProperty( VisualizeMaskType::InvertedNormals, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "mirrored" ), renderParams.modelMatrix.det() < 0.0f ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "enableShading" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::EnableShading, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "flatShading" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::FlatShading, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objMesh_->getColoringType() == ColoringType::VertsColorMap ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perFaceColoring" ), objMesh_->getColoringType() == ColoringType::FacesColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( auto fixed_colori = glGetUniformLocation( shader, "fixed_color" ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specExp" ), objMesh_->getShininess() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specularStrength" ), objMesh_->getSpecularStrength() ) );
    float ambient = objMesh_->getAmbientStrength() * ( objMesh_->isSelected() ? SceneSettings::get( SceneSettings::FloatType::AmbientCoefSelectedObj ) : 1.0f );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "ambientStrength" ), ambient ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objMesh_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &renderParams.lightPos.x ) );
    GL_EXEC( glUniform4f( fixed_colori, 0.0, 0.0, 0.0, 0.0 ) );

    const auto mainColor = Vector4f( objMesh_->getFrontColor( objMesh_->isSelected(), renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "showSelFaces" ), objMesh_->getVisualizeProperty( MeshVisualizePropertyType::SelectedFaces, renderParams.viewportId ) ) );
    const auto selectionColor = Vector4f( objMesh_->getSelectedFacesColor( renderParams.viewportId ) );
    const auto backColor = Vector4f( objMesh_->getBackColor( renderParams.viewportId ) );
    const auto selectionBackfacesColor = Vector4f( backColor.x * selectionColor.x, backColor.y * selectionColor.y, backColor.z * selectionColor.z, backColor.w * selectionColor.w );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selectionColor" ), selectionColor[0], selectionColor[1], selectionColor[2], selectionColor[3] ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selBackColor" ), selectionBackfacesColor[0], selectionBackfacesColor[1], selectionBackfacesColor[2], selectionBackfacesColor[3] ) );

    // Render fill
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::Faces, renderParams.viewportId ) )
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

        // Texture
        GL_EXEC( auto useTexture = glGetUniformLocation( shader, "useTexture" ) );
        GL_EXEC( glUniform1i( useTexture, objMesh_->getVisualizeProperty( MeshVisualizePropertyType::Texture, renderParams.viewportId ) ||
            objMesh_->hasAncillaryTexture() ) );

        GL_EXEC( glDepthFunc( getDepthFunctionLess( renderParams.depthFunction ) ) );
        drawMesh_( true, renderParams.viewportId );
        GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
    }
    // Render wireframe
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::Edges, renderParams.viewportId ) )
        renderMeshEdges_( renderParams, useAlphaSort );
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::BordersHighlight, renderParams.viewportId ) )
        renderEdges_( renderParams, useAlphaSort, borderArrayObjId_, objMesh_->getBordersColor( renderParams.viewportId ), DIRTY_BORDER_LINES );
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::SelectedEdges, renderParams.viewportId ) )
        renderEdges_( renderParams, useAlphaSort, selectedEdgesArrayObjId_, objMesh_->getSelectedEdgesColor( renderParams.viewportId ), DIRTY_EDGES_SELECTION );

    if ( renderParams.allowAlphaSort && desiredPass == RenderModelPassMask::Transparent )
    {
        // enable back masks, disabled for alpha sort
        GL_EXEC( glDepthMask( GL_TRUE ) );
        GL_EXEC( glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE ) );
#ifndef __EMSCRIPTEN__
        GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
    }

    return true;
}

void RenderMeshObject::renderPicker( const ModelBaseRenderParams& parameters, unsigned geomId )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objMesh_->resetDirty();
        return;
    }
    update_( parameters.viewportId );

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );

    bindMeshPicker_();
#ifdef __EMSCRIPTEN__
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );
#else
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::MeshDesktopPicker );
#endif

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrix.data() ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 3 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y, parameters.clipPlane.n.z, parameters.clipPlane.d ) );
    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );

    GL_EXEC( glDepthFunc( getDepthFunctionLess( parameters.depthFunction ) ) );
    drawMesh_( true, parameters.viewportId, true );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
}

size_t RenderMeshObject::heapBytes() const
{
    return 0;
}

size_t RenderMeshObject::glBytes() const
{
    return borderBuffer_.size()
        + selectedEdgesBuffer_.size()
        + vertPosBuffer_.size()
        + vertUVBuffer_.size()
        + vertNormalsBuffer_.size()
        + vertColorsBuffer_.size()
        + facesIndicesBuffer_.size()
        + texture_.size()
        + faceSelectionTex_.size()
        + faceSelectionTex_.size()
        + facesNormalsTex_.size()
        + edgesTexture_.size()
        + selEdgesTexture_.size()
        + borderTexture_.size();
}

void RenderMeshObject::forceBindAll()
{
    update_( ViewportMask::all() );
    bindMesh_( false );
    bindEdges_();
    bindSelectedEdges_();
    bindBorders_();
}

void RenderMeshObject::renderEdges_( const ModelRenderParams& renderParams, bool alphaSort, GLuint vao, const Color& colorChar, uint32_t dirtyFlag )
{
    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( vao ) );

    auto shader = alphaSort ?
        GLStaticHolder::getShaderId( GLStaticHolder::TransparentLines ) :
        GLStaticHolder::getShaderId( GLStaticHolder::Lines );

    GL_EXEC( glUseProgram( shader ) );
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    int size = 0;
    switch ( dirtyFlag )
    {
    case DIRTY_BORDER_LINES:
        bindBorders_();
        size = bordersSize_;
        break;
    case DIRTY_EDGES_SELECTION:
        bindSelectedEdges_();
        size = selEdgeSize_;
        break;
    default:
        break;
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "vertices" ), 0 ) );
    bindEmptyTextures_( shader );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );

    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
        float( renderParams.viewport.x ), float( renderParams.viewport.y ),
        float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "width" ), objMesh_->getEdgeWidth() * 5 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), false ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perLineColoring" ), false ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y, renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    // colors
    auto color = Vector4f( colorChar );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ),
        color[0], color[1], color[2], color[3] ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objMesh_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 * size );

    GL_EXEC( glDepthFunc( getDepthFunctionLEqual( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, size * 6 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );

    dirty_ &= ~dirtyFlag;
}

void RenderMeshObject::renderMeshEdges_( const ModelRenderParams& renderParams, bool alphaSort )
{
    // Send lines data to GL, install lines properties
    GL_EXEC( glBindVertexArray( edgesArrayObjId_ ) );

    auto shader = alphaSort ?
        GLStaticHolder::getShaderId( GLStaticHolder::TransparentLines ) :
        GLStaticHolder::getShaderId( GLStaticHolder::Lines );

    GL_EXEC( glUseProgram( shader ) );

    // Positions
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    bindEdges_();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "vertices" ), 0 ) );
    bindEmptyTextures_( shader );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );

    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
        float( renderParams.viewport.x ), float( renderParams.viewport.y ),
        float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "width" ), objMesh_->getEdgeWidth() ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), false ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perLineColoring" ), false ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objMesh_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y, renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    // colors
    auto color = Vector4f( objMesh_->getEdgesColor( renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ),
        color[0], color[1], color[2], color[3] ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objMesh_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 2 * edgeSize_ );

    GL_EXEC( glDepthFunc( getDepthFunctionLess( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawArrays( GL_TRIANGLES, 0, edgeSize_ * 6 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
}

void RenderMeshObject::bindMesh_( bool alphaSort )
{
    auto shader = alphaSort ? GLStaticHolder::getShaderId( GLStaticHolder::TransparentMesh ) : GLStaticHolder::getShaderId( GLStaticHolder::Mesh );
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto normals = loadVertNormalsBuffer_();
    bindVertexAttribArray( shader, "normal", vertNormalsBuffer_, normals, 3, normals.dirty(), normals.glSize() != 0 );

    auto colormaps = loadVertColorsBuffer_();
    bindVertexAttribArray( shader, "K", vertColorsBuffer_, colormaps, 4, colormaps.dirty(), colormaps.glSize() != 0 );

    auto uvs = loadVertUVBuffer_();
    bindVertexAttribArray( shader, "texcoord", vertUVBuffer_, uvs, 2, uvs.dirty(), uvs.glSize() != 0 );

    auto faces = loadFaceIndicesBuffer_();
    facesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, faces.dirty(), faces );

    const auto& texture = objMesh_->hasAncillaryTexture() ? objMesh_->getAncillaryTexture() : objMesh_->getTexture();
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    texture_.loadDataOpt( dirty_ & DIRTY_TEXTURE,
        {
            .resolution = texture.resolution,
            .internalFormat = GL_RGBA,
            .format = GL_RGBA,
            .type = GL_UNSIGNED_BYTE,
            .wrap = texture.wrap,
            .filter = texture.filter
        },
        texture.pixels );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "tex" ), 0 ) );

    // Diffuse
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( dirty_ & DIRTY_PRIMITIVE_COLORMAP )
    {
        // TODO: avoid copying if no need to resize, and avoid double copying if resize is needed
        auto facesColorMap = objMesh_->getFacesColorMap();
        auto res = calcTextureRes( int( facesColorMap.size() ), maxTexSize_ );
        facesColorMap.resize( res.x * res.y );
        faceColorsTex_.loadData(
            { .resolution = res, .internalFormat = GL_RGBA8, .format = GL_RGBA, .type = GL_UNSIGNED_BYTE },
            facesColorMap );
    }
    else
        faceColorsTex_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "faceColors" ), 1 ) );

    // Normals
    auto faceNormals = loadFaceNormalsTextureBuffer_();
    GL_EXEC( glActiveTexture( GL_TEXTURE2 ) );
    facesNormalsTex_.loadDataOpt( faceNormals.dirty(),
        { .resolution = faceNormalsTextureSize_, .internalFormat = GL_RGBA32F, .format = GL_RGBA, .type = GL_FLOAT },
        faceNormals );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "faceNormals" ), 2 ) );

    // Selection
    auto faceSelection = loadFaceSelectionTextureBuffer_();
    GL_EXEC( glActiveTexture( GL_TEXTURE3 ) );
    faceSelectionTex_.loadDataOpt( faceSelection.dirty(),
        { .resolution = faceSelectionTextureSize_, .internalFormat = GL_R32UI, .format = GL_RED_INTEGER, .type = GL_UNSIGNED_INT },
        faceSelection );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 3 ) );

    dirty_ &= ~DIRTY_MESH;
    dirty_ &= ~DIRTY_VERTS_COLORMAP;
}

void RenderMeshObject::bindMeshPicker_()
{
#ifdef __EMSCRIPTEN__
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );
#else
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::MeshDesktopPicker );
#endif

    GL_EXEC( glBindVertexArray( meshPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );

    auto positions = loadVertPosBuffer_();
    bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

    auto faces = loadFaceIndicesBuffer_();
    facesIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, faces.dirty(), faces );

    dirty_ &= ~DIRTY_POSITION;
    dirty_ &= ~DIRTY_FACE;
}

void RenderMeshObject::bindEdges_()
{
    if ( !dirtyEdges_ || !objMesh_->mesh() )
    {
        edgesTexture_.bind();
        return;
    }
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    const auto& mesh = *objMesh_->mesh();
    const auto& topology = mesh.topology;
    auto lastValid = topology.lastNotLoneEdge();
    edgeSize_ = lastValid.valid() ? lastValid.undirected() + 1 : 0;
    auto res = calcTextureRes( int( edgeSize_ * 2 ), maxTexSize_ );
    auto positions = glBuffer.prepareBuffer<Vector3f>( res.x * res.y );
    tbb::parallel_for( tbb::blocked_range<int>( 0, int( edgeSize_ ) ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int ue = range.begin(); ue < range.end(); ++ue )
        {
            auto uEId = UndirectedEdgeId( ue );
            if ( topology.hasEdge( uEId ) )
            {
                positions[2 * ue] = mesh.orgPnt( uEId );
                positions[2 * ue + 1] = mesh.destPnt( uEId );
            }
            else
            {
                // important to be the same point so renderer does not rasterize these edges
                positions[2 * ue] = positions[2 * ue + 1] = Vector3f();
            }
        }
    } );
    edgesTexture_.loadData(
        { .resolution = res, .internalFormat = GL_RGB32UI, .format = GL_RGB_INTEGER, .type = GL_UNSIGNED_INT },
        positions );
    dirtyEdges_ = false;
}

void RenderMeshObject::bindBorders_()
{
    if ( !( dirty_ & DIRTY_BORDER_LINES ) || !objMesh_->mesh() )
    {
        borderTexture_.bind();
        return;
    }
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto boundary = findRightBoundary( topology );
    bordersSize_ = 0;
    for ( const auto& b : boundary )
        bordersSize_ += ( int )b.size();
    auto res = calcTextureRes( 2 * bordersSize_, maxTexSize_ );
    auto positions = glBuffer.prepareBuffer<Vector3f>( res.x * res.y );

    int i = 0;
    for ( const auto& b : boundary )
    {
        for ( auto e : b )
        {
            positions[i++] = mesh->orgPnt( e );
            positions[i++] = mesh->destPnt( e );
        }
    }
    borderTexture_.loadData(
        { .resolution = res, .internalFormat = GL_RGB32UI, .format = GL_RGB_INTEGER, .type = GL_UNSIGNED_INT },
        positions );
}

void RenderMeshObject::bindSelectedEdges_()
{
    if ( !( dirty_ & DIRTY_EDGES_SELECTION ) || !objMesh_->mesh() )
    {
        selEdgesTexture_.bind();
        return;
    }
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto selectedEdges = objMesh_->getSelectedEdges();
    for ( auto e : selectedEdges )
        if ( !topology.hasEdge( e ) )
            selectedEdges.reset( e );
    selEdgeSize_ = int( selectedEdges.count() );
    auto res = calcTextureRes( 2 * selEdgeSize_, maxTexSize_ );
    auto positions = glBuffer.prepareBuffer<Vector3f>( res.x * res.y );
    int i = 0;
    for ( auto e : selectedEdges )
    {
        positions[i++] = mesh->orgPnt( e );
        positions[i++] = mesh->destPnt( e );
    }
    selEdgesTexture_.loadData(
        { .resolution = res, .internalFormat = GL_RGB32UI, .format = GL_RGB_INTEGER, .type = GL_UNSIGNED_INT },
        positions );
}

void RenderMeshObject::bindEmptyTextures_(GLuint shaderId)
{
    // VertColors
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( !emptyVertsColorTexture_.valid() )
        emptyVertsColorTexture_.gen();
    emptyVertsColorTexture_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shaderId, "vertColors" ), 1 ) );

    // LineColors
    GL_EXEC( glActiveTexture( GL_TEXTURE2 ) );
    // bind empty texture
    if ( !emptyLinesColorTexture_.valid() )
        emptyLinesColorTexture_.gen();
    emptyLinesColorTexture_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shaderId, "lineColors" ), 2 ) );
}

void RenderMeshObject::drawMesh_( bool /*solid*/, ViewportId viewportId, bool picker ) const
{
    /* Avoid Z-buffer fighting between filled triangles & wireframe lines */
    GL_EXEC( glEnable( GL_POLYGON_OFFSET_FILL ) );
    if ( ( objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::Edges )
         | objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::PolygonOffsetFromCamera )
       // intentionally do not check selected edges and borders since they are typically thicker and include not all edges
       //  | objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::SelectedEdges )
       //  | objMesh_->getVisualizePropertyMask( MeshVisualizePropertyType::BordersHighlight )
        ).contains( viewportId ) )
    {
        // offset triangles further with factor depending on triangle orientation to clearly see edges on top of them
        GL_EXEC( glPolygonOffset( 1.0, 1.0 ) );
    }
    else
    {
        // offset all triangles on fixed amount to avoid halo effect in flat shading mode
        GL_EXEC( glPolygonOffset( 0.0, 1.0 ) );
    }

    if ( !picker )
        getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleElementsNum, faceIndicesSize_ );

    GL_EXEC( glDrawElements( GL_TRIANGLES, int( 3 * faceIndicesSize_ ), GL_UNSIGNED_INT, nullptr ) );

    GL_EXEC( glDisable( GL_POLYGON_OFFSET_FILL ) );
}

void RenderMeshObject::initBuffers_()
{
    // Mesh: Vertex Array Object & Buffer objects
    GL_EXEC( glGenVertexArrays( 1, &meshArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( meshArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &edgesArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( edgesArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &meshPickerArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( meshPickerArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &borderArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( borderArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &selectedEdgesArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( selectedEdgesArrayObjId_ ) );

    GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize_ ) );
    assert( maxTexSize_ > 0 );

    dirty_ = DIRTY_ALL - DIRTY_CORNERS_RENDER_NORMAL - DIRTY_VERTS_RENDER_NORMAL;
}

void RenderMeshObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &meshArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &edgesArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &meshPickerArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &borderArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &selectedEdgesArrayObjId_ ) );
}

void RenderMeshObject::update_( ViewportMask mask )
{
    MR_TIMER;
    auto objDirty = objMesh_->getDirtyFlags();
    uint32_t dirtyNormalFlag = objMesh_->getNeededNormalsRenderDirtyValue( mask );
    if ( dirtyNormalFlag & DIRTY_FACES_RENDER_NORMAL )
    {
        // vertNormalsBufferObj_ should be valid no matter what normals we use
        if ( objMesh_->creases().none() )
            dirtyNormalFlag |= DIRTY_VERTS_RENDER_NORMAL;
        else
            dirtyNormalFlag |= DIRTY_CORNERS_RENDER_NORMAL;
    }
    objDirty &= ~( DIRTY_RENDER_NORMALS - dirtyNormalFlag );
    dirty_ |= objDirty;

    if ( dirty_ & DIRTY_FACE || dirty_ & DIRTY_POSITION )
        dirtyEdges_ = true;

    objMesh_->resetDirtyExeptMask( DIRTY_RENDER_NORMALS - dirtyNormalFlag );

#ifndef __EMSCRIPTEN__
    if ( !cornerMode && bool( dirty_ & DIRTY_CORNERS_RENDER_NORMAL ) )
    {
        // always need corner mode for creases
        // it should not affect dirtyEdges_
        cornerMode = true;
        dirty_ |= DIRTY_POSITION;
        dirty_ |= DIRTY_VERTS_COLORMAP;
        dirty_ |= DIRTY_UV;
        dirty_ |= DIRTY_FACE;
    }
    if ( cornerMode && bool( dirty_ & DIRTY_VERTS_RENDER_NORMAL ) )
    {
        assert( objMesh_->creases().none() );
        // disable corner mode if no creases
        // it should not affect dirtyEdges_
        cornerMode = false;
        dirty_ |= DIRTY_POSITION;
        dirty_ |= DIRTY_VERTS_COLORMAP;
        dirty_ |= DIRTY_UV;
        dirty_ |= DIRTY_FACE;
    }
#endif
}

RenderBufferRef<Vector3f> RenderMeshObject::loadVertPosBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_POSITION ) || !objMesh_->mesh() )
        return glBuffer.prepareBuffer<Vector3f>( vertPosSize_, false );

    MR_NAMED_TIMER( "vertbased_dirty_positions" );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    if ( cornerMode )
    {
        auto numF = topology.lastValidFace() + 1;
        auto buffer = glBuffer.prepareBuffer<Vector3f>( vertPosSize_ = 3 * numF );

        tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId{ numF } ), [&] ( const tbb::blocked_range<FaceId>& range )
        {
            for ( FaceId f = range.begin(); f < range.end(); ++f )
            {
                if ( !mesh->topology.hasFace( f ) )
                    continue;
                auto ind = 3 * f;
                Vector3f v[3];
                mesh->getTriPoints( f, v[0], v[1], v[2] );
                for ( int i = 0; i < 3; ++i )
                    buffer[ind + i] = v[i];
            }
        } );

        return buffer;
    }
    else
    {
        auto buffer = glBuffer.prepareBuffer<Vector3f>( vertPosSize_ = topology.lastValidVert() + 1 );
        std::copy( MR::begin( mesh->points ), MR::begin( mesh->points ) + vertPosSize_, buffer.data() );
        return buffer;
    }
}

RenderBufferRef<Vector3f> RenderMeshObject::loadVertNormalsBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !objMesh_->mesh() )
        return glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;

    if ( dirty_ & DIRTY_CORNERS_RENDER_NORMAL )
    {
        MR_NAMED_TIMER( "dirty_corners_normals" )

        auto buffer = glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = 3 * numF );

        const auto& creases = objMesh_->creases();

        const auto cornerNormals = computePerCornerNormals( *mesh, creases.any() ? &creases : nullptr );
        tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId{ numF } ), [&] ( const tbb::blocked_range<FaceId>& range )
        {
            for ( FaceId f = range.begin(); f < range.end(); ++f )
            {
                if ( !mesh->topology.hasFace( f ) )
                    continue;
                auto ind = 3 * f;
                const auto& cornerN = getAt( cornerNormals, f );
                for ( int i = 0; i < 3; ++i )
                    buffer[ind + i] = cornerN[i];
            }
        } );

        return buffer;
    }
    else if ( dirty_ & DIRTY_VERTS_RENDER_NORMAL )
    {
        MR_NAMED_TIMER( "dirty_vertices_normals" );

        const auto vertNormals = computePerVertNormals( *mesh );
        if ( cornerMode )
        {
            auto buffer = glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = 3 * numF );

            tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId{ numF } ), [&] ( const tbb::blocked_range<FaceId>& range )
            {
                for ( FaceId f = range.begin(); f < range.end(); ++f )
                {
                    if ( !mesh->topology.hasFace( f ) )
                        continue;
                    auto ind = 3 * f;
                    VertId v[3];
                    topology.getTriVerts( f, v );
                    for ( int i = 0; i < 3; ++i )
                    {
                        const auto& norm = getAt( vertNormals, v[i] );
                        buffer[ind + i] = norm;
                    }
                }
            } );

            return buffer;
        }
        else
        {
            auto buffer = glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = topology.lastValidVert() + 1 );
            std::copy( MR::begin( vertNormals ), MR::end( vertNormals ), buffer.data() );
            return buffer;
        }
    }
    else
    {
        return glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_, false );
    }
}

RenderBufferRef<Color> RenderMeshObject::loadVertColorsBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_VERTS_COLORMAP ) || !objMesh_->mesh() )
        return glBuffer.prepareBuffer<Color>( vertColorsSize_, false ); // use updated color map
    if ( objMesh_->getColoringType() != ColoringType::VertsColorMap )
        return glBuffer.prepareBuffer<Color>( vertColorsSize_ = 0 ); // clear color map if not used
    
    MR_NAMED_TIMER( "vert_colormap" );
    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    const auto& vertsColorMap = objMesh_->getVertsColorMap();

    if ( cornerMode )
    {
        auto numF = topology.lastValidFace() + 1;
        auto buffer = glBuffer.prepareBuffer<Color>( vertColorsSize_ = 3 * numF );

        tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId{ numF } ), [&] ( const tbb::blocked_range<FaceId>& range )
        {
            for ( FaceId f = range.begin(); f < range.end(); ++f )
            {
                if ( !mesh->topology.hasFace( f ) )
                    continue;
                auto ind = 3 * f;
                VertId v[3];
                topology.getTriVerts( f, v );
                for ( int i = 0; i < 3; ++i )
                    buffer[ind + i] = getAt( vertsColorMap, v[i] );
            }
        } );

        return buffer;
    }
    else
    {
        auto buffer = glBuffer.prepareBuffer<Color>( vertColorsSize_ = topology.lastValidVert() + 1 );
        std::copy( MR::begin( vertsColorMap ), MR::begin( vertsColorMap ) + vertColorsSize_, buffer.data() );
        return buffer;
    }
}

RenderBufferRef<UVCoord> RenderMeshObject::loadVertUVBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_UV ) || !objMesh_->mesh() )
        return glBuffer.prepareBuffer<UVCoord>( vertUVSize_, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto numV = topology.lastValidVert() + 1;

    const auto& uvCoords = objMesh_->hasAncillaryTexture() ? objMesh_->getAncillaryUVCoords() : objMesh_->getUVCoords();
    if ( objMesh_->getVisualizeProperty( MeshVisualizePropertyType::Texture, ViewportMask::any() ) )
    {
        assert( uvCoords.size() >= numV );
    }
    if ( uvCoords.size() >= numV )
    {
        if ( cornerMode )
        {
            auto buffer = glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = 3 * numF );

            tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId{ numF } ), [&] ( const tbb::blocked_range<FaceId>& range )
            {
                for ( FaceId f = range.begin(); f < range.end(); ++f )
                {
                    if ( !mesh->topology.hasFace( f ) )
                        continue;
                    auto ind = 3 * f;
                    VertId v[3];
                    topology.getTriVerts( f, v );
                    for ( int i = 0; i < 3; ++i )
                        buffer[ind + i] = getAt( uvCoords, v[i] );
                }
            } );

            return buffer;
        }
        else
        {
            auto buffer = glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = numV );
            std::copy( MR::begin( uvCoords ), MR::begin( uvCoords ) + numV, buffer.data() );
            return buffer;
        }
    }
    else
    {
        return glBuffer.prepareBuffer<UVCoord>( vertUVSize_ = 0 );
    }
}

RenderBufferRef<Vector3i> RenderMeshObject::loadFaceIndicesBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_FACE ) || !objMesh_->mesh() )
        return glBuffer.prepareBuffer<Vector3i>( faceIndicesSize_, !facesIndicesBuffer_.valid() );

    // CORNDER BASED

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;
    auto buffer = glBuffer.prepareBuffer<Vector3i>( faceIndicesSize_ = numF );

    tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId{ numF } ), [&] ( const tbb::blocked_range<FaceId>& range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            if ( !topology.hasFace( f ) )
                buffer[f] = Vector3i();
            else
            {
                if ( cornerMode )
                {
                    auto ind = 3 * f;
                    buffer[f] = Vector3i{ ind, ind + 1, ind + 2 };
                }
                else
                {
                    topology.getTriVerts( f, ( ThreeVertIds& )buffer[f] );
                }
            }
        }
    } );

    return buffer;
}

RenderBufferRef<unsigned> RenderMeshObject::loadFaceSelectionTextureBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_SELECTION ) || !objMesh_->mesh() )
        return glBuffer.prepareBuffer<unsigned>( faceSelectionTextureSize_.x * faceSelectionTextureSize_.y, false );

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;

    auto size = numF / 32 + 1;
    faceSelectionTextureSize_ = calcTextureRes( size, maxTexSize_ );
    assert( faceSelectionTextureSize_.x * faceSelectionTextureSize_.y >= size );
    auto buffer = glBuffer.prepareBuffer<unsigned>( faceSelectionTextureSize_.x * faceSelectionTextureSize_.y );

    const auto& selection = objMesh_->getSelectedFaces().m_bits;
    const unsigned* selectionData = ( unsigned* )selection.data();
    tbb::parallel_for( tbb::blocked_range<int>( 0, (int)buffer.size() ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int r = range.begin(); r < range.end(); ++r )
        {
            auto& block = buffer[r];
            if ( r / 2 >= selection.size() )
            {
                block = 0;
                continue;
            }
            block = selectionData[r];
        }
    } );

    return buffer;
}

RenderBufferRef<Vector4f> RenderMeshObject::loadFaceNormalsTextureBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_FACES_RENDER_NORMAL ) || !objMesh_->mesh() )
        return glBuffer.prepareBuffer<Vector4f>( faceNormalsTextureSize_.x * faceNormalsTextureSize_.y, false );

    MR_NAMED_TIMER( "dirty_faces_normals" )

    const auto& mesh = objMesh_->mesh();
    const auto& topology = mesh->topology;
    auto numF = topology.lastValidFace() + 1;

    faceNormalsTextureSize_ = calcTextureRes( numF, maxTexSize_ );
    assert( faceNormalsTextureSize_.x * faceNormalsTextureSize_.y >= numF );
    auto buffer = glBuffer.prepareBuffer<Vector4f>( faceNormalsTextureSize_.x * faceNormalsTextureSize_.y );

    computePerFaceNormals4( *mesh, buffer.data(), buffer.size() );

    return buffer;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectMeshHolder, RenderObjectCombinator<RenderDefaultUiObject, RenderMeshObject> )

}
