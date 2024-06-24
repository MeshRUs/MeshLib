#include "MRRenderPointsObject.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRObjectPointsHolder.h"
#include "MRMesh/MRTimer.h"
#include "MRCreateShader.h"
#include "MRMesh/MRPointCloud.h"
#include "MRGLMacro.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRGLStaticHolder.h"
#include "MRRenderGLHelpers.h"
#include "MRRenderHelpers.h"
#include "MRViewer.h"
#include "MRGladGlfw.h"
#include "MRMesh/MRParallelFor.h"
#include "MRViewer/MRRenderDefaultObjects.h"

namespace MR
{

RenderPointsObject::RenderPointsObject( const VisualObject& visObj )
{
    objPoints_ = dynamic_cast< const ObjectPointsHolder* >( &visObj );
    assert( objPoints_ );
    if ( Viewer::constInstance()->isGLInitialized() )
        initBuffers_();
}

RenderPointsObject::~RenderPointsObject()
{
    freeBuffers_();
}

bool RenderPointsObject::render( const ModelRenderParams& renderParams )
{
    RenderModelPassMask desiredPass =
        !objPoints_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) ? RenderModelPassMask::NoDepthTest :
        ( objPoints_->getGlobalAlpha( renderParams.viewportId ) < 255 || objPoints_->getFrontColor( objPoints_->isSelected(), renderParams.viewportId ).a < 255 ) ? RenderModelPassMask::Transparent :
        RenderModelPassMask::Opaque;
    if ( !bool( renderParams.passMask & desiredPass ) )
        return false; // Nothing to draw in this pass.

    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objPoints_->resetDirty();
        return false;
    }
    update_();

    if ( !objPoints_->hasVisualRepresentation() )
        return false;

    // Initialize uniform
    GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
        ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );

    if ( objPoints_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    bindPoints_();

    // Send transformations to the GPU

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawPoints );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );
    if ( renderParams.normMatrixPtr )
    {
        GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, renderParams.normMatrixPtr->data() ) );
    }

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "invertNormals" ), objPoints_->getVisualizeProperty( VisualizeMaskType::InvertedNormals, renderParams.viewportId ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "perVertColoring" ), objPoints_->getColoringType() == ColoringType::VertsColorMap ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objPoints_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "hasNormals" ), int( hasNormalsBackup_ ) ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specExp" ), objPoints_->getShininess() ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specularStrength" ), objPoints_->getSpecularStrength() ) );
    float ambient = objPoints_->getAmbientStrength() * ( objPoints_->isSelected() ? SceneSettings::get( SceneSettings::FloatType::AmbientCoefSelectedObj ) : 1.0f );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "ambientStrength" ), ambient ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "globalAlpha" ), objPoints_->getGlobalAlpha( renderParams.viewportId ) / 255.0f ) );
    GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &renderParams.lightPos.x ) );

    const auto& backColor = Vector4f( objPoints_->getBackColor( renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "backColor" ), backColor[0], backColor[1], backColor[2], backColor[3] ) );

    const auto& mainColor = Vector4f( objPoints_->getFrontColor( objPoints_->isSelected(), renderParams.viewportId ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "mainColor" ), mainColor[0], mainColor[1], mainColor[2], mainColor[3] ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "showSelVerts" ), objPoints_->getVisualizeProperty( PointsVisualizePropertyType::SelectedVertices, renderParams.viewportId ) ) );
    const auto selectionColor = Vector4f( objPoints_->getSelectedVerticesColor( renderParams.viewportId ) );
    const auto selBackColor = Vector4f( backColor.x * selectionColor.x, backColor.y * selectionColor.y, backColor.z * selectionColor.z, backColor.w * selectionColor.w );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selectionColor" ), selectionColor[0], selectionColor[1], selectionColor[2], selectionColor[3] ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "selBackColor" ), selBackColor[0], selBackColor[1], selBackColor[2], selBackColor[3] ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::PointElementsNum, validIndicesSize_ );

#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), objPoints_->getPointSize() ) );
#else
    GL_EXEC( glPointSize( objPoints_->getPointSize() ) );
#endif
    GL_EXEC( glDepthFunc( getDepthFunctionLess( renderParams.depthFunction ) ) );
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )validIndicesSize_, GL_UNSIGNED_INT, 0 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );

    return true;
}

void RenderPointsObject::renderPicker( const ModelBaseRenderParams& parameters, unsigned geomId )
{
    if ( !Viewer::constInstance()->isGLInitialized() )
    {
        objPoints_->resetDirty();
        return;
    }
    update_();

    if ( !objPoints_->hasVisualRepresentation() )
        return;

    GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0, ( GLsizei )parameters.viewport.z, ( GLsizei )parameters.viewport.w ) );

    bindPointsPicker_();

    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, parameters.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, parameters.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, parameters.projMatrix.data() ) );

    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "primBucketSize" ), 1 ) );

    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ), objPoints_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, parameters.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        parameters.clipPlane.n.x, parameters.clipPlane.n.y, parameters.clipPlane.n.z, parameters.clipPlane.d ) );
    GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );
#ifdef __EMSCRIPTEN__
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "pointSize" ), objPoints_->getPointSize() ) );
#else
    GL_EXEC( glPointSize( objPoints_->getPointSize() ) );
#endif
    GL_EXEC( glDepthFunc( getDepthFunctionLess( parameters.depthFunction ) ) );
    GL_EXEC( glDrawElements( GL_POINTS, ( GLsizei )validIndicesSize_, GL_UNSIGNED_INT, 0 ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );
}

size_t RenderPointsObject::heapBytes() const
{
    return 0;
}

size_t RenderPointsObject::glBytes() const
{
    return vertPosBuffer_.size()
        + vertNormalsBuffer_.size()
        + vertColorsBuffer_.size()
        + validIndicesBuffer_.size()
        + vertSelectionTex_.size();
}

void RenderPointsObject::forceBindAll()
{
    update_();
    bindPoints_();
}

RenderBufferRef<Vector3f> RenderPointsObject::loadVertPosBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_POSITION ) || !objPoints_->pointCloud() )
        return glBuffer.prepareBuffer<Vector3f>( vertPosSize_, false );

    const auto step = objPoints_->getRenderDiscretization();
    const auto& points = objPoints_->pointCloud()->points;
    const auto num = objPoints_->pointCloud()->validPoints.find_last() + 1;
    if ( step == 1 )
        // we are sure that points will not be changed, so can do const_cast
        return RenderBufferRef<Vector3f>( const_cast< Vector3f* >( points.data() ), vertPosSize_ = num, !points.empty() );
    auto buffer = glBuffer.prepareBuffer<Vector3f>( vertPosSize_ = int( num / step ) );

    ParallelFor( VertId( 0 ), VertId( vertPosSize_ ), [&] ( VertId v )
    {
            buffer[v] = points[VertId( v * step )];
    } );

    return buffer;
}


RenderBufferRef<Vector3f> RenderPointsObject::loadVertNormalsBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_RENDER_NORMALS ) || !objPoints_->pointCloud() )
        return glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_, false );

    const auto& normals = objPoints_->pointCloud()->normals;
    int num = int( objPoints_->pointCloud()->validPoints.find_last() ) + 1;
    if ( normals.size() < num )
        num = 0;
    const auto step = objPoints_->getRenderDiscretization();
    if ( step == 1 )
        // we are sure that normals will not be changed, so can do const_cast
        return RenderBufferRef<Vector3f>( const_cast< Vector3f* >( normals.data() ), vertNormalsSize_ = num, !normals.empty() );

    auto buffer = glBuffer.prepareBuffer<Vector3f>( vertNormalsSize_ = int( num / step ) );

    ParallelFor( VertId( 0 ), VertId( vertNormalsSize_ ), [&] ( VertId v )
    {
        buffer[v] = normals[VertId( v * step )];
    } );

    return buffer;
}

RenderBufferRef<Color> RenderPointsObject::loadVertColorsBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_VERTS_COLORMAP ) || !objPoints_->pointCloud() || objPoints_->getVertsColorMap().empty() )
        return glBuffer.prepareBuffer<Color>( vertColorsSize_, false );

    const auto& colors = objPoints_->getVertsColorMap();
    const auto num = objPoints_->pointCloud()->validPoints.find_last() + 1;
    const auto step = objPoints_->getRenderDiscretization();
    if ( step == 1 )
        // we are sure that colors will not be changed, so can do const_cast
        return RenderBufferRef<Color>( const_cast< Color* >( colors.data() ), vertColorsSize_ = num, !colors.empty() );

    auto buffer = glBuffer.prepareBuffer<Color>( vertColorsSize_ = int( num / step ) );

    ParallelFor( VertId( 0 ), VertId( vertColorsSize_ ), [&] ( VertId v )
    {
        buffer[v] = colors[VertId( v * step )];
    } );

    return buffer;
}


void RenderPointsObject::bindPoints_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::DrawPoints );
    GL_EXEC( glBindVertexArray( pointsArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    if ( objPoints_->hasVisualRepresentation() )
    {
        auto pointCloud = objPoints_->pointCloud();

        const auto positions = loadVertPosBuffer_();
        bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );

        const auto normals = loadVertNormalsBuffer_();
        bindVertexAttribArray( shader, "normal", vertNormalsBuffer_, normals, 3, normals.dirty(), normals.glSize() != 0 );
        hasNormalsBackup_ = !pointCloud->normals.empty();
    }
    else
    {
        bindVertexAttribArray( shader, "position", vertPosBuffer_, std::vector<Vector3f>{}, 3, false, vertPosBuffer_.size() != 0 );
        bindVertexAttribArray( shader, "normal", vertNormalsBuffer_, std::vector<Vector3f>{}, 3, false, vertNormalsBuffer_.size() != 0 );
    }

    const auto colors = loadVertColorsBuffer_();
    bindVertexAttribArray( shader, "K", vertColorsBuffer_, colors, 4, colors.dirty(), colors.glSize() != 0 );

    auto validIndices = loadValidIndicesBuffer_();
    validIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, validIndices.dirty(), validIndices );

    // Selection
    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    auto vertSelectionTexture = loadVertSelectionTextureBuffer_();
    vertSelectionTex_.loadDataOpt( vertSelectionTexture.dirty(),
        {
            .resolution = vertSelectionTextureSize_,
            .internalFormat = GL_R32UI,
            .format = GL_RED_INTEGER,
            .type = GL_UNSIGNED_INT
        },
        vertSelectionTexture );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "selection" ), 0 ) );

    dirty_ &= ~DIRTY_MESH;
}

void RenderPointsObject::bindPointsPicker_()
{
    auto shader = GLStaticHolder::getShaderId( GLStaticHolder::Picker );
    GL_EXEC( glBindVertexArray( pointsPickerArrayObjId_ ) );
    GL_EXEC( glUseProgram( shader ) );
    if ( objPoints_->hasVisualRepresentation() )
    {
        const auto positions = loadVertPosBuffer_();
        bindVertexAttribArray( shader, "position", vertPosBuffer_, positions, 3, positions.dirty(), positions.glSize() != 0 );
    }
    else
        bindVertexAttribArray( shader, "position", vertPosBuffer_, std::vector<Vector3f>{}, 3, false, vertPosBuffer_.size() != 0 );

    auto validIndices = loadValidIndicesBuffer_();
    validIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, validIndices.dirty(), validIndices );
    dirty_ &= ~DIRTY_POSITION;
}

void RenderPointsObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &pointsArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( pointsArrayObjId_ ) );

    GL_EXEC( glGenVertexArrays( 1, &pointsPickerArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( pointsPickerArrayObjId_ ) );

    GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize_ ) );
    assert( maxTexSize_ > 0 );

    dirty_ = DIRTY_ALL;
}

void RenderPointsObject::freeBuffers_()
{
    if ( !Viewer::constInstance()->isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &pointsArrayObjId_ ) );
    GL_EXEC( glDeleteVertexArrays( 1, &pointsPickerArrayObjId_ ) );
}

void RenderPointsObject::update_()
{
    if ( cachedRenderDiscretization_ != objPoints_->getRenderDiscretization() )
    {
        cachedRenderDiscretization_ = objPoints_->getRenderDiscretization();
        dirty_ |= DIRTY_POSITION;
        dirty_ |= DIRTY_RENDER_NORMALS;
        dirty_ |= DIRTY_VERTS_COLORMAP;
        dirty_ |= DIRTY_SELECTION;
    }

    dirty_ |= objPoints_->getDirtyFlags();
    objPoints_->resetDirty();
}

RenderBufferRef<VertId> RenderPointsObject::loadValidIndicesBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_POSITION ) || !objPoints_->hasVisualRepresentation() )
        return glBuffer.prepareBuffer<VertId>( validIndicesSize_, !validIndicesBuffer_.valid() );

    const auto& points = objPoints_->pointCloud();
    const auto step = objPoints_->getRenderDiscretization();
    const auto num = objPoints_->pointCloud()->validPoints.find_last() + 1;    

    const auto& validPoints = points->validPoints;
    auto firstValid = validPoints.find_first();
    assert( firstValid );
    
    validIndicesSize_ = ( num / step );
    if ( step != 1 )
    {
        firstValid = {};
        for ( VertId v = VertId( (firstValid / step)*step  ); v < step * validIndicesSize_; v += step )
        {
            if ( validPoints.test( v ) )
            {
                firstValid = v;
                break;
            }
        }
        
        if ( !firstValid.valid() )
        {
            validIndicesSize_ = 0;
            return glBuffer.prepareBuffer<VertId>( 0 );
        }
    }

    auto buffer = glBuffer.prepareBuffer<VertId>( validIndicesSize_ );

    BitSetParallelForAll( validPoints, [&] ( VertId v )
    {
        if ( v % step != 0 || v >= step * validIndicesSize_ )
            return;

        if ( validPoints.test( v ) )
        {
            buffer[v / step] = VertId( v / step );
        }
        else
        {
            buffer[v / step] = VertId( firstValid / step );
        }
    });    

    return buffer;
}

RenderBufferRef<unsigned> RenderPointsObject::loadVertSelectionTextureBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_SELECTION ) || !objPoints_->hasVisualRepresentation() )
        return glBuffer.prepareBuffer<unsigned>( vertSelectionTextureSize_.x * vertSelectionTextureSize_.y,
            ( dirty_ & DIRTY_SELECTION ) && vertSelectionTextureSize_.x * vertSelectionTextureSize_.y == 0 );

    const auto& points = objPoints_->pointCloud();
    const auto step = objPoints_->getRenderDiscretization();
    const int num = points->validPoints.find_last() + 1;
    const auto numV = num / int( step );
    auto size = numV / 32 + 1;
    vertSelectionTextureSize_ = calcTextureRes( size, maxTexSize_ );
    assert( vertSelectionTextureSize_.x * vertSelectionTextureSize_.y >= size );
    auto buffer = glBuffer.prepareBuffer<unsigned>( vertSelectionTextureSize_.x * vertSelectionTextureSize_.y );

    const auto& selectedPoints = objPoints_->getSelectedPoints();
    const size_t selectionSize = selectedPoints.m_bits.size();
    
    const unsigned* selectionData = ( unsigned* )selectedPoints.m_bits.data();    

    ParallelFor( 0, ( int )buffer.size(), [&]( int r )
    {
        auto& block = buffer[r];
        block = 0;
        if ( r * step / 2 >= selectionSize )
            return;

        if ( step == 1 )
        {
            block = selectionData[r];
            return;
        }

        for ( int bit = 0; bit < 32; ++bit )
        {
            const int bitIndex = ( r * 32 + bit ) * int( step );
            if ( bitIndex >= selectionSize * 64 )
                continue;

            const auto selectionBit = std::div( bitIndex, 32 );
            if ( selectionData[selectionBit.quot] & ( 1 << ( selectionBit.rem ) ) )
                block |= 1 << bit;
        }
    } );

    return buffer;
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectPointsHolder, RenderObjectCombinator<RenderDefaultUiObject, RenderPointsObject> )

}
