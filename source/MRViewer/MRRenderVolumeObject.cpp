#include "MRRenderVolumeObject.h"
#ifndef MRMESH_NO_OPENVDB
#include "MRMesh/MRObjectVoxels.h"
#include "MRViewer.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRGLStaticHolder.h"
#include "MRMesh/MRFloatGrid.h"
#include "MRMesh/MRMatrix4.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRPch/MRTBB.h"
#include "MRViewer/MRRenderDefaultObjects.h"

namespace MR
{

RenderVolumeObject::RenderVolumeObject( const VisualObject& visObj )
{
    objVoxels_ = dynamic_cast< const ObjectVoxels* >( &visObj );
    assert( objVoxels_ );
    if ( getViewerInstance().isGLInitialized() )
        initBuffers_();
}

RenderVolumeObject::~RenderVolumeObject()
{
    freeBuffers_();
}

bool RenderVolumeObject::render( const ModelRenderParams& renderParams )
{
    RenderModelPassMask desiredPass =
        !objVoxels_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) ? RenderModelPassMask::NoDepthTest :
        ( objVoxels_->getGlobalAlpha( renderParams.viewportId ) < 255 || objVoxels_->getFrontColor( objVoxels_->isSelected(), renderParams.viewportId ).a < 255 ) ? RenderModelPassMask::Transparent :
        RenderModelPassMask::Opaque;
    if ( !bool( renderParams.passMask & desiredPass ) )
        return false; // Nothing to draw in this pass.

    render_( renderParams, &renderParams, unsigned( ~0 ) );

    return true;
}

void RenderVolumeObject::renderPicker( const ModelBaseRenderParams& renderParams, unsigned geomId )
{
    render_( renderParams, nullptr, geomId );
}

size_t RenderVolumeObject::heapBytes() const
{
    return 0;
}

size_t RenderVolumeObject::glBytes() const
{
    return volume_.size() * sizeof( uint16_t )
        + denseMap_.size();
}

void RenderVolumeObject::forceBindAll()
{
    update_();
    bindVolume_( true );
}

RenderBufferRef<unsigned> RenderVolumeObject::loadActiveVoxelsTextureBuffer_()
{
    auto& glBuffer = GLStaticHolder::getStaticGLBuffer();
    if ( !( dirty_ & DIRTY_SELECTION ) || !objVoxels_->vdbVolume().data )
        return glBuffer.prepareBuffer<unsigned>( activeVoxelsTextureSize_.x * activeVoxelsTextureSize_.y, false );

    const auto& dims = objVoxels_->vdbVolume().dims;
    auto numV = dims.x * dims.y * dims.z;

    auto size = numV / 32 + 1;
    activeVoxelsTextureSize_ = calcTextureRes( size, maxTexSize_ );
    assert( activeVoxelsTextureSize_.x * activeVoxelsTextureSize_.y >= size );
    auto buffer = glBuffer.prepareBuffer<unsigned>( activeVoxelsTextureSize_.x * activeVoxelsTextureSize_.y );


    if ( objVoxels_->getVolumeRenderActiveVoxels().empty() )
    {
        tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )buffer.size() ), [&] ( const tbb::blocked_range<int>& range )
        {
            for ( int r = range.begin(); r < range.end(); ++r )
                buffer[r] = 0xFFFFFFFF;
        } );
        return buffer;
    }
    const auto& activeVoxels = objVoxels_->getVolumeRenderActiveVoxels().m_bits;
    const unsigned* activeVoxelsData = ( unsigned* )activeVoxels.data();
    tbb::parallel_for( tbb::blocked_range<int>( 0, ( int )buffer.size() ), [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int r = range.begin(); r < range.end(); ++r )
            buffer[r] = activeVoxelsData[r];
    } );

    return buffer;
}

void RenderVolumeObject::render_( const ModelBaseRenderParams& renderParams, const ModelRenderParams* nonPickerParams, unsigned geomId )
{
    if ( !getViewerInstance().isGLInitialized() )
    {
        objVoxels_->resetDirty();
        return;
    }
    update_();

    bool picker = geomId != unsigned( ~0 );
    // Initialize uniform
    if ( picker )
    {
        GL_EXEC( glViewport( ( GLsizei )0, ( GLsizei )0,
            ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );
    }
    else
    {
        GL_EXEC( glViewport( ( GLsizei )renderParams.viewport.x, ( GLsizei )renderParams.viewport.y,
            ( GLsizei )renderParams.viewport.z, ( GLsizei )renderParams.viewport.w ) );
    }

    if ( objVoxels_->getVisualizeProperty( VisualizeMaskType::DepthTest, renderParams.viewportId ) )
    {
        GL_EXEC( glEnable( GL_DEPTH_TEST ) );
    }
    else
    {
        GL_EXEC( glDisable( GL_DEPTH_TEST ) );
    }

    GL_EXEC( glEnable( GL_BLEND ) );
    GL_EXEC( glBlendFuncSeparate( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA ) );

    bindVolume_( picker );
    auto shader = picker ? GLStaticHolder::getShaderId( GLStaticHolder::VolumePicker ) :
        GLStaticHolder::getShaderId( GLStaticHolder::Volume );

    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "model" ), 1, GL_TRUE, renderParams.modelMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "view" ), 1, GL_TRUE, renderParams.viewMatrix.data() ) );
    GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "proj" ), 1, GL_TRUE, renderParams.projMatrix.data() ) );
    if ( !picker )
    {
        if ( nonPickerParams->normMatrixPtr )
        {
            GL_EXEC( glUniformMatrix4fv( glGetUniformLocation( shader, "normal_matrix" ), 1, GL_TRUE, nonPickerParams->normMatrixPtr->data() ) );
        }
        GL_EXEC( glUniform3fv( glGetUniformLocation( shader, "ligthPosEye" ), 1, &nonPickerParams->lightPos.x ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specExp" ), objVoxels_->getShininess() ) );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "specularStrength" ), objVoxels_->getSpecularStrength() ) );
        float ambient = objVoxels_->getAmbientStrength() * ( objVoxels_->isSelected() ? SceneSettings::get( SceneSettings::FloatType::AmbientCoefSelectedObj ) : 1.0f );
        GL_EXEC( glUniform1f( glGetUniformLocation( shader, "ambientStrength" ), ambient ) );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "useClippingPlane" ),
        objVoxels_->getVisualizeProperty( VisualizeMaskType::ClippedByPlane, renderParams.viewportId ) ) );
    GL_EXEC( glUniform4f( glGetUniformLocation( shader, "clippingPlane" ),
        renderParams.clipPlane.n.x, renderParams.clipPlane.n.y,
        renderParams.clipPlane.n.z, renderParams.clipPlane.d ) );

    if ( geomId != unsigned( ~0 ) )
    {
        GL_EXEC( glUniform1ui( glGetUniformLocation( shader, "uniGeomId" ), geomId ) );
    }

    if ( picker )
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
            float( 0 ), float( 0 ),
            float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
    }
    else
    {
        GL_EXEC( glUniform4f( glGetUniformLocation( shader, "viewport" ),
            float( renderParams.viewport.x ), float( renderParams.viewport.y ),
            float( renderParams.viewport.z ), float( renderParams.viewport.w ) ) );
    }

    const auto& voxelSize = objVoxels_->vdbVolume().voxelSize;
    auto minCorner = Vector3f( objVoxels_->getActiveBounds().min ) - Vector3f::diagonal( 0.5f );
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "minCorner" ), minCorner.x, minCorner.y, minCorner.z ) );
    GL_EXEC( glUniform3f( glGetUniformLocation( shader, "voxelSize" ), voxelSize.x, voxelSize.y, voxelSize.z ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "step" ), std::min( { voxelSize.x, voxelSize.y, voxelSize.z } ) ) );


    constexpr std::array<float, 24> cubePoints =
    {
        0.0f,0.0f,0.0f,
        0.0f,1.0f,0.0f,
        1.0f,1.0f,0.0f,
        1.0f,0.0f,0.0f,
        0.0f,0.0f,1.0f,
        0.0f,1.0f,1.0f,
        1.0f,1.0f,1.0f,
        1.0f,0.0f,1.0f
    };
    constexpr std::array<unsigned, 36> cubeTriangles =
    {
        0,1,2,
        2,3,0,
        0,4,5,
        5,1,0,
        0,3,7,
        7,4,0,
        6,5,4,
        4,7,6,
        1,5,6,
        6,2,1,
        6,7,3,
        3,2,6
    };

    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );
    bindVertexAttribArray( shader, "position", volumeVertsBuffer_, cubePoints, 3, !volumeVertsBuffer_.valid() );
    volumeIndicesBuffer_.loadDataOpt( GL_ELEMENT_ARRAY_BUFFER, !volumeIndicesBuffer_.valid(),
        cubeTriangles.data(), cubeTriangles.size() );

    getViewerInstance().incrementThisFrameGLPrimitivesCount( Viewer::GLPrimitivesType::TriangleArraySize, 12 );
#ifndef __EMSCRIPTEN__
    GL_EXEC( glDisable( GL_MULTISAMPLE ) );
#endif
    GL_EXEC( glEnable( GL_CULL_FACE ) );
    GL_EXEC( glCullFace( GL_BACK ) );

    // currently only less supported for volume rendering
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Less ) ) );
    GL_EXEC( glDrawElements( GL_TRIANGLES, 36, GL_UNSIGNED_INT, nullptr ) );
    GL_EXEC( glDepthFunc( getDepthFunctionLess( DepthFunction::Default ) ) );

    GL_EXEC( glDisable( GL_CULL_FACE ) );
#ifndef __EMSCRIPTEN__
    GL_EXEC( glEnable( GL_MULTISAMPLE ) );
#endif
}

void RenderVolumeObject::bindVolume_( bool picker )
{
    auto shader = picker ? GLStaticHolder::getShaderId( GLStaticHolder::VolumePicker ) :
        GLStaticHolder::getShaderId( GLStaticHolder::Volume );

    const auto& params = objVoxels_->getVolumeRenderingParams();

    GL_EXEC( glUseProgram( shader ) );

    GL_EXEC( glActiveTexture( GL_TEXTURE0 ) );
    if ( dirty_ & DIRTY_PRIMITIVES )
    {
        auto volume = objVoxels_->getVolumeRenderingData();
        if ( !volume )
        {
            objVoxels_->prepareDataForVolumeRendering();
            volume = objVoxels_->getVolumeRenderingData();
        }
        assert( volume );
        volume_.loadData(
            {
                .resolution = volume->dims,
                .internalFormat = GL_R16F,
                .format = GL_RED,
                .type = GL_FLOAT,
                .filter = params.volumeFilterType
            },
            volume->data );
    }
    else
    {
        volume_.bind();
        setTextureFilterType( params.volumeFilterType, true );
    }
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "volume" ), 0 ) );

    const auto& volume = objVoxels_->vdbVolume();
    GL_EXEC( glActiveTexture( GL_TEXTURE1 ) );
    if ( dirty_ & DIRTY_TEXTURE )
    {
        std::vector<Color> denseMap;
        bool grayShades = params.lutType == ObjectVoxels::VolumeRenderingParams::LutType::GrayShades;
        if ( grayShades ||
             params.lutType == ObjectVoxels::VolumeRenderingParams::LutType::OneColor )
        {
            denseMap.resize( 2 );
            denseMap[0] = grayShades ? Color::white() : params.oneColor;
            denseMap[1] = grayShades ? Color::black() : params.oneColor;
            switch ( params.alphaType )
            {
            case ObjectVoxels::VolumeRenderingParams::AlphaType::LinearIncreasing :
                denseMap[0].a = 0;
                denseMap[1].a = params.alphaLimit;
                break;
            case ObjectVoxels::VolumeRenderingParams::AlphaType::LinearDecreasing:
                denseMap[0].a = params.alphaLimit;
                denseMap[1].a = 0;
                break;
            case ObjectVoxels::VolumeRenderingParams::AlphaType::Constant:
            default:
                denseMap[0].a = denseMap[1].a = params.alphaLimit;
                break;
            }
        }
        else if ( params.lutType == ObjectVoxels::VolumeRenderingParams::LutType::Rainbow )
        {
            denseMap = {
                Color::red(),
                Color( 255,127,0 ),
                Color::yellow(),
                Color::green(),
                Color::blue(),
                Color( 75,0,130 ),
                Color( 148,0,211 )
            };
            float alphaStep = float( params.alphaLimit ) / denseMap.size();

            for ( int i = 0; i < denseMap.size(); ++i )
            {
                switch ( params.alphaType )
                {
                case ObjectVoxels::VolumeRenderingParams::AlphaType::LinearIncreasing:
                    denseMap[i].a = uint8_t( std::clamp( i * alphaStep, 0.0f, float( params.alphaLimit ) ) );
                    break;
                case ObjectVoxels::VolumeRenderingParams::AlphaType::LinearDecreasing:
                    denseMap[int( denseMap.size() ) - i - 1].a = uint8_t( std::clamp( i * alphaStep, 0.0f, float( params.alphaLimit ) ) );
                    break;
                case ObjectVoxels::VolumeRenderingParams::AlphaType::Constant:
                default:
                    denseMap[i].a = params.alphaLimit;
                    break;
                }
            }
        }
        denseMap_.loadData(
            {
                .resolution = Vector2i( (int)denseMap.size(), 1 ),
                .internalFormat = GL_RGBA8,
                .format = GL_RGBA,
                .type = GL_UNSIGNED_BYTE,
                .filter = FilterType::Linear },
            denseMap );
    }
    else
        denseMap_.bind();
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "denseMap" ), 1 ) );

    // Active Voxels
    auto activeVoxels = loadActiveVoxelsTextureBuffer_();
    GL_EXEC( glActiveTexture( GL_TEXTURE2 ) );
    activeVoxelsTex_.loadDataOpt( activeVoxels.dirty(),
        { .resolution = activeVoxelsTextureSize_, .internalFormat = GL_R32UI, .format = GL_RED_INTEGER, .type = GL_UNSIGNED_INT },
        activeVoxels );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "activeVoxels" ), 2 ) );

    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "minValue" ), ( params.min - volume.min ) / ( volume.max - volume.min ) ) );
    GL_EXEC( glUniform1f( glGetUniformLocation( shader, "maxValue" ), ( params.max - volume.min ) / ( volume.max - volume.min ) ) );
    GL_EXEC( glUniform1i( glGetUniformLocation( shader, "shadingMode" ), int( params.shadingType ) ) );

    dirty_ &= ~( DIRTY_PRIMITIVES | DIRTY_TEXTURE | DIRTY_SELECTION );
}

void RenderVolumeObject::initBuffers_()
{
    GL_EXEC( glGenVertexArrays( 1, &volumeArrayObjId_ ) );
    GL_EXEC( glBindVertexArray( volumeArrayObjId_ ) );

    GL_EXEC( glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTexSize_ ) );
    assert( maxTexSize_ > 0 );

    dirty_ = DIRTY_PRIMITIVES | DIRTY_TEXTURE | DIRTY_SELECTION;
}

void RenderVolumeObject::freeBuffers_()
{
    if ( !getViewerInstance().isGLInitialized() || !loadGL() )
        return;
    GL_EXEC( glDeleteVertexArrays( 1, &volumeArrayObjId_ ) );
}

void RenderVolumeObject::update_()
{
    dirty_ |= objVoxels_->getDirtyFlags();
    objVoxels_->resetDirty();
}

MR_REGISTER_RENDER_OBJECT_IMPL( ObjectVoxels, RenderObjectCombinator<RenderDefaultUiObject, RenderVolumeObject> )

}
#endif
