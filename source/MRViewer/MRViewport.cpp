#include "MRViewport.h"
#include "MRViewer.h"
#include "ImGuiMenu.h"
#include "MRGLMacro.h"
#include "MRGLStaticHolder.h"
#include "MRMouseController.h"
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRArrow.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRToFromEigen.h>
#include <MRMesh/MRClosestPointInTriangle.h>
#include <MRMesh/MRTimer.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPolylineProject.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRPch/MRSuppressWarning.h"
#include "MRPch/MRTBB.h"

using VisualObjectTreeDataVector = std::vector<MR::VisualObject*>;
namespace
{
void getPickerDataVector( MR::Object& obj, MR::ViewportMask id, VisualObjectTreeDataVector& outVector )
{
    if ( !obj.isVisible( id ) )
        return;
    if ( auto visobj = obj.asType<MR::VisualObject>() )
        if ( visobj->isPickable( id ) )
            outVector.push_back( {visobj} );
    for ( const auto& child : obj.children() )
        getPickerDataVector( *child, id, outVector );
}
}

namespace MR
{

Viewport::Viewport()
{
    cameraLookAlong( Vector3f( -1.f, -1.f, -1.f ), Vector3f( -1, -1, 2 ) );
}

Viewport::~Viewport()
{
}

void Viewport::init()
{
    viewportGL_ = ViewportGL();
    initBaseAxes();
    updateSceneBox_();
    setRotationPivot_( sceneBox_.valid() ? sceneBox_.center() : Vector3f() );
    setupProjMatrix_();
    setupStaticProjMatrix_();
}

void Viewport::shut()
{
    viewportGL_.free();
}

// ================================================================
// draw functions part


void Viewport::draw(const VisualObject& obj, const AffineXf3f& xf, 
     DepthFuncion depthFunc, bool alphaSort ) const
{
    draw( obj, xf, projM_, depthFunc, alphaSort );
}

void Viewport::draw( const VisualObject& obj, const AffineXf3f& xf, const Matrix4f& projM,
     DepthFuncion depthFunc, bool alphaSort ) const
{
    auto modelTemp = Matrix4f( xf );
    auto normTemp = viewM_ * modelTemp;
    if ( normTemp.det() == 0 )
    {
        auto norm = normTemp.norm();
        if ( std::isnormal( norm ) )
        {
            normTemp /= norm;
            normTemp.w = { 0, 0, 0, 1 };
        }
        else
        {
            spdlog::warn( "Object transform is degenerate" );
            return;
        }
    }
    auto normM = normTemp.inverse().transposed();

    ModelRenderParams params
    {
        { viewM_, projM, id, toVec4<int>( viewportRect_ ) },
        modelTemp, &normM,
        params_.clippingPlane, depthFunc,
        params_.lightPosition, alphaSort
    };
    obj.render( params );
}

void Viewport::clearFramebuffers()
{
    if ( !viewportGL_.checkInit() )
        viewportGL_.init();
    viewportGL_.fillViewport( viewportRect_, params_.backgroundColor );
}

ObjAndPick Viewport::pick_render_object( uint16_t pickRadius ) const
{
    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    return pick_render_object( renderVector, pickRadius );
}

ObjAndPick Viewport::pick_render_object() const
{
    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    return pick_render_object( renderVector, getViewerInstance().glPickRadius );
}

ObjAndPick Viewport::pick_render_object( const Vector2f& viewportPoint ) const
{
    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );
    return pick_render_object( renderVector, viewportPoint );
}

ObjAndPick Viewport::pick_render_object( const std::vector<VisualObject*>& renderVector ) const
{
    return pick_render_object( renderVector, getViewerInstance().glPickRadius );
}

ObjAndPick Viewport::pick_render_object( const std::vector<VisualObject*>& renderVector, uint16_t pickRadius ) const
{
    auto& viewer = getViewerInstance();
    const auto& mousePos = viewer.mouseController().getMousePos();
    auto vp = viewer.screenToViewport(
        Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f ), id );
    if ( pickRadius == 0 )
        return pick_render_object( renderVector, Vector2f( vp.x, vp.y ) );
    else
    {
        std::vector<Vector2f> pixels;
        pixels.reserve( sqr( 2 * pickRadius + 1 ) );
        pixels.push_back( Vector2f( vp.x, vp.y ) );
        for ( int i = -int( pickRadius ); i <= int( pickRadius ); i++ )
        for ( int j = -int( pickRadius ); j <= int( pickRadius ); j++ )
        {
            if ( i == 0 && j == 0 )
                continue;
            if ( i * i + j * j <= pickRadius * pickRadius + 1 )
                pixels.push_back( Vector2f( vp.x + i, vp.y + j ) );
        }
        auto res = multiPickObjects( renderVector, pixels );
        if ( res.empty() )
            return {};
        if ( bool( res.front().first ) )
            return res.front();
        int minIndex = int( res.size() );
        float minZ = FLT_MAX;
        for ( int i = 0; i < res.size(); ++i )
        {
            const auto& [obj, pick] = res[i];
            if ( !obj )
                continue;
            if ( pick.zBuffer < minZ )
            {
                minZ = pick.zBuffer;
                minIndex = i;
            }
        }
        if ( minIndex < res.size() )
            return res[minIndex];
        return {};
    }
}

ObjAndPick Viewport::pick_render_object( const std::vector<VisualObject*>& renderVector, const Vector2f& viewportPoint ) const
{
    return multiPickObjects( renderVector, {viewportPoint} ).front();
}

std::vector<ObjAndPick> Viewport::multiPickObjects( const std::vector<VisualObject*>& renderVector, const std::vector<Vector2f>& viewportPoints ) const
{
    MR_TIMER;
    if ( viewportPoints.empty() )
        return {};
    std::vector<Vector2i> picks( viewportPoints.size() );
    ViewportGL::PickParameters params{ renderVector, getBaseRenderParams(), params_.clippingPlane };

    for ( int i = 0; i < viewportPoints.size(); ++i )
        picks[i] = Vector2i( viewportPoints[i] );

    std::vector<ObjAndPick> result( picks.size() );

    if ( width( viewportRect_ ) == 0 || height( viewportRect_ ) == 0 )
        return result;

    bool needBindSceneTexture = getViewerInstance().isSceneTextureBound();
    auto pickResult = viewportGL_.pickObjects( params, picks );
    if ( needBindSceneTexture )
        getViewerInstance().bindSceneTexture( true );

    for ( int i = 0; i < pickResult.size(); ++i )
    {
        auto& pickRes = pickResult[i];
        if ( pickRes.geomId == -1 || pickRes.primId == -1 )
            continue;

        PointOnObject res;
        res.primId = int( pickRes.primId );
        res.zBuffer = pickRes.zBuffer;
#ifndef __EMSCRIPTEN__
        auto voxObj = renderVector[pickRes.geomId]->asType<ObjectVoxels>();
        if ( voxObj && voxObj->isVolumeRenderingEnabled() )
        {
            res.point = renderVector[pickRes.geomId]->worldXf( id ).inverse()( 
                unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
            // TODO: support VoxelId
        }
        else
#endif
        if ( auto pointObj = renderVector[pickRes.geomId]->asType<ObjectPointsHolder>() )
        {
            if ( auto pc = pointObj->pointCloud() )
            {
                auto vid = VertId( int( pickRes.primId ) );
                if ( !pc->validPoints.test( vid ) )
                    continue;
                res.point = pc->points[vid];
            }
            else
            {
                res.point = renderVector[pickRes.geomId]->worldXf( id ).inverse()( unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
            }
        }
        else if ( auto linesObj = renderVector[pickRes.geomId]->asType<ObjectLinesHolder>() )
        {
            res.point = renderVector[pickRes.geomId]->worldXf( id ).inverse()( unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
            UndirectedEdgeId ue{ int( pickRes.primId ) };
            if ( auto pl = linesObj->polyline() )
                res.point = closestPointOnLineSegm( res.point, pl->edgeSegment( ue ) );
        }
        else if ( auto meshObj = renderVector[pickRes.geomId]->asType<ObjectMeshHolder>() )
        {
            if ( res.face.valid() )
            {
                const auto& mesh = meshObj->mesh();
                if ( mesh && !mesh->topology.hasFace( res.face ) )
                {
                    assert( false );
                    continue;
                }

                res.point = renderVector[pickRes.geomId]->worldXf( id ).inverse()( unprojectFromViewportSpace( Vector3f( viewportPoints[i].x, viewportPoints[i].y, pickRes.zBuffer ) ) );
                if ( mesh )
                {
                    Vector3f a, b, c;
                    mesh->getTriPoints( res.face, a, b, c );
                    res.point = closestPointInTriangle( res.point, a, b, c ).first;
                }
            }
        }
        result[i] = { std::dynamic_pointer_cast<VisualObject>( renderVector[pickRes.geomId]->getSharedPtr() ),res };
    }
    return result;
}

std::vector<std::shared_ptr<MR::VisualObject>> Viewport::findObjectsInRect( const Box2i& rect,
                                                                            int maxRenderResolutionSide ) const
{
    MR_TIMER;

    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    ViewportGL::PickParameters params{ renderVector, getBaseRenderParams(), params_.clippingPlane };

    auto viewportRect = Box2i( Vector2i( 0, 0 ), Vector2i( int( width( viewportRect_ ) ), int( height( viewportRect_ ) ) ) );
    auto pickResult = viewportGL_.findUniqueObjectsInRect( params, rect.intersection( viewportRect ), maxRenderResolutionSide );
    std::vector<std::shared_ptr<VisualObject>> result( pickResult.size() );
    for ( int i = 0; i < pickResult.size(); ++i )
    {
        result[i] = std::dynamic_pointer_cast<VisualObject>( renderVector[pickResult[i]]->getSharedPtr() );
    }

    return result;
}

std::unordered_map<std::shared_ptr<MR::ObjectMesh>, MR::FaceBitSet> Viewport::findVisibleFaces( const BitSet& includePixBs, 
    int maxRenderResolutionSide /*= 512 */ ) const
{
    MR_TIMER;

    VisualObjectTreeDataVector renderVector;
    getPickerDataVector( SceneRoot::get(), id, renderVector );

    ViewportGL::PickParameters params{ renderVector, getBaseRenderParams(), params_.clippingPlane };

    int width = int( MR::width( viewportRect_ ) );
    int height = int( MR::height( viewportRect_ ) );
    tbb::enumerable_thread_specific<Box2i> tlBoxes;
    BitSetParallelFor( includePixBs, [&] ( size_t i )
    {
        auto& localBox = tlBoxes.local();
        localBox.include( Vector2i( int( i ) % width, int( i ) / width ) );
    } );
    Box2i rect;
    for ( const auto& box : tlBoxes )
        rect.include( box );


    bool needBindSceneTexture = getViewerInstance().isSceneTextureBound();
    auto viewportRect = Box2i( Vector2i( 0, 0 ), Vector2i( width, height ) );
    auto realRect = rect.intersection( viewportRect );
    auto [pickResult, updatedBox] = viewportGL_.pickObjectsInRect( params, realRect, maxRenderResolutionSide );
    if ( needBindSceneTexture )
        getViewerInstance().bindSceneTexture( true );

    std::unordered_map<std::shared_ptr<MR::ObjectMesh>, MR::FaceBitSet> resMap;

    for ( int i = 0; i < pickResult.size(); ++i )
    {
        Vector2f downscaledPosRatio;
        downscaledPosRatio.x = std::clamp( float( i % ( MR::width( updatedBox ) + 1 ) ) / float( MR::width( updatedBox ) + 1 ), 0.0f, 1.0f );
        downscaledPosRatio.y = std::clamp( 1.0f - float( i / ( MR::width( updatedBox ) + 1 ) ) / float( MR::height( updatedBox ) + 1 ), 0.0f, 1.0f );

        Vector2i coord = realRect.min + Vector2i( mult( downscaledPosRatio, Vector2f( realRect.size() ) ) );
        assert( coord.x < width );
        assert( coord.y < height );
        
        int realId = coord.x + coord.y * width;
        if ( !includePixBs.test( realId ) )
            continue;

        auto gId = pickResult[i].geomId;
        if ( gId == unsigned( -1 ) )
            continue;

        auto pId = pickResult[i].primId;
        if ( pId == unsigned( -1 ) )
            continue;

        std::shared_ptr<ObjectMesh> meshObj;
        if ( auto parent = renderVector[gId]->parent() )
        {
            for ( auto& child : parent->children() )
            {
                if ( child.get() == renderVector[gId] )
                {
                    meshObj = std::dynamic_pointer_cast< ObjectMesh >( child );
                    break;
                }
            }
        }
        if ( !meshObj )
            continue;

        auto& fbs = resMap[meshObj];
        if ( fbs.empty() )
            fbs.resize( meshObj->mesh()->topology.lastValidFace() + 1 );
        fbs.set( FaceId( int( pId ) ) );
    }
    return resMap;
}

ConstObjAndPick Viewport::const_pick_render_object() const
{
    return pick_render_object();
}

ConstObjAndPick Viewport::const_pick_render_object( const std::vector<const VisualObject*>& objects ) const
{
    // not to duplicate code
    return pick_render_object( reinterpret_cast<const std::vector<VisualObject*>&> ( objects ) );
}

std::vector<ConstObjAndPick> Viewport::constMultiPickObjects( const std::vector<const VisualObject*>& objects, const std::vector<Vector2f>& viewportPoints ) const
{
    auto pickRes = multiPickObjects( reinterpret_cast<const std::vector<VisualObject*>&> ( objects ), viewportPoints );
    std::vector<ConstObjAndPick> res( pickRes.size() );
    for ( int i = 0; i < res.size(); ++i )
        res[i] = pickRes[i];
    return res;
}

void Viewport::setupView()
{
    setupViewMatrix_();
    setupProjMatrix_();
    setupStaticProjMatrix_();
}

void Viewport::preDraw()
{
    if ( !viewportGL_.checkInit() )
        viewportGL_.init();
    draw_rotation_center();
    draw_global_basis();
}

void Viewport::postDraw() const
{
    draw_border();
    draw_clipping_plane();

    // important to be last
    drawAxes();
}

void Viewport::updateSceneBox_()
{
    sceneBox_ = SceneRoot::get().getWorldTreeBox( id );
}

void Viewport::setViewportRect( const ViewportRectangle& rect )
{
    if ( rect == viewportRect_ )
        return;
    needRedraw_ = true;
    viewportRect_ = rect;
    initBaseAxes();
}

const ViewportRectangle& Viewport::getViewportRect() const
{
    return viewportRect_;
}

// ================================================================
// projection part

const Box3f& Viewport::getSceneBox() const
{
    return sceneBox_;
}

void Viewport::setBackgroundColor( const Color& color )
{
    if ( params_.backgroundColor == color )
        return;
    params_.backgroundColor = color; 
    needRedraw_ = true;
}

void Viewport::setClippingPlane( const Plane3f& plane )
{
    if ( params_.clippingPlane == plane )
        return;
    params_.clippingPlane = plane; 
    needRedraw_ = true;
}

void Viewport::setLabel( std::string s )
{
    params_.label = std::move( s );
    needRedraw_ = true;
}

void Viewport::showAxes( bool on )
{
    Viewer::constInstance()->basisAxes->setVisible( on, id );
    needRedraw_ |= Viewer::constInstance()->basisAxes->getRedrawFlag( id );
    Viewer::constInstance()->basisAxes->resetRedrawFlag();
}

void Viewport::showClippingPlane( bool on )
{
    Viewer::constInstance()->clippingPlaneObject->setVisible( on, id );
    needRedraw_ |= Viewer::constInstance()->clippingPlaneObject->getRedrawFlag( id );
    Viewer::constInstance()->clippingPlaneObject->resetRedrawFlag();
}

void Viewport::showRotationCenter( bool on )
{
    Viewer::constInstance()->rotationSphere->setVisible( on, id );
}

void Viewport::rotationCenterMode( Parameters::RotationCenterMode mode )
{
    if ( mode == params_.rotationMode )
        return;
    params_.rotationMode = mode;
    needRedraw_ = true;
}

void Viewport::showGlobalBasis( bool on )
{
    Viewer::constInstance()->globalBasisAxes->setVisible( on, id );
    needRedraw_ |= Viewer::constInstance()->globalBasisAxes->getRedrawFlag( id );
    Viewer::constInstance()->globalBasisAxes->resetRedrawFlag();
}

void Viewport::setParameters( const Viewport::Parameters& params )
{
    if ( params == params_ )
        return;
    params_ = params;
    needRedraw_ = true;
}

void Viewport::setAxesSize( const int axisPixSize )
{
    if ( axisPixSize == axisPixSize_ )
        return;
    needRedraw_ = true;
    axisPixSize_ = axisPixSize;
    initBaseAxes();
}

void Viewport::setAxesPos( const int pixelXoffset, const int pixelYoffset )
{
    if ( pixelXoffset_ == pixelXoffset &&
         pixelYoffset_ == pixelYoffset )
        return;
    needRedraw_ = true;
    pixelXoffset_ = pixelXoffset;
    pixelYoffset_ = pixelYoffset;
    initBaseAxes();
}

// ================================================================
// GL functions part

void Viewport::draw_border() const
{
    viewportGL_.drawBorder( viewportRect_, params_.borderColor );
}

// ================================================================
// additional elements

void Viewport::initBaseAxes()
{
    // find relative points for axes
    auto scaling = 1.0f;
    if ( auto menu = getViewerInstance().getMenuPlugin() )
        scaling = menu->menu_scaling();
    float axesX, axesY;
    if(pixelXoffset_ < 0)
        axesX = width( viewportRect_ ) + pixelXoffset_ * scaling;
    else
        axesX = float( pixelXoffset_ * scaling );
    if(pixelYoffset_ < 0)
        axesY = height( viewportRect_ ) + pixelYoffset_ * scaling;
    else
        axesY = float( pixelYoffset_ * scaling );
    const float pixSize = float( axisPixSize_ * scaling ) / sqrtf( 2 );
    relPoseBase = { axesX, axesY, 0.5f };
    relPoseSide = { axesX + pixSize, axesY + pixSize, 0.5f };
}

void Viewport::drawAxes() const
{
    if ( Viewer::constInstance()->basisAxes->isVisible( id ) )
    {
        // compute inverse in double precision to avoid NaN for very small scales
        auto fullInversedM = Matrix4f( ( Matrix4d( staticProj_ ) * Matrix4d( viewM_ ) ).inverse() );
        auto transBase = fullInversedM( viewportSpaceToClipSpace( relPoseBase ) );
        auto transSide = fullInversedM( viewportSpaceToClipSpace( relPoseSide ) );

        float scale = (transSide - transBase).length();
        const auto basisAxesXf = AffineXf3f( Matrix3f::scale( scale ), transBase );
        draw( *Viewer::constInstance()->basisAxes, basisAxesXf, staticProj_, DepthFuncion::Always );
        draw( *Viewer::constInstance()->basisAxes, basisAxesXf, staticProj_ );
        for ( const auto& child : getViewerInstance().basisAxes->children() )
        {
            if ( auto visualChild = child->asType<VisualObject>() )
                draw( *visualChild, basisAxesXf, staticProj_ );
        }
    }
}

void Viewport::draw_clipping_plane() const
{
    const auto& v = Viewer::constInstance();
    if ( !v->clippingPlaneObject->isVisible( id ) )
        return;

    AffineXf3f transform = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), params_.clippingPlane.n ) );
    transform = AffineXf3f::linear( Matrix3f::scale( params_.cameraDfar - params_.cameraDnear )/ params_.cameraZoom ) * transform;
    transform.b = params_.clippingPlane.n * params_.clippingPlane.d;
    draw( *Viewer::constInstance()->clippingPlaneObject, transform );
}

void Viewport::draw_global_basis() const
{
    if ( !Viewer::instance()->globalBasisAxes->isVisible( id ) )
        return;

    draw( *Viewer::constInstance()->globalBasisAxes, params_.globalBasisAxesXf() );
}

bool Viewport::Parameters::operator==( const Viewport::Parameters& other ) const
{
    return
        backgroundColor == other.backgroundColor &&
        lightPosition == other.lightPosition &&
        cameraTrackballAngle == other.cameraTrackballAngle &&
        cameraTranslation == other.cameraTranslation &&
        cameraZoom == other.cameraZoom &&
        cameraViewAngle == other.cameraViewAngle &&
        cameraDnear == other.cameraDnear &&
        cameraDfar == other.cameraDfar &&
        depthTest == other.depthTest &&
        orthographic == other.orthographic &&
        objectScale == objectScale &&
        borderColor == other.borderColor &&
        label == other.label &&
        clippingPlane == other.clippingPlane &&
        rotationMode == other.rotationMode &&
        selectable == other.selectable;
}

}
