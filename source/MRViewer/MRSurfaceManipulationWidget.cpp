#include "MRSurfaceManipulationWidget.h"
#include "MRMouseController.h"
#include "MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRViewerInstance.h"
#include "MRAppendHistory.h"
#include "MRMouse.h"
#include "MRViewer/MRGladGlfw.h"
#include "MRMesh/MREdgePaths.h"
#include "MRMesh/MRPositionVertsSmoothly.h"
#include "MRMesh/MRSurfaceDistance.h"
#include "MRMesh/MRExpandShrink.h"
#include "MRMesh/MREnumNeighbours.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRMeshRelax.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRFillHoleNicely.h"
#include "MRMesh/MRLaplacian.h"

namespace MR
{
//const float k = r < 1-a ? std::sqrt( sqr( 1 - a ) - sqr( r ) ) + ( 1 - a ) : -std::sqrt( sqr( a ) - sqr( r - 1 ) ) + a; // alternative version F_point_shift(r,i) (i == a)

// not in the header to be able to destroy Laplacian
SurfaceManipulationWidget::SurfaceManipulationWidget()
{
}

// not in the header to be able to destroy Laplacian
SurfaceManipulationWidget::~SurfaceManipulationWidget()
{
}

void SurfaceManipulationWidget::init( const std::shared_ptr<ObjectMesh>& objectMesh )
{
    obj_ = objectMesh;
    diagonal_ = obj_->getBoundingBox().diagonal();

    if ( firstInit_ )
    {
        settings_.radius = diagonal_ * 0.02f;
        settings_.relaxForce = 0.2f;
        settings_.editForce = diagonal_ * 0.01f;
        settings_.relaxForceAfterEdit = 0.25f;
        settings_.workMode = WorkMode::Add;
        firstInit_ = false;
    }


    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;
    singleEditingRegion_ = VertBitSet( numV, false );
    visualizationRegion_ = VertBitSet( numV, false );
    generalEditingRegion_ = VertBitSet( numV, false );
    pointsShift_ = VertScalars( numV, 0.f );
    editingDistanceMap_ = VertScalars( numV, 0.f );
    visualizationDistanceMap_ = VertScalars( numV, 0.f );

    obj_->setAncillaryTexture( { { { Color { 255, 64, 64, 255 }, Color { 0, 0, 0, 0 } }, Vector2i { 1, 2 } } } );
    obj_->setAncillaryUVCoords( VertUVCoords( numV, { 0, 1 } ) );

    initConnections_();
    mousePressed_ = false;
    mousePos_ = { -1, -1 };
}

void SurfaceManipulationWidget::reset()
{
    oldMesh_.reset();

    obj_->clearAncillaryTexture();
    obj_->setPickable( true );
    obj_.reset();

    singleEditingRegion_.clear();
    visualizationRegion_.clear();
    generalEditingRegion_.clear();
    pointsShift_.clear();
    editingDistanceMap_.clear();
    visualizationDistanceMap_.clear();

    resetConnections_();
    mousePressed_ = false;
}

void SurfaceManipulationWidget::setSettings( const Settings& settings )
{
    if ( mousePressed_ )
        return;

    settings_ = settings;
    settings_.radius = std::max( settings_.radius, 1.e-5f );
    settings_.relaxForce = std::clamp( settings_.relaxForce, 0.001f, 0.5f );
    settings_.editForce = std::max( settings_.editForce, 1.e-5f );
    settings_.relaxForceAfterEdit = std::clamp( settings_.relaxForceAfterEdit, 0.f, 0.5f );
    settings_.sharpness = std::clamp( settings_.sharpness, 0.f, 100.f );
    updateRegion_( mousePos_ );
}

bool SurfaceManipulationWidget::onMouseDown_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( button != MouseButton::Left )
        return false;

    auto [obj, pick] = getViewerInstance().viewport().pick_render_object();
    if ( !obj || obj != obj_ )
        return false;

    mousePressed_ = true;
    if ( settings_.workMode == WorkMode::Laplacian )
    {
        if ( !pick.face.valid() )
            return false;

        if ( badRegion_ )
        {
            mousePressed_ = false;
            return false;
        }
        laplacianPickVert_( pick );
    }
    else
    {
        if ( settings_.workMode != WorkMode::Patch )
        {
            // in patch mode the mesh does not change till mouse up, and we always need to pick in it (before and right after patch)
            oldMesh_ = std::dynamic_pointer_cast< ObjectMesh >( obj_->clone() );
            oldMesh_->setAncillary( true );
            obj_->setPickable( false );
        }
        appendHistoryAction_ = true;
        std::string name = "Brush: ";
        if ( settings_.workMode == WorkMode::Add )
            name += "Add";
        else if ( settings_.workMode == WorkMode::Remove )
            name += "Remove";
        else if ( settings_.workMode == WorkMode::Relax )
            name += "Smooth";
        else if ( settings_.workMode == WorkMode::Patch )
            name += "Patch";
        historyAction_ = std::make_shared<ChangeMeshAction>( name, obj_ );
        changeSurface_();
    }

    return true;
}

bool SurfaceManipulationWidget::onMouseUp_( Viewer::MouseButton button, int /*modifier*/ )
{
    if ( button != MouseButton::Left || !mousePressed_ )
        return false;

    mousePressed_ = false;
    if ( settings_.workMode == WorkMode::Laplacian )
        return true;

    size_t numV = obj_->mesh()->topology.lastValidVert() + 1;
    pointsShift_ = VertScalars( numV, 0.f );

    auto & mesh = *obj_->varMesh();
    if ( settings_.workMode == WorkMode::Patch )
    {
        ownMeshChangedSignal_ = true;
        auto faces = getIncidentFaces( mesh.topology, generalEditingRegion_ );
        if ( faces.any() )
        {
            auto bds = delRegionKeepBd( mesh, faces );
            for ( const auto & bd : bds )
            {
                if ( bd.empty() )
                    continue;
                // assert( isHoleBd( mesh.topology, bd ) ) can probably fail due to different construction of loops,
                // so we check every edge of every loop below
                const auto len = calcPathLength( bd, mesh );
                const auto avgLen = len / bd.size();
                FillHoleNicelySettings settings
                {
                    .triangulateParams =
                    {
                        .metric = getUniversalMetric( mesh ),
                        .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong
                    },
                    .maxEdgeLen = 2 * (float)avgLen,
                    .edgeWeights = settings_.edgeWeights
                };
                for ( auto e : bd )
                    if ( !mesh.topology.left( e ) )
                        fillHoleNicely( mesh, e, settings );
            }
            obj_->setDirtyFlags( DIRTY_ALL );

            // otherwise whole surface becomes red after patch and before mouse move
            updateRegion_( mousePos_ );
        }
    }
    else if ( ( settings_.workMode == WorkMode::Add || settings_.workMode == WorkMode::Remove ) &&
        settings_.relaxForceAfterEdit > 0.f && generalEditingRegion_.any() )
    {
        ownMeshChangedSignal_ = true;

        MeshRelaxParams params;
        params.region = &generalEditingRegion_;
        params.force = settings_.relaxForceAfterEdit;
        params.iterations = 5;
        relax( mesh, params );
        obj_->setDirtyFlags( DIRTY_PRIMITIVES );
    }
    generalEditingRegion_ = VertBitSet( numV, false );

    obj_->setPickable( true );

    oldMesh_.reset();

    return true;
}

bool SurfaceManipulationWidget::onMouseMove_( int mouse_x, int mouse_y )
{
    auto mousePos = Vector2f{ float( mouse_x ), float( mouse_y ) };
    if ( settings_.workMode == WorkMode::Laplacian )
    {
        if ( mousePressed_ )
        {
            if ( appendHistoryAction_ )
            {
                appendHistoryAction_ = false;
                AppendHistory( std::move( historyAction_ ) );
            }
            laplacianMoveVert_( mousePos );
        }
        else
            updateRegion_( mousePos );
    }
    else
    {
        updateRegion_( mousePos );
        if ( mousePressed_ )
            changeSurface_();
    }

    return true;
}

void SurfaceManipulationWidget::postDraw_()
{
    if ( !badRegion_ )
        return;

    auto drawList = ImGui::GetBackgroundDrawList();
    const auto& mousePos = Vector2f( getViewerInstance().mouseController().getMousePos() );
    drawList->AddCircleFilled( ImVec2( mousePos.x, mousePos.y ), 10.f, Color::gray().getUInt32() );
}

void SurfaceManipulationWidget::initConnections_()
{
    if ( connectionsInitialized_ )
        return;
    connectionsInitialized_ = true;
    meshChangedConnection_ = obj_->meshChangedSignal.connect( [&] ( uint32_t )
    {
        if ( ownMeshChangedSignal_ )
        {
            ownMeshChangedSignal_ = false;
            return;
        }
        abortEdit_();
        init( obj_ );
        updateRegion_( mousePos_ );
    } );
    connect( &getViewerInstance(), 10, boost::signals2::at_front );
}

void SurfaceManipulationWidget::resetConnections_()
{
    connectionsInitialized_ = false;
    meshChangedConnection_.disconnect();
    disconnect();
}

void SurfaceManipulationWidget::changeSurface_()
{
    if ( !singleEditingRegion_.any() || badRegion_ )
        return;

    if ( appendHistoryAction_ )
    {
        appendHistoryAction_ = false;
        AppendHistory( std::move( historyAction_ ) );
    }

    MR_TIMER;

    ownMeshChangedSignal_ = true;

    if ( settings_.workMode == WorkMode::Patch )
    {
        generalEditingRegion_ |= singleEditingRegion_;
        return; // everything is done on mouse up
    }

    if ( settings_.workMode == WorkMode::Relax )
    {
        MeshRelaxParams params;
        params.region = &singleEditingRegion_;
        params.force = settings_.relaxForce;
        relax( *obj_->varMesh(), params );
        obj_->setDirtyFlags( DIRTY_POSITION );
        return;
    }

    Vector3f normal;
    auto objMeshPtr = oldMesh_ ? oldMesh_ : obj_;
    const auto& mesh = *objMeshPtr->mesh();
    for ( auto v : singleEditingRegion_ )
        normal += mesh.normal( v );
    normal = normal.normalized();

    auto& points = obj_->varMesh()->points;

    const float maxShift = settings_.editForce;
    const float intensity = ( 100.f - settings_.sharpness ) / 100.f * 0.5f + 0.25f;
    const float a1 = -1.f * ( 1 - intensity ) / intensity / intensity;
    const float a2 = intensity / ( 1 - intensity ) / ( 1 - intensity );
    const float direction = settings_.workMode == WorkMode::Remove ? -1.f : 1.f;
    BitSetParallelFor( singleEditingRegion_, [&] ( VertId v )
    {
        const float r = std::clamp( editingDistanceMap_[v] / settings_.radius, 0.f, 1.f );
        const float k = r < intensity ? a1 * r * r + 1 : a2 * ( r - 1 ) * ( r - 1 ); // I(r)
        float pointShift = maxShift * k; // shift = F * I(r)
        if ( pointShift > pointsShift_[v] )
        {
            pointShift -= pointsShift_[v];
            pointsShift_[v] += pointShift;
        }
        else
            return;
        points[v] += direction * pointShift * normal;
    } );
    generalEditingRegion_ |= singleEditingRegion_;
    obj_->setDirtyFlags( DIRTY_PRIMITIVES );
}

void SurfaceManipulationWidget::updateUVmap_( bool set )
{
    VertUVCoords uvs;
    obj_->updateAncillaryUVCoords( uvs );
    uvs.resizeWithReserve( obj_->mesh()->points.size(), UVCoord{ 0, 1 } );
    const float normalize = 0.5f / settings_.radius;
    BitSetParallelFor( visualizationRegion_, [&] ( VertId v )
    {
        uvs[v] = set ? UVCoord{ 0, visualizationDistanceMap_[v] * normalize } : UVCoord{ 0, 1 };
    } );
    obj_->setAncillaryUVCoords( std::move( uvs ) );
}

void SurfaceManipulationWidget::updateRegion_( const Vector2f& mousePos )
{
    MR_TIMER;

    const auto& viewerRef = getViewerInstance();
    std::vector<Vector2f> viewportPoints;
    if ( !mousePressed_ || ( mousePos - mousePos_ ).lengthSq() < 16.f )
        viewportPoints.push_back( Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewerRef.getHoveredViewportId() ) ) );
    else
    {
        const Vector2f newMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos ), viewerRef.getHoveredViewportId() ) );
        const Vector2f oldMousePos = Vector2f( viewerRef.screenToViewport( Vector3f( mousePos_ ), viewerRef.getHoveredViewportId() ) );
        const Vector2f vec = newMousePos - oldMousePos;
        const int count = int( std::ceil( vec.length() ) ) + 1;
        const Vector2f step = vec / ( count - 1.f );
        viewportPoints.resize( count );
        for ( int i = 0; i < count; ++i )
            viewportPoints[i] = oldMousePos + step * float( i );
    }
    mousePos_ = mousePos;

    auto objMeshPtr = oldMesh_ ? oldMesh_ : obj_;
    // to pick some object, it must have a parent object
    std::shared_ptr<Object> parent;
    if ( oldMesh_ )
    {
        parent = std::make_shared<Object>();
        parent->addChild( oldMesh_ );
    }
    std::vector<ObjAndPick> movedPosPick = getViewerInstance().viewport().multiPickObjects( std::array{ static_cast<VisualObject*>( objMeshPtr.get() ) }, viewportPoints );
    if ( oldMesh_ )
    {
        oldMesh_->detachFromParent();
        parent.reset();
    }

    updateVizualizeSelection_( movedPosPick.empty() ? ObjAndPick() : movedPosPick.back() );
    const auto& mesh = *objMeshPtr->mesh();
    if ( !mousePressed_ )
    {
        editingDistanceMap_ = visualizationDistanceMap_;
        singleEditingRegion_ = visualizationRegion_;
    }
    else
    {
        std::vector<MeshTriPoint> triPoints;
        VertBitSet newVerts( singleEditingRegion_.size() );
        triPoints.reserve( movedPosPick.size() );
        for ( int i = 0; i < movedPosPick.size(); ++i )
        {
            if ( movedPosPick[i].first == objMeshPtr )
            {
                const auto& pick = movedPosPick[i].second;
                VertId v[3];
                mesh.topology.getTriVerts( pick.face, v );
                for ( int j = 0; j < 3; ++j )
                    newVerts.set( v[j] );
                triPoints.push_back( mesh.toTriPoint( pick.face, pick.point ) );
            }
        }

        if ( triPoints.size() == 1 )
        {
            PointOnFace pOnFace{ mesh.topology.left( triPoints[0].e ), mesh.triPoint( triPoints[0] ) };
            editingDistanceMap_ = computeSpaceDistances( mesh, pOnFace, settings_.radius );
            singleEditingRegion_ = findNeighborVerts( mesh, pOnFace, settings_.radius );
        }
        else
        {
            // bad logic. need rework to SpaceDistance
            singleEditingRegion_ = newVerts;
            dilateRegion( mesh, singleEditingRegion_, settings_.radius * 1.5f );
            editingDistanceMap_ = computeSurfaceDistances( mesh, triPoints, settings_.radius * 1.5f, &singleEditingRegion_ );
        }
    }
    for ( auto v : singleEditingRegion_ )
        singleEditingRegion_.set( v, editingDistanceMap_[v] <= settings_.radius );
}

void SurfaceManipulationWidget::abortEdit_()
{
    if ( !mousePressed_ )
        return;
    mousePressed_ = false;
    oldMesh_.reset();
    obj_->setPickable( true );
    obj_->clearAncillaryTexture();
    appendHistoryAction_ = false;
    historyAction_.reset();
}

void SurfaceManipulationWidget::laplacianPickVert_( const PointOnFace& pick )
{
    appendHistoryAction_ = true;
    storedDown_ = getViewerInstance().mouseController().getMousePos();
    const auto& mesh = *obj_->mesh();
    touchVertId_ = mesh.getClosestVertex( pick );
    touchVertIniPos_ = mesh.points[touchVertId_];
    laplacian_ = std::make_unique<Laplacian>( *obj_->varMesh() );
    laplacian_->init( singleEditingRegion_, settings_.edgeWeights );
    historyAction_ = std::make_shared<ChangeMeshAction>( "Brush: Deform", obj_ );
}

void SurfaceManipulationWidget::laplacianMoveVert_( const Vector2f& mousePos )
{
    ownMeshChangedSignal_ = true;
    auto& viewerRef = getViewerInstance();
    const float zpos = viewerRef.viewport().projectToViewportSpace( obj_->worldXf()( touchVertIniPos_ ) ).z;
    auto viewportPoint1 = viewerRef.screenToViewport( Vector3f( mousePos.x, mousePos.y, zpos ), viewerRef.viewport().id );
    auto pos1 = viewerRef.viewport().unprojectFromViewportSpace( viewportPoint1 );
    auto viewportPoint0 = viewerRef.screenToViewport( Vector3f( float( storedDown_.x ), float( storedDown_.y ), zpos ), viewerRef.viewport().id );
    auto pos0 = viewerRef.viewport().unprojectFromViewportSpace( viewportPoint0 );
    const Vector3f move = obj_->worldXf().A.inverse()* ( pos1 - pos0 );
    laplacian_->fixVertex( touchVertId_, touchVertIniPos_ + move );
    laplacian_->apply();
    obj_->setDirtyFlags( DIRTY_POSITION );
}

void SurfaceManipulationWidget::updateVizualizeSelection_( const ObjAndPick& objAndPick )
{
    updateUVmap_( false );
    auto objMeshPtr = oldMesh_ ? oldMesh_ : obj_;
    const auto& mesh = *objMeshPtr->mesh();
    visualizationRegion_.reset();
    badRegion_ = false;
    if ( objAndPick.first == objMeshPtr )
    {
        PointOnFace pOnFace = objAndPick.second;
        if ( settings_.workMode == WorkMode::Laplacian )
        {
            const VertId vert = mesh.getClosestVertex( pOnFace );
            pOnFace = PointOnFace{ objAndPick.second.face, mesh.points[vert] };
        }
        visualizationDistanceMap_ = computeSpaceDistances( mesh, pOnFace, settings_.radius );
        visualizationRegion_ = findNeighborVerts( mesh, pOnFace, settings_.radius );
        expand( mesh.topology, visualizationRegion_ );
        {
            int pointsCount = 0;
            for ( auto vId : visualizationRegion_ )
            {
                if ( visualizationDistanceMap_[vId] <= settings_.radius )
                    ++pointsCount;
                if ( pointsCount == 3 )
                    break;
            }
            badRegion_ = pointsCount < 3;
        }
        if ( !badRegion_ )
            updateUVmap_( true );
    }
}

}
