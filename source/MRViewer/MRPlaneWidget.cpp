#include "MRPlaneWidget.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRMakePlane.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectLines.h"

namespace MR
{

PlaneWidget::PlaneWidget( const Plane3f& plane, const Box3f& box, std::function<void( const Plane3f& )> onPlaneUpdate )
: plane_(plane),
box_(box),
onPlaneUpdate_( onPlaneUpdate )
{ 
    std::shared_ptr<Mesh> planeMesh = std::make_shared<Mesh>( makePlane() );
    planeObj_ = std::make_shared<ObjectMesh>();
    planeObj_->setName( "PlaneObject" );
    planeObj_->setMesh( planeMesh );
    planeObj_->setAncillary( true );
    planeObj_->setFrontColor( Color( Vector4f::diagonal( 0.3f ) ), false );
    planeObj_->setBackColor( Color( Vector4f::diagonal( 0.3f ) ) );
    SceneRoot::get().addChild( planeObj_ );

    updateWidget();
}

PlaneWidget::~PlaneWidget()
{    
    planeObj_->detachFromParent();
}

void PlaneWidget::updatePlane( const Plane3f& plane, bool updateCameraRotation )
{
    plane_ = plane;
    updateWidget( updateCameraRotation );
    if ( onPlaneUpdate_ )
        onPlaneUpdate_( plane );
}

void PlaneWidget::updateBox( const Box3f& box, bool updateCameraRotation )
{
    box_ = box;
    updateWidget( updateCameraRotation );
}

void PlaneWidget::updateWidget( bool updateCameraRotation )
{
    auto viewer = Viewer::instance();
    plane_ = plane_.normalized();

    auto trans1 = AffineXf3f::translation( plane_.project( box_.center() ) );
    auto rot1 = AffineXf3f::linear( Matrix3f::rotation( Vector3f::plusZ(), plane_.n ) );
    auto scale1 = AffineXf3f::linear( Matrix3f::scale( box_.diagonal() ) );
    AffineXf3f transform = trans1 * rot1 * scale1;
    if ( updateCameraRotation )
        cameraUp3Old_ = viewer->viewport().getUpDirection();
    Vector3f cameraUp3 = cameraUp3Old_;
    auto rot2 = Matrix3f::rotation( transform.A * Vector3f::plusY(),
                                    plane_.project( transform( Vector3f() ) + cameraUp3 ) - transform( Vector3f() ) );

    auto lastPlaneTransform = trans1 * AffineXf3f::linear( rot2 ) * rot1;
    transform = lastPlaneTransform * scale1;
    planeObj_->setXf( transform );
}

}