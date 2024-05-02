#include "MRMesh/MRPython.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRViewer/MRCommandLoop.h"

namespace MR
{

namespace
{

bool selectName( const std::string& modelName )
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( auto& obj : selected )
    {
        if ( modelName == obj->name() )
        {
            obj->select( true );
        }
        else
        {
            obj->select( false );
        }
    }
    return true;
}

template <typename T>
bool selectType()
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Any );
    for ( auto& obj : selected )
    {
        if ( auto objConverted = obj->asType<T>() )
        {
            obj->select( true );
        }
        else
        {
            obj->select( false );
        }
    }
    return true;
}

bool unselect()
{
    auto selected = getAllObjectsInTree<VisualObject>( &SceneRoot::get(), ObjectSelectivityType::Selected );
    for ( auto& obj : selected )
    {
        obj->select( false );
    }
    return true;
}

} // namespace

} // namespace MR

namespace
{

void pythonSelectName( const std::string modelName )
{
    MR::CommandLoop::runCommandFromGUIThread( [modelName] ()
    {
        MR::selectName( modelName );
    } );
}
void pythonUnselect()
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        MR::unselect();
    } );
}
void pythonSelectType( const std::string modelType )
{
    MR::CommandLoop::runCommandFromGUIThread( [modelType] ()
    {
        if ( modelType == "Meshes" )
        {
            MR::selectType<MR::ObjectMesh>();
            return;
        }
        if ( modelType == "Points" )
        {
            MR::selectType<MR::ObjectPoints>();
            return;
        }
        if ( modelType == "Voxels" )
        {
            MR::selectType<MR::ObjectVoxels>();
            return;
        }
        MR::unselect();
    } );
}

void pythonClearScene()
{
    MR::CommandLoop::runCommandFromGUIThread( [] ()
    {
        MR::SceneRoot::get().removeAllChildren();
    } );
}

void pythonAddMeshToScene( const MR::Mesh& mesh, const std::string& name )
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        std::shared_ptr<MR::ObjectMesh> objMesh = std::make_shared<MR::ObjectMesh>();
        objMesh->setMesh( std::make_shared<MR::Mesh>( mesh ) );
        objMesh->setName( name );
        MR::SceneRoot::get().addChild( objMesh );
    } );
}

void pythonAddPointCloudToScene( const MR::PointCloud& points, const std::string& name )
{
    MR::CommandLoop::runCommandFromGUIThread( [&] ()
    {
        std::shared_ptr<MR::ObjectPoints> objPoints = std::make_shared<MR::ObjectPoints>();
        objPoints->setPointCloud( std::make_shared<MR::PointCloud>( points ) );
        objPoints->setName( name );
        MR::SceneRoot::get().addChild( objPoints );
    } );
}

template <typename ObjectType, auto MemberPtr>
auto pythonGetSelectedModels()
{
    using ReturnedElemType = std::remove_cv_t<typename std::remove_cvref_t<decltype((std::declval<ObjectType>().*MemberPtr)())>::element_type>;
    using ReturnedVecType = std::vector<ReturnedElemType>;

    ReturnedVecType ret;

    auto objects = getAllObjectsInTree<ObjectType>( MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected );
    ret.reserve( objects.size() );

    for ( const auto& object : objects )
        ret.push_back( *( ( *object ).*MemberPtr)() );

    return ret;
}

} // namespace

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, Scene, [] ( pybind11::module_& m )
{
    m.def( "addMeshToScene", &pythonAddMeshToScene, pybind11::arg( "mesh" ), pybind11::arg( "name" ), "add given mesh to scene tree" );
    m.def( "addPointCloudToScene", &pythonAddPointCloudToScene, pybind11::arg( "points" ), pybind11::arg( "name" ), "add given point cloud to scene tree" );
    m.def( "clearScene", &pythonClearScene, "remove all objects from scene tree" );
    m.def( "selectByName", &pythonSelectName, pybind11::arg( "objectName" ), "select objects in scene tree with given name, unselect others" );
    m.def( "selectByType", &pythonSelectType, pybind11::arg( "typeName" ), "string typeName: {\"Meshes\", \"Points\", \"Voxels\"}\nobjects in scene tree with given type, unselect others" );
    m.def( "unselectAll", &pythonUnselect, "unselect all objects in scene tree" );

    m.def( "getSelectedMeshes", &pythonGetSelectedModels<MR::ObjectMeshHolder, &MR::ObjectMeshHolder::mesh>, "Get copies of all selected meshes in the scene." );
    // We don't have python bindings for those types yet.
    // m.def( "getSelectedPointClouds", &pythonGetSelectedModels<MR::ObjectPointsHolder, &MR::ObjectPointsHolder::pointCloud>, "Get copies of all selected point clouds in the scene." );
    // m.def( "getSelectedPolylines", &pythonGetSelectedModels<MR::ObjectLinesHolder, &MR::ObjectLinesHolder::polyline>, "Get copies of all selected polylines in the scene." );
} )
