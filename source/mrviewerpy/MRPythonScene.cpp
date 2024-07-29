#include "MRMesh/MRObjectLinesHolder.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRPython.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRVDBFloatGrid.h"
#include "MRMesh/MRMeta.h"
#include "MRViewer/MRCommandLoop.h"
#include "MRPch/MRFmt.h"

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
    using ReturnedElemType = std::remove_cvref_t<decltype((std::declval<ObjectType>().*MemberPtr)())>;
    using ReturnedVecType = std::vector<std::remove_cvref_t<typename MR::Meta::SharedPtrTraits<ReturnedElemType>::elemType>>;

    ReturnedVecType ret;

    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto objects = MR::getAllObjectsInTree<ObjectType>( MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected );
        ret.reserve( objects.size() );

        for ( const auto& object : objects )
        {
            if constexpr ( MR::Meta::SharedPtrTraits<ReturnedElemType>::isSharedPtr )
                ret.push_back( *( ( *object ).*MemberPtr)() );
            else
                ret.push_back( ( ( *object ).*MemberPtr)() );
        }
    } );

    return ret;
}

void pythonModifySelectedMesh( MR::Mesh mesh )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto selected = MR::getAllObjectsInTree<MR::ObjectMesh>( &MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected );
        if ( selected.size() != 1 )
            throw std::runtime_error( fmt::format( "Exactly one mesh must be selected, but have {}.", selected.size() ) );
        if ( !selected[0] )
            throw std::runtime_error( "Internal error (the object is null?)." );
        selected[0]->setMesh( std::make_shared<MR::Mesh>( std::move( mesh ) ) );
        selected[0]->setDirtyFlags( MR::DIRTY_ALL );
    } );
}

template <typename T, auto M>
auto pythonGetSelectedBitset()
{
    std::vector<std::remove_cvref_t<decltype( ( std::declval<T>().*M )() )>> ret;

    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto selected = MR::getAllObjectsInTree<T>( &MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected );
        ret.resize( selected.size() );
        for ( std::size_t i = 0; i < ret.size(); i++ )
            ret[i] = ( ( *selected[i] ).*M )();
    } );
    return ret;
}

template <typename T, typename U, auto M>
void pythonSetSelectedBitset( const std::vector<U>& bitsets )
{
    MR::CommandLoop::runCommandFromGUIThread( [&]
    {
        auto selected = MR::getAllObjectsInTree<T>( &MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected );
        if ( selected.size() != bitsets.size() )
            throw std::runtime_error( fmt::format( "Specified {} bitsets, but {} objects are selected.", bitsets.size(), selected.size() ) );
        for ( std::size_t i = 0; i < selected.size(); i++ )
            ( ( *selected[i] ).*M )( bitsets[i] );
    } );
}

} // namespace

MR_ADD_PYTHON_CUSTOM_DEF( mrviewerpy, Scene, [] ( pybind11::module_& m )
{
    m.def( "addMeshToScene", &pythonAddMeshToScene, pybind11::arg( "mesh" ), pybind11::arg( "name" ), "add given mesh to scene tree" );

    m.def( "modifySelectedMesh", &pythonModifySelectedMesh, pybind11::arg( "mesh" ), "Assign a new mesh to the selected mesh object. Exactly one object must be selected." );

    m.def( "getSelectedMeshFaces", &pythonGetSelectedBitset<MR::ObjectMeshHolder, &MR::ObjectMeshHolder::getSelectedFaces>, "Get selected face bitsets of the selected mesh objects." );
    m.def( "getSelectedMeshEdges", &pythonGetSelectedBitset<MR::ObjectMeshHolder, &MR::ObjectMeshHolder::getSelectedEdges>, "Get selected edge bitsets of the selected mesh objects." );
    m.def( "getSelectedPointCloudPoints", &pythonGetSelectedBitset<MR::ObjectPointsHolder, &MR::ObjectPointsHolder::getSelectedPoints>, "Get selected point bitsets of the selected point cloud objects." );
    m.def( "setSelectedMeshFaces", &pythonSetSelectedBitset<MR::ObjectMeshHolder, MR::FaceBitSet, &MR::ObjectMeshHolder::selectFaces>, "Set selected face bitsets of the selected mesh objects." );
    m.def( "setSelectedMeshEdges", &pythonSetSelectedBitset<MR::ObjectMeshHolder, MR::UndirectedEdgeBitSet, &MR::ObjectMeshHolder::selectEdges>, "Set selected edge bitsets of the selected mesh objects." );
    m.def( "setSelectedPointCloudPoints", &pythonSetSelectedBitset<MR::ObjectPointsHolder, MR::VertBitSet, &MR::ObjectPointsHolder::selectPoints>, "Set selected point bitsets of the selected point cloud objects." );
    // Polylines don't have selection bitsets at the moment.

    m.def( "addPointCloudToScene", &pythonAddPointCloudToScene, pybind11::arg( "points" ), pybind11::arg( "name" ), "add given point cloud to scene tree" );
    m.def( "clearScene", &pythonClearScene, "remove all objects from scene tree" );
    m.def( "selectByName", &pythonSelectName, pybind11::arg( "objectName" ), "select objects in scene tree with given name, unselect others" );
    m.def( "selectByType", &pythonSelectType, pybind11::arg( "typeName" ), "string typeName: {\"Meshes\", \"Points\", \"Voxels\"}\nobjects in scene tree with given type, unselect others" );
    m.def( "unselectAll", &pythonUnselect, "unselect all objects in scene tree" );

    m.def( "getSelectedObjects", []{ return MR::getAllObjectsInTree( &MR::SceneRoot::get(), MR::ObjectSelectivityType::Selected ); } );
    m.def( "getSelectedMeshes", &pythonGetSelectedModels<MR::ObjectMeshHolder, &MR::ObjectMeshHolder::mesh>, "Get copies of all selected meshes in the scene." );
    m.def( "getSelectedPointClouds", &pythonGetSelectedModels<MR::ObjectPointsHolder, &MR::ObjectPointsHolder::pointCloud>, "Get copies of all selected point clouds in the scene." );
    m.def( "getSelectedPolylines", &pythonGetSelectedModels<MR::ObjectLinesHolder, &MR::ObjectLinesHolder::polyline>, "Get copies of all selected polylines in the scene." );
    m.def( "getSelectedVoxels", &pythonGetSelectedModels<MR::ObjectVoxels, &MR::ObjectVoxels::vdbVolume>, "Get copies of all selected voxel grids in the scene." );
    m.def( "getSelectedDistanceMaps", &pythonGetSelectedModels<MR::ObjectDistanceMap, &MR::ObjectDistanceMap::getDistanceMap>, "Get copies of all selected voxel grids in the scene." );
} )
