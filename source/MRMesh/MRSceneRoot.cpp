#include "MRSceneRoot.h"
#include "MRObjectFactory.h"
#include "MRPch/MRJson.h"

namespace MR
{

#ifndef MR_SCENEROOT_CONST

SceneRootObject& SceneRoot::get()
{
    return *instace_().root_;
}

std::shared_ptr<SceneRootObject>& SceneRoot::getSharedPtr()
{
    return instace_().root_;
}

void SceneRoot::setScenePath( const std::filesystem::path& scenePath )
{
    instace_().scenePath_ = scenePath;
}

#endif

SceneRoot& SceneRoot::instace_()
{
    static SceneRoot scene;
    return scene;
}

SceneRoot::SceneRoot()
{
    root_ = std::make_shared<SceneRootObject>();
}

const SceneRootObject& SceneRoot::constGet()
{
    return *instace_().root_;
}

std::shared_ptr<const SceneRootObject> SceneRoot::constGetSharedPtr()
{
    return std::const_pointer_cast<const SceneRootObject >( instace_().root_ );
}

const std::filesystem::path& SceneRoot::getScenePath()
{
    return instace_().scenePath_;
}

MR_ADD_CLASS_FACTORY( SceneRootObject )

SceneRootObject::SceneRootObject()
{
    // this changes are required as root object has fixed properties
    setName( "Root" );
    setAncillary( false );
    select( false );
}

const std::string& SceneRootObject::name() const
{
    static std::string sRootName = "Root";
    return sRootName;
}

void SceneRootObject::serializeFields_( Json::Value& root ) const
{
    Object::serializeFields_( root );
    // append base type
    root["Type"].append( SceneRootObject::TypeName() );
}

void SceneRootObject::deserializeFields_( const Json::Value& root )
{
    Object::deserializeFields_( root );
    // this changes are required as root object has fixed properties
    setName( "Root" );
    setAncillary( false );
    select( false );
}

}