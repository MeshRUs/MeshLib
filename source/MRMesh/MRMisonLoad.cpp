#include "MRMisonLoad.h"
#ifndef __EMSCRIPTEN__
#include "MRIOFormatsRegistry.h"
#include "MRSerializer.h"
#include "MRStringConvert.h"
#include "MRPch/MRJson.h"
#include "MRObjectLoad.h"
#include "MRAffineXf3.h"
#include "MRObject.h"
#include <fstream>

namespace MR
{

Expected<std::shared_ptr<MR::Object>> fromSceneMison( const std::filesystem::path& path, 
    std::string* loadWarn, ProgressCallback callback )
{
    std::ifstream in( path, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( path ) );

    return addFileNameInError( fromSceneMison( in, loadWarn, callback ), path );
}

// TODO: we need some code to prevent recursive openeing same file with this format
Expected<std::shared_ptr<MR::Object>> fromSceneMison( std::istream& in,
    std::string* loadWarn, ProgressCallback callback )
{
    auto rootVal = deserializeJsonValue( in );
    const auto invalidFormatError = unexpected( "Mison format invalid: " + rootVal.error() );
    if ( !rootVal.has_value() )
        return invalidFormatError;
    const Json::Value& rootJSObj = *rootVal;
    if ( !rootJSObj.isArray() && !rootJSObj["Objects"].isArray() )
        return invalidFormatError;
    const auto& root = rootJSObj.isArray() ? rootJSObj : rootJSObj["Objects"];
    int numFiles = int( root.size() );
    std::shared_ptr<MR::Object> rootObj = std::make_shared<MR::Object>();
    for ( int i = 0; i < numFiles; ++i )
    {
        const Json::Value& el = root[i];
        if ( !el["Filename"].isString() )
            return invalidFormatError;
        auto path = pathFromUtf8( el["Filename"].asString() );

        auto loadRes = loadObjectFromFile( path, subprogress( callback, i / float( numFiles ), ( i + 1 ) / float( numFiles ) ) );
        if ( !loadRes.has_value() )
            return unexpected( loadRes.error() + ": " + utf8string( path ) );

        if ( loadRes->objs.empty() )
            continue;
        if ( loadWarn && !loadRes->warnings.empty() )
            *loadWarn += ( loadRes->warnings + "\n" );

        AffineXf3f xf;
        if ( el["XF"].isObject() )
            deserializeFromJson( el["XF"], xf );
        std::string name;
        bool hasName = el["Name"].isString();
        if ( hasName )
        {
            name = el["Name"].asString();
            if ( loadRes->objs.size() == 1 )
                loadRes->objs.front()->setName(name);
            else
                for ( auto& obj : loadRes->objs )
                    obj->setName( name + ": " + obj->name() );
        }
        for ( auto& obj : loadRes->objs )
        {
            obj->setXf( xf * obj->xf() );
            rootObj->addChild( obj );
        }
    }

    // remove last "\n"
    if ( loadWarn && !loadWarn->empty() )
        loadWarn->erase( loadWarn->begin() + loadWarn->size() - 1 );

    return rootObj;
}

MR_ADD_SCENE_LOADER( IOFilter( "MeshInSpector Object Notation (.mison)", "*.mison" ), fromSceneMison )

}

#endif
