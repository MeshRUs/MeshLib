#include "MRViewerIO.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRVoxelsSave.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRObjectVoxels.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRPch/MRSpdlog.h"
#include "MRMenu.h"

namespace MR
{

std::string saveObjectToFile( const std::shared_ptr<VisualObject>& obj, const std::filesystem::path& filename, ProgressCallback callback )
{
    if ( !obj )
        return {};

    if ( !callback( 0.f ) )
        return "Saving canceled";

    std::string error;

    if ( auto objPoints = obj->asType<ObjectPoints>() )
    {
        if ( objPoints->pointCloud() )
        {
            const auto& colors = objPoints->getVertsColorMap();
            auto res = PointsSave::toAnySupportedFormat( *objPoints->pointCloud(), filename,
                                                         colors.empty() ? nullptr : &colors, callback );
            if ( !res.has_value() )
                error = res.error();
        }
        else
            error = "ObjectPoints has no PointCloud in it";
    }
    else if ( auto objLines = obj->asType<ObjectLines>() )
    {
        if ( objLines->polyline() )
        {
            auto res = LinesSave::toAnySupportedFormat( *objLines->polyline(), filename, callback );
            if ( !res.has_value() )
                error = res.error();
        }
        else
            error = "ObjectLines has no Polyline in it";
    }
    else if ( auto objMesh = obj->asType<ObjectMesh>() )
    {
        if ( objMesh->mesh() )
        {
            const Vector<Color, VertId>* colors{ nullptr };
            if ( objMesh->getColoringType() == ColoringType::VertsColorMap )
                colors = &obj->getVertsColorMap();

            auto res = MeshSave::toAnySupportedFormat( *objMesh->mesh(), filename, colors, callback );
            if ( !res.has_value() )
                error = res.error();
        }
        else
            error = "ObjectMesh has no Mesh in it";
    }
#ifndef __EMSCRIPTEN__
    else if ( auto objVoxels = obj->asType<ObjectVoxels>() )
    {
        auto ext = filename.extension().u8string();
        for ( auto& c : ext )
            c = ( char )tolower( c );

        if ( ext == u8".raw" )
        {
            auto res = VoxelsSave::saveRAW( filename, *objVoxels, callback );
            if ( !res.has_value() )
                error = res.error();
        }
    }
#endif

    if ( error.empty() )
        getViewerInstance().recentFilesStore.storeFile( filename );

    spdlog::error( error );
    return error;
}

}
