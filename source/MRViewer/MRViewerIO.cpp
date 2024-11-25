#include "MRViewerIO.h"

#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshSave.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRDistanceMapSave.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectDistanceMap.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRTimer.h"

#ifndef MRVIEWER_NO_VOXELS
#include <MRVoxels/MRVoxelsSave.h>
#include "MRVoxels/MRObjectVoxels.h"
#endif

#include "MRPch/MRSpdlog.h"

namespace MR
{

Expected<void> saveObjectToFile( const Object& obj, const std::filesystem::path& filename, const SaveObjectSettings & settings )
{
    MR_TIMER
    if ( !reportProgress( settings.callback, 0.f ) )
        return unexpected( std::string( "Saving canceled" ) );

    std::optional<std::filesystem::path> copyPath;
    std::error_code ec;
    std::string copySuffix = ".tmpcopy";
    if ( settings.backupOriginalFile && std::filesystem::is_regular_file( filename, ec ) )
    {
        copyPath = utf8string( filename ) + copySuffix;
        spdlog::info( "copy file {} into {}", utf8string( filename ), utf8string( copyPath.value() ) );
        std::filesystem::copy_file( filename, copyPath.value(), ec );
        if ( ec )
            spdlog::error( "copy file {} into {} failed: {}", utf8string( filename ), utf8string( copyPath.value() ), systemToUtf8( ec.message() ) );
    }

    spdlog::info( "save object to file {}", utf8string( filename ) );
    AffineXf3d xf( obj.worldXf() );
    SaveSettings saveSettings
    {
        .xf = ( xf == AffineXf3d() ) ? nullptr : &xf,
        .progress = settings.callback
    };
    Expected<void> result;

    if ( auto objPoints = obj.asType<ObjectPoints>() )
    {
        if ( objPoints->pointCloud() )
        {
            const auto& colors = objPoints->getVertsColorMap();
            if ( !colors.empty() )
                saveSettings.colors = &colors;
            result = PointsSave::toAnySupportedFormat( *objPoints->pointCloud(), filename, { saveSettings } );
        }
        else
            result = unexpected( std::string( "ObjectPoints has no PointCloud in it" ) );
    }
    else if ( auto objLines = obj.asType<ObjectLines>() )
    {
        if ( objLines->polyline() )
        {
            result = LinesSave::toAnySupportedFormat( *objLines->polyline(), filename, saveSettings );
        }
        else
            result = unexpected( std::string( "ObjectLines has no Polyline in it" ) );
    }
    else if ( auto objMesh = obj.asType<ObjectMesh>() )
    {
        if ( objMesh->mesh() )
        {
            if ( objMesh->getColoringType() == ColoringType::VertsColorMap )
                saveSettings.colors = &objMesh->getVertsColorMap();
            if ( objMesh->getUVCoords().size() >= objMesh->mesh()->topology.lastValidVert() )
                saveSettings.uvMap = &objMesh->getUVCoords();
            if ( !objMesh->getTexture().pixels.empty() )
                saveSettings.texture = &objMesh->getTexture();
            saveSettings.materialName = utf8string( filename.stem() );
            result = MeshSave::toAnySupportedFormat( *objMesh->mesh(), filename, saveSettings );
        }
        else
            result = unexpected( std::string( "ObjectMesh has no Mesh in it" ) );
    }
    else if ( auto objDistanceMap = obj.asType<ObjectDistanceMap>() )
    {
        if ( auto distanceMap = objDistanceMap->getDistanceMap() )
        {
            result = DistanceMapSave::toAnySupportedFormat( filename, *distanceMap, &objDistanceMap->getToWorldParameters() );
        }
        else
        {
            result = unexpected( std::string( "ObjectDistanceMap has no DistanceMap in it" ) );
        }
    }
#ifndef MRVIEWER_NO_VOXELS
    else if ( auto objVoxels = obj.asType<ObjectVoxels>() )
    {
        auto ext = filename.extension().u8string();
        for ( auto& c : ext )
            c = ( char )tolower( c );

        result = VoxelsSave::toAnySupportedFormat( objVoxels->vdbVolume(), filename, settings.callback );
    }
#endif

    if ( !result.has_value() )
    {
        spdlog::error( "save object to file {} failed: {}", utf8string( filename ), result.error() );
        spdlog::info( "remove file {}", utf8string( filename ) );
        std::filesystem::remove( filename, ec );
        if ( ec )
            spdlog::error( "remove file {} failed: {}", utf8string( filename ), systemToUtf8( ec.message() ) );
        if ( copyPath.has_value() )
        {
            spdlog::info( "rename file {} into {}", utf8string( copyPath.value() ), utf8string( filename ) );
            std::filesystem::rename( copyPath.value(), filename, ec );
            if ( ec )
                spdlog::error( "rename file {} into {} failed: {}", utf8string( copyPath.value() ), utf8string( filename ), systemToUtf8( ec.message() ) );
        }
    }
    else if ( copyPath.has_value() )
    {
        spdlog::info( "remove file {}", utf8string( *copyPath ) );
        std::filesystem::remove( *copyPath, ec );
        if ( ec )
            spdlog::error( "remove file {} failed: {}", utf8string( *copyPath ), systemToUtf8( ec.message() ) );
    }

    return result;
}

} //namespace MR
