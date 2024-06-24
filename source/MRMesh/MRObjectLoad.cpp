#include "MRObjectLoad.h"
#include "MRObjectMesh.h"
#include "MRMeshLoad.h"
#include "MRLinesLoad.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRDistanceMapLoad.h"
#include "MRImageLoad.h"
#include "MRPointsLoad.h"
#include "MRVoxelsLoad.h"
#include "MRObjectVoxels.h"
#include "MRObjectLines.h"
#include "MRObjectPoints.h"
#include "MRDistanceMap.h"
#include "MRObjectDistanceMap.h"
#include "MRObjectGcode.h"
#include "MRGcodeLoad.h"
#include "MRStringConvert.h"
#include "MRIOFormatsRegistry.h"
#include "MRMeshLoadObj.h"
#include "MRMeshLoadStep.h"
#include "MRSerializer.h"
#include "MRDirectory.h"
#include "MRSceneSettings.h"
#include "MRPch/MRSpdlog.h"
#include "MRMeshLoadSettings.h"
#include "MRZip.h"
#include "MRPointsLoadE57.h"
#include "MRMisonLoad.h"
#include "MRPch/MRTBB.h"

#ifndef MRMESH_NO_GLTF
#include "MRGltfSerializer.h"
#endif

namespace MR
{

namespace
{

std::optional<MR::IOFilter> findFilter( const MR::IOFilters& filters, const std::string& extension )
{
    const auto it = std::find_if( filters.begin(), filters.end(), [&extension] ( const MR::IOFilter& filter )
    {
        return std::string::npos != filter.extensions.find( extension );
    } );
    if ( it != filters.end() )
        return *it;
    else
        return std::nullopt;
}

/// finds if given mesh has enough sharp edges (>25 degrees) to recommend flat shading
bool detectFlatShading( const Mesh& mesh )
{
    MR_TIMER

    constexpr float sharpAngle = 25 * PI_F / 180; // Critical angle from planar, degrees
    const float sharpAngleCos = std::cos( sharpAngle );

    struct Data
    {
        double sumDblArea = 0;
        double sumSharpDblArea = 0;
        Data operator + ( const Data & b ) const 
        {
            return { sumDblArea + b.sumDblArea, sumSharpDblArea + b.sumSharpDblArea };
        }
    };

    auto total = parallel_deterministic_reduce(
        tbb::blocked_range( 0_ue, UndirectedEdgeId{ mesh.topology.undirectedEdgeSize() } ),
        Data(),
        [&mesh, sharpAngleCos] ( const auto& range, Data current )
        {
            for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
            {
                const EdgeId e = ue;
                const auto l = mesh.topology.left( e );
                const auto r = mesh.topology.right( e );
                if ( !l || !r )
                    continue;
                const auto da = mesh.dblArea( l ) + mesh.dblArea( r );
                current.sumDblArea += da;
                auto dihedralCos = mesh.dihedralAngleCos( ue );
                if ( dihedralCos <= sharpAngleCos )
                    current.sumSharpDblArea += da;
            }
            return current;
        },
        std::plus<Data>() );

    // triangles' area near sharp edges is more than 5% of total area
    return total.sumSharpDblArea > 0.05 * total.sumDblArea;
}

// Prepare object after it has been imported from external format (not .mru)
void postImportObject( const std::shared_ptr<Object> &o, const std::filesystem::path &filename )
{
    if ( std::shared_ptr<ObjectMesh> mesh = std::dynamic_pointer_cast< ObjectMesh >( o ) )
    {
        // Detect flat shading needed
        bool flat;
        if ( SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::AutoDetect )
            flat = filename.extension() == ".step" || filename.extension() == ".stp" ||
                   ( mesh->mesh() && detectFlatShading( *mesh->mesh().get() ) );
        else
            flat = SceneSettings::getDefaultShadingMode() == SceneSettings::ShadingMode::Flat;
        mesh->setVisualizeProperty( flat, MeshVisualizePropertyType::FlatShading, ViewportMask::all() );
    }
    for ( const std::shared_ptr<Object>& child : o->children() )
        postImportObject( child, filename );
}

} // namespace

const IOFilters allFilters = SceneFileFilters
                             | ObjectLoad::getFilters()
                             | MeshLoad::getFilters()
                             | VoxelsLoad::Filters
                             | LinesLoad::Filters
                             | PointsLoad::Filters;

Expected<ObjectMesh, std::string> makeObjectMeshFromFile( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    MeshLoadSettings newSettings = settings;
    VertColors colors;
    newSettings.colors = &colors;
    AffineXf3f xf;
    newSettings.xf = &xf;
    auto mesh = MeshLoad::fromAnySupportedFormat( file, newSettings );
    if ( !mesh.has_value() )
    {
        return unexpected( mesh.error() );
    }

    ObjectMesh objectMesh;
    objectMesh.setName( utf8string( file.stem() ) );
    objectMesh.setMesh( std::make_shared<MR::Mesh>( std::move( mesh.value() ) ) );
    if ( !colors.empty() )
    {
        objectMesh.setVertsColorMap( std::move( colors ) );
        objectMesh.setColoringType( ColoringType::VertsColorMap );
    }
    objectMesh.setXf( xf );

    return objectMesh;
}

Expected<std::shared_ptr<Object>, std::string> makeObjectFromMeshFile( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    MeshLoadSettings newSettings = settings;
    VertColors colors;
    newSettings.colors = &colors;
    VertNormals normals;
    newSettings.normals = &normals;
    AffineXf3f xf;
    newSettings.xf = &xf;
    auto mesh = MeshLoad::fromAnySupportedFormat( file, newSettings );
    if ( !mesh.has_value() )
        return unexpected( mesh.error() );
    
    if ( !mesh->points.empty() && mesh->topology.numValidFaces() <= 0 )
    {
        auto pointCloud = std::make_shared<MR::PointCloud>();
        pointCloud->points = std::move( mesh->points );
        pointCloud->normals = std::move( normals );
        pointCloud->validPoints.resize( pointCloud->points.size(), true );

        auto objectPoints = std::make_unique<ObjectPoints>();
        objectPoints->setName( utf8string( file.stem() ) );
        objectPoints->setPointCloud( pointCloud );

        if ( !colors.empty() )
        {
            objectPoints->setVertsColorMap( std::move( colors ) );
            objectPoints->setColoringType( ColoringType::VertsColorMap );
        }
        objectPoints->setXf( xf );

        return objectPoints;
    }

    auto objectMesh = std::make_unique<ObjectMesh>();
    objectMesh->setName( utf8string( file.stem() ) );
    objectMesh->setMesh( std::make_shared<MR::Mesh>( std::move( mesh.value() ) ) );
    if ( !colors.empty() )
    {
        objectMesh->setVertsColorMap( std::move( colors ) );
        objectMesh->setColoringType( ColoringType::VertsColorMap );
    }
    objectMesh->setXf( xf );

    return objectMesh;
}

Expected<ObjectLines, std::string> makeObjectLinesFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    auto lines = LinesLoad::fromAnySupportedFormat( file, callback );
    if ( !lines.has_value() )
    {
        return unexpected( lines.error() );
    }

    ObjectLines objectLines;
    objectLines.setName( utf8string( file.stem() ) );
    objectLines.setPolyline( std::make_shared<MR::Polyline3>( std::move( lines.value() ) ) );

    return objectLines;
}

Expected<ObjectPoints, std::string> makeObjectPointsFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    VertColors colors;
    AffineXf3f xf;
    auto pointsCloud = PointsLoad::fromAnySupportedFormat( file, &colors, &xf, callback );
    if ( !pointsCloud.has_value() )
    {
        return unexpected( pointsCloud.error() );
    }

    ObjectPoints objectPoints;
    objectPoints.setName( utf8string( file.stem() ) );
    objectPoints.setPointCloud( std::make_shared<MR::PointCloud>( std::move( pointsCloud.value() ) ) );
    objectPoints.setXf( xf );
    if ( !colors.empty() )
    {
        objectPoints.setVertsColorMap( std::move( colors ) );
        objectPoints.setColoringType( ColoringType::VertsColorMap );
    }

    return objectPoints;
}

Expected<ObjectDistanceMap, std::string> makeObjectDistanceMapFromFile( const std::filesystem::path& file, ProgressCallback callback )
{
    MR_TIMER;

    DistanceMapToWorld params;
    auto distanceMap = DistanceMapLoad::fromAnySupportedFormat( file, &params, callback );
    if ( !distanceMap.has_value() )
    {
        return unexpected( distanceMap.error() );
    }

    ObjectDistanceMap objectDistanceMap;
    objectDistanceMap.setName( utf8string( file.stem() ) );
    objectDistanceMap.setDistanceMap( std::make_shared<MR::DistanceMap>( std::move( distanceMap.value() ) ), params );

    return objectDistanceMap;
}

Expected<ObjectGcode, std::string> makeObjectGcodeFromFile( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;

    auto gcodeSource = GcodeLoad::fromAnySupportedFormat( file, callback );
    if ( !gcodeSource.has_value() )
    {
        return unexpected( gcodeSource.error() );
    }

    ObjectGcode objectGcode;
    objectGcode.setName( utf8string( file.stem() ) );
    objectGcode.setGcodeSource( std::make_shared<GcodeSource>( *gcodeSource ) );

    return objectGcode;
}

#ifndef MRMESH_NO_OPENVDB
Expected<std::vector<std::shared_ptr<ObjectVoxels>>, std::string> makeObjectVoxelsFromFile( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER;

    auto cb = callback;
    if ( cb )
        cb = [callback] ( float v ) { return callback( v / 3.f ); };
    auto loadRes = VoxelsLoad::fromAnySupportedFormat( file, cb );
    if ( !loadRes.has_value() )
    {
        return unexpected( loadRes.error() );
    }
    auto& loadResRef = *loadRes;
    std::vector<std::shared_ptr<ObjectVoxels>> res;
    int size = int( loadResRef.size() );
    for ( int i = 0; i < size; ++i )
    {
        std::shared_ptr<ObjectVoxels> obj = std::make_shared<ObjectVoxels>();
        const std::string name = i > 1 ? fmt::format( "{} {}", utf8string(file.stem()), i) : utf8string(file.stem());
        obj->setName( name );
        int step = 0;
        bool callbackRes = true;
        if ( cb )
            cb = [callback, &i, &step, size, &callbackRes] ( float v )
        {
            callbackRes = callback( ( 1.f + 2 * ( i + ( step + v ) / 2.f ) / size ) / 3.f );
            return callbackRes;
        };

        obj->construct( loadResRef[i], cb );
        if ( cb && !callbackRes )
            return unexpected( getCancelMessage( file ) );
        step = 1;
        obj->setIsoValue( ( loadResRef[i].min + loadResRef[i].max ) / 2.f, cb );
        if ( cb && !callbackRes )
            return unexpected( getCancelMessage( file ) );
        res.emplace_back( obj );
    }
    
    return res;
}
#endif

static std::string makeWarningString( int skippedFaceCount, int duplicatedVertexCount )
{
    std::string res;
    if ( skippedFaceCount )
        res = fmt::format( "{} triangles were skipped as inconsistent with others.", skippedFaceCount );
    if ( duplicatedVertexCount )
    {
        if ( !res.empty() )
            res += '\n';
        res += fmt::format( "{} vertices were duplicated to make them manifold.", duplicatedVertexCount );
    }
    return res;
}

Expected<std::vector<std::shared_ptr<MR::Object>>, std::string> loadObjectFromFile( const std::filesystem::path& filename,
                                                                                    std::string* loadWarn, ProgressCallback callback )
{
    if ( callback && !callback( 0.f ) )
        return unexpected( std::string( "Loading canceled" ) );

    Expected<std::vector<std::shared_ptr<Object>>, std::string> result;
    bool loadedFromSceneFile = false;

    auto ext = std::string( "*" ) + utf8string( filename.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );   
    
    if ( ext == "*.obj" )
    {
        MeshLoadSettings settings;
        settings.callback = callback;
        int skippedFaceCount = 0;
        settings.skippedFaceCount = &skippedFaceCount;
        int duplicatedVertexCount = 0;
        settings.duplicatedVertexCount = &duplicatedVertexCount;
        AffineXf3f xf;
        settings.xf = &xf;
        auto res = MeshLoad::fromSceneObjFile( filename, false, settings );
        if ( res.has_value() )
        {
            std::vector<std::shared_ptr<Object>> objects( res.value().size() );
            auto& resValue = *res;
            for ( int i = 0; i < objects.size(); ++i )
            {
                std::shared_ptr<ObjectMesh> objectMesh = std::make_shared<ObjectMesh>();
                if ( resValue[i].name.empty() )
                    objectMesh->setName( utf8string( filename.stem() ) );
                else
                    objectMesh->setName( std::move( resValue[i].name ) );
                objectMesh->select( true );
                objectMesh->setMesh( std::make_shared<Mesh>( std::move( resValue[i].mesh ) ) );
                if ( resValue[i].diffuseColor )
                    objectMesh->setFrontColor( *resValue[i].diffuseColor, false );

                objectMesh->setUVCoords( std::move( resValue[i].uvCoords ) );

                auto image = ImageLoad::fromAnySupportedFormat( resValue[i].pathToTexture );
                if ( image.has_value() )
                {
                    objectMesh->setTexture( { image.value(), FilterType::Linear } );
                    objectMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
                }
                if ( !resValue[i].colors.empty() )
                {
                    objectMesh->setVertsColorMap( std::move( resValue[i].colors ) );
                    objectMesh->setColoringType( ColoringType::VertsColorMap );
                }

                objectMesh->setXf( xf );

                objects[i] = std::dynamic_pointer_cast< Object >( objectMesh );
            }
            result = objects;

            if ( loadWarn )
                *loadWarn = makeWarningString( skippedFaceCount, duplicatedVertexCount );
        }
        else
            result = unexpected( res.error() );
    }
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
    else if ( ext == "*.e57" )
    {
        auto enclouds = PointsLoad::fromSceneE57File( filename, { .progress = callback } );
        if ( enclouds.has_value() )
        {
            auto& nclouds = *enclouds;
            std::vector<std::shared_ptr<Object>> objects( nclouds.size() );
            for ( int i = 0; i < objects.size(); ++i )
            {
                auto objectPoints = std::make_shared<ObjectPoints>();
                if ( nclouds[i].name.empty() )
                    objectPoints->setName( utf8string( filename.stem() ) );
                else
                    objectPoints->setName( std::move( nclouds[i].name ) );
                objectPoints->select( true );
                objectPoints->setPointCloud( std::make_shared<PointCloud>( std::move( nclouds[i].cloud ) ) );
                objectPoints->setXf( nclouds[i].xf );
                if ( !nclouds[i].colors.empty() )
                {
                    objectPoints->setVertsColorMap( std::move( nclouds[i].colors ) );
                    objectPoints->setColoringType( ColoringType::VertsColorMap );
                }
                objects[i] = std::dynamic_pointer_cast< Object >( std::move( objectPoints ) );
            }
            result = std::move( objects );
        }
        else
            result = unexpected( std::move( enclouds.error() ) );
    }
#endif //!defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_E57 )
    else if ( std::find_if( SceneFileFilters.begin(), SceneFileFilters.end(), [ext] ( const auto& filter ) { return filter.extensions.find( ext ) != std::string::npos; }) != SceneFileFilters.end() )
    {
        const auto objTree = loadSceneFromAnySupportedFormat( filename, loadWarn, callback );
        if ( !objTree.has_value() )
            return unexpected( objTree.error() );
        
        result = std::vector( { *objTree } );
        ( *result )[0]->setName( utf8string( filename.stem() ) );
        loadedFromSceneFile = true;
    }
    else if ( const auto filter = findFilter( ObjectLoad::getFilters(), ext ) )
    {
        const auto loader = ObjectLoad::getObjectLoader( *filter );
        result = loader( filename, loadWarn, std::move( callback ) );
    }
    else
    {
        MeshLoadSettings settings;
        settings.callback = callback;
        int skippedFaceCount = 0;
        int duplicatedVertexCount = 0;
        if ( loadWarn )
        {
            settings.skippedFaceCount = &skippedFaceCount;
            settings.duplicatedVertexCount = &duplicatedVertexCount;
        }
        auto object = makeObjectFromMeshFile( filename, settings );
        if ( object && *object )
        {
            (*object)->select( true );
            result = { *object };
            if ( loadWarn )
                *loadWarn = makeWarningString( skippedFaceCount, duplicatedVertexCount );
        }
        else if ( object.error() == "Loading canceled" )
        {
            result = unexpected( std::move( object.error() ) );
        }
        else
        {
            result = unexpected( std::move( object.error() ) );

            auto objectPoints = makeObjectPointsFromFile( filename, callback );
            if ( objectPoints.has_value() )
            {
                objectPoints->select( true );
                auto obj = std::make_shared<ObjectPoints>( std::move( objectPoints.value() ) );
                result = { obj };
            }
            else if ( result.error() == "unsupported file extension" )
            {
                result = unexpected( objectPoints.error() );

                auto objectLines = makeObjectLinesFromFile( filename, callback );
                if ( objectLines.has_value() )
                {
                    objectLines->select( true );
                    auto obj = std::make_shared<ObjectLines>( std::move( objectLines.value() ) );
                    result = { obj };
                }
                else if ( result.error() == "unsupported file extension" )
                {
                    result = unexpected( objectLines.error() );

                    auto objectDistanceMap = makeObjectDistanceMapFromFile( filename, callback );
                    if ( objectDistanceMap.has_value() )
                    {
                        objectDistanceMap->select( true );
                        auto obj = std::make_shared<ObjectDistanceMap>( std::move( objectDistanceMap.value() ) );
                        result = { obj };
                    }
                    else if ( result.error() == "unsupported file extension" )
                    {
                        result = unexpected( objectDistanceMap.error() );

                        auto objectGcode = makeObjectGcodeFromFile( filename, callback );
                        if ( objectGcode.has_value() )
                        {
                            objectGcode->select( true );
                            auto obj = std::make_shared<ObjectGcode>( std::move( objectGcode.value() ) );
                            result = { obj };
                        }
                        else if ( result.error() == "unsupported file extension" )
                        {
                            result = unexpected( objectDistanceMap.error() );

#ifndef MRMESH_NO_OPENVDB
                            auto objsVoxels = makeObjectVoxelsFromFile( filename, callback );
                            std::vector<std::shared_ptr<Object>> resObjs;
                            if ( objsVoxels.has_value() )
                            {
                                auto& objsVoxelsRef = *objsVoxels;
                                for ( auto& objPtr : objsVoxelsRef )
                                {
                                    objPtr->select( true );
                                    resObjs.emplace_back( std::dynamic_pointer_cast< Object >( objPtr ) );
                                }
                                result = resObjs;
                            }
                            else
                                result = unexpected( objsVoxels.error() );
#endif
                        }
                    }
                }
            }
        }
    }

    if ( result.has_value() && !loadedFromSceneFile )
        for ( const std::shared_ptr<Object>& o : result.value() )
        {
            postImportObject( o, filename );
            if ( auto objectPoints = o->asType<ObjectPoints>(); objectPoints && loadWarn )
            {
                if ( !objectPoints->pointCloud()->hasNormals() )
                    *loadWarn += "Point cloud " + o->name() + " has no normals.\n";
                if ( objectPoints->getRenderDiscretization() > 1 )
                    *loadWarn += "Point cloud " + o->name() + " has too many points in PointCloud:\n"
                    "Visualization is simplified (only part of the points is drawn)\n";
            }
        }

    if ( !result.has_value() )
        spdlog::error( result.error() );

    return result;
}


bool isSupportedFileInSubfolders( const std::filesystem::path& folder )
{
    std::vector<std::filesystem::path> filesList;
    filesList.push_back( folder );

    while ( !filesList.empty() )
    {
        auto path = filesList[0];
        filesList.erase( filesList.begin() );

        std::error_code ec;
        for ( auto entry : Directory{ path, ec } )
        {
            auto subpath = entry.path();
            if ( entry.is_directory( ec ) )
            {
                filesList.push_back( path = subpath );
            }
            else if ( entry.is_regular_file( ec ) )
            {
                auto ext = utf8string( subpath.extension() );
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext.empty() )
                    continue;

                if ( std::find_if( allFilters.begin(), allFilters.end(), [&ext] ( const IOFilter& f )
                    { return f.extensions.find( ext ) != std::string::npos; }) != allFilters.end() )
                    return true;
            }
        }
    }
    return false;
}

Expected<Object, std::string> makeObjectTreeFromFolder( const std::filesystem::path & folder, ProgressCallback callback )
{
    MR_TIMER;

    if ( callback && !callback( 0.f ) )
        return unexpected( getCancelMessage( folder ) );

    struct FilePathNode
    {
        std::filesystem::path path;
        std::vector<FilePathNode> subfolders;
        std::vector<FilePathNode> files;
    };

    FilePathNode filesTree;
    filesTree.path = folder;


    // Global variable is not correctly initialized in emscripten build
    const IOFilters filters = SceneFileFilters | MeshLoad::getFilters() | VoxelsLoad::Filters | LinesLoad::Filters | PointsLoad::Filters;

    std::function<void( FilePathNode& )> fillFilesTree = {};
    fillFilesTree = [&fillFilesTree, &filters] ( FilePathNode& node )
    {
        std::error_code ec;
        for ( auto entry : Directory{ node.path, ec } )
        {
            auto path = entry.path();
            if ( entry.is_directory( ec ) )
            {
                node.subfolders.push_back( { .path = path } );
                fillFilesTree( node.subfolders[node.subfolders.size() - 1] );
            }
            else if ( entry.is_regular_file( ec ) )
            {
                auto ext = utf8string( path.extension() );
                for ( auto& c : ext )
                    c = ( char )tolower( c );

                if ( ext.empty() )
                    continue;

                if ( std::find_if( filters.begin(), filters.end(), [&ext] ( const IOFilter& f )
                {
                    return f.extensions.find( ext ) != std::string::npos;
                } ) != filters.end() )
                    node.files.push_back( { .path = path } );
            }
        }
    };
    fillFilesTree( filesTree );

    // clear empty folders
    std::function<void( FilePathNode& )> clearEmptySubfolders = {};
    clearEmptySubfolders = [&clearEmptySubfolders] ( FilePathNode& node )
    {
        for ( int i = int( node.subfolders.size() ) - 1; i >= 0; --i )
        {
            clearEmptySubfolders( node.subfolders[i] );
            if ( node.subfolders[i].files.empty() && node.subfolders[i].subfolders.empty() )
                node.subfolders.erase( node.subfolders.begin() + i );
        }
    };
    clearEmptySubfolders( filesTree );

    
    if ( filesTree.subfolders.empty() && filesTree.files.empty() )
        return unexpected( std::string( "Error: folder is empty." ) );


    using loadObjResultType = Expected<std::vector<std::shared_ptr<MR::Object>>, std::string>;
    // create folders objects
    struct LoadTask
    {
        std::future<loadObjResultType> future;
        Object* parent = nullptr;
        LoadTask( std::future<loadObjResultType> future, Object* parent ) : future( std::move( future ) ), parent( parent ) {}
        bool finished = false;
    };
    std::vector<LoadTask> loadTasks;

    std::atomic_bool loadingCanceled = false;
    std::function<void( const FilePathNode&, Object* )> createFolderObj = {};
    createFolderObj = [&] ( const FilePathNode& node, Object* objPtr )
    {

        for ( const FilePathNode& folder : node.subfolders )
        {
            auto pObj = std::make_shared<Object>();
            pObj->setName( utf8string( folder.path.stem() ) );
            objPtr->addChild( pObj );
            createFolderObj( folder, pObj.get() );
        }
        for ( const FilePathNode& file : node.files )
        {
            loadTasks.emplace_back( std::async( std::launch::async, [&] ()
            {
                return loadObjectFromFile( file.path, nullptr, [&]( float ){ return !loadingCanceled; } );
            } ), objPtr );
        }
    };
    Object result;
    result.setName( utf8string( folder.stem() ) );
    createFolderObj( filesTree, &result );

    // processing of results
    bool atLeastOneLoaded = false;
    std::unordered_map<std::string, int> allErrors;
    const float taskCount = float( loadTasks.size() );
    int finishedTaskCount = 0;
    std::chrono::system_clock::time_point afterSecond = std::chrono::system_clock::now();
    while ( finishedTaskCount < taskCount )
    {
        afterSecond += +std::chrono::seconds( 1 );
        for ( auto& t : loadTasks )
        {
            if ( !t.future.valid() )
                continue;
            std::future_status status = t.future.wait_until( afterSecond );
            if ( status != std::future_status::ready )
                continue;
            auto res = t.future.get();
            if ( res.has_value() )
            {
                for ( const auto& objPtr : *res )
                {
                    t.parent->addChild( objPtr );
                }
                if ( !atLeastOneLoaded )
                    atLeastOneLoaded = true;
            }
            else
            {
                ++allErrors[res.error()];
            }
            ++finishedTaskCount;
            if ( callback && !callback( finishedTaskCount / taskCount ) )
                loadingCanceled = true;
        }
    }

    std::string errorString;
    for ( const auto& error : allErrors )
    {
        errorString += ( errorString.empty() ? "" : "\n" ) + error.first;
        if ( error.second > 1 )
        {
            errorString += std::string( " (" ) + std::to_string( error.second ) + std::string( ")" );
        }
    }

    if ( !errorString.empty() )
        spdlog::warn( "Load folder error:\n{}", errorString );
    if ( loadingCanceled )
        return unexpected( getCancelMessage( folder ) );
    if ( !atLeastOneLoaded )
        return unexpected( errorString );

    return result;
}

Expected <Object, std::string> makeObjectTreeFromZip( const std::filesystem::path& zipPath, ProgressCallback callback )
{
    auto tmpFolder = UniqueTemporaryFolder( {} );
    auto contentsFolder = tmpFolder / zipPath.stem();

    std::ifstream in( zipPath, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( zipPath.filename() ) );

    std::error_code ec;
    std::filesystem::create_directory( contentsFolder, ec );
    auto resZip = decompressZip( in, contentsFolder );
    if ( !resZip )
        return unexpected( "ZIP container error: " + resZip.error() );

    return makeObjectTreeFromFolder( contentsFolder, callback );
}

Expected<std::shared_ptr<Object>, std::string> loadSceneFromAnySupportedFormat( const std::filesystem::path& path, [[maybe_unused]] std::string* loadWarn,
    ProgressCallback callback )
{
    auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<std::shared_ptr<Object>> res = unexpected( std::string( "unsupported file extension" ) );

    auto itF = std::find_if( SceneFileFilters.begin(), SceneFileFilters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( itF == SceneFileFilters.end() )
        return res;

    if ( ext == "*.mru" )
    {
        res = deserializeObjectTree( path, {}, callback );
    }
#ifndef MRMESH_NO_GLTF
    else if ( ext == "*.gltf" || ext == "*.glb" )
    {
        res = deserializeObjectTreeFromGltf( path, callback );
    }
#endif
#ifndef MRMESH_NO_OPENCASCADE
    else if ( ext == "*.step" || ext == "*.stp" )
    {
        res = MeshLoad::fromSceneStepFile( path, { .callback = callback } );
    }
#endif
    else if ( ext == "*.zip" )
    {
        auto result = makeObjectTreeFromZip( path, callback );
        if ( result )
            res = std::make_shared<Object>( std::move( *result ) );
        else
            res = unexpected( result.error() );
    }
#ifndef __EMSCRIPTEN__
    else if ( ext == "*.mison" )
    {
        res = MR::fromSceneMison( path, loadWarn, callback );
    }
#endif // !__EMSCRIPTEN__

    if ( res.has_value() && ( ext != "*.mru" && ext != "*.zip" ) )
        postImportObject( res.value(), path );

    return res;
}

} //namespace MR
