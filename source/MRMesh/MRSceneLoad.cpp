#include "MRSceneLoad.h"
#include "MRIOFormatsRegistry.h"
#include "MRObjectLoad.h"
#include "MRStringConvert.h"
#include "MRSceneRoot.h"

#include <MRPch/MRSpdlog.h>

namespace
{

using namespace MR;

// check if stream is empty (has no data)
inline bool isEmpty( std::ostream& os )
{
    return os.tellp() == std::streampos( 0 );
}

// find corresponding filter in the async object loader registry
std::optional<IOFilter> findAsyncObjectLoadFilter( const std::filesystem::path& path )
{
    // TODO: resolve GCC bug
    //auto ext = std::string( "*" ) + utf8string( path.extension().u8string() );
    auto ext = utf8string( path.extension().u8string() );
    ext = std::string( "*" ) + ext;
    for ( auto& c : ext )
        c = (char)std::tolower( c );

    const auto asyncFilters = AsyncObjectLoad::getFilters();
    const auto asyncFilter = std::find_if( asyncFilters.begin(), asyncFilters.end(), [&ext] ( auto&& filter )
    {
        return filter.extensions.find( ext ) != std::string::npos;
    } );
    if ( asyncFilter != asyncFilters.end() )
        return *asyncFilter;
    else
        return std::nullopt;
}

// helper class to unify scene construction process
class SceneConstructor
{
public:
    // gather objects and error and warning messages
    void process( const std::filesystem::path& path, Expected<LoadedObjects> res )
    {
        const auto fileName = utf8string( path );
        spdlog::info( "Load file {} - {}", fileName, res.has_value() ? "success" : res.error().c_str() );
        if ( !res.has_value() )
        {
            // TODO: user-defined error format
            if ( !isEmpty( errorSummary_ ) )
                errorSummary_ << "\n\n";
            if ( res.error().find( fileName ) == std::string::npos )
                errorSummary_ << fileName << ":\n" << res.error() << "\n";
            else
                errorSummary_ << res.error() << "\n";
            return;
        }
        if ( res.has_value() && !res->warnings.empty() )
        {
            // TODO: user-defined warning format
            if ( !isEmpty( warningSummary_ ) )
                warningSummary_ << "\n\n";
            if ( res->warnings.find( fileName ) == std::string::npos )
                warningSummary_ << fileName << ":\n" << res->warnings << "\n";
            else
                warningSummary_ << res->warnings << "\n";
        }

        const auto prevObjectCount = loadedObjects_.size();
        for ( auto& obj : res->objs )
            if ( obj )
                loadedObjects_.emplace_back( std::move( obj ) );
        if ( prevObjectCount != loadedObjects_.size() )
            loadedFiles_.emplace_back( path );
        else
            errorSummary_ << ( !isEmpty( errorSummary_ ) ? "\n" : "" ) << "\n" << fileName << ":\n" << "No objects found" << "\n";
    }

    // construct a scene object
    SceneLoad::SceneLoadResult construct() const
    {
        auto scene = std::make_shared<SceneRootObject>();

        bool constructed;
        if ( loadedObjects_.size() == 1 )
        {
            const auto& object = loadedObjects_.front();
            if ( object->typeName() == SceneRootObject::TypeName() || ( object->typeName() == Object::TypeName() && object->xf() == AffineXf3f() ) )
            {
                scene = createRootFormObject( object );
                constructed = false;
            }
            else
            {
                constructed = true;
                scene->addChild( object );
            }
        }
        else
        {
            constructed = true;
            for ( const auto& object : loadedObjects_ )
                scene->addChild( object );
        }

        SceneLoad::SceneLoadResult ret{
            .scene = std::move( scene ),
            .isSceneConstructed = constructed,
            .loadedFiles = loadedFiles_,
            .errorSummary = errorSummary_.str(),
            .warningSummary = warningSummary_.str(),
        };

        if ( ret.loadedFiles.empty() )
            ret.scene = nullptr; // Don't emit the root object on failure.

        return ret;
    }

private:
    std::vector<std::filesystem::path> loadedFiles_;
    std::vector<std::shared_ptr<Object>> loadedObjects_;
    std::ostringstream errorSummary_;
    std::ostringstream warningSummary_;
};

// async loading context
struct AsyncLoadContext
{
    std::vector<std::filesystem::path> paths;
    std::vector<Expected<LoadedObjects>> results;

    std::atomic_size_t asyncCount{ 0 };

    ProgressCallback progressCallback;
    std::map<size_t, float> asyncProgressMap;
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
    std::mutex asyncProgressMutex;
#endif

    void initializeProgressMap( const BitSet& asyncBitSet )
    {
        for ( const auto index : asyncBitSet )
            asyncProgressMap.emplace( index, .00f );
    }

    // create a progress callback for async task
    ProgressCallback progressCallbackFor( size_t index )
    {
        if ( !progressCallback )
            return {};

        return [this, index] ( float v )
        {
            float sum = .00f;
            {
#if !defined( __EMSCRIPTEN__ ) || defined( __EMSCRIPTEN_PTHREADS__ )
                std::unique_lock lock( asyncProgressMutex );
#endif
                asyncProgressMap[index] = v;
                for ( const auto& [_, v1] : asyncProgressMap )
                    sum += v1 / (float)asyncProgressMap.size();
            }
            return reportProgress( progressCallback, sum );
        };
    }
};

}

namespace MR::SceneLoad
{

SceneLoadResult fromAnySupportedFormat( const std::vector<std::filesystem::path>& files, ProgressCallback callback )
{
    SceneConstructor constructor;
    for ( auto index = 0ull; index < files.size(); ++index )
    {
        const auto& path = files[index];
        if ( path.empty() )
            continue;

        spdlog::info( "Loading file {}", utf8string( path ) );
        constructor.process( path, loadObjectFromFile( path, subprogress( callback, index, files.size() ) ) );
    }
    return constructor.construct();
}

void asyncFromAnySupportedFormat( const std::vector<std::filesystem::path>& files,
                                  SceneLoad::PostLoadCallback postLoadCallback, ProgressCallback progressCallback )
{
    auto ctx = std::make_shared<AsyncLoadContext>();
    ctx->paths = files;
    std::erase_if( ctx->paths, [] ( auto&& path ) { return path.empty(); } );

    const auto count = ctx->paths.size();
    ctx->results.resize( count, unexpected( "Uninitialized" ) );

    size_t syncIndex = 0;
    BitSet asyncBitSet( count );
    for ( auto index = 0ull; index < count; ++index )
    {
        const auto& path = ctx->paths[index];
        if ( findAsyncObjectLoadFilter( path ) )
        {
            asyncBitSet.set( index );
        }
        else
        {
            spdlog::info( "Loading file {}", utf8string( path ) );
            ctx->results[index] = loadObjectFromFile( path, subprogress( progressCallback, syncIndex++, count ) );
        }
    }
    assert( syncIndex + asyncBitSet.count() == count );

    ctx->progressCallback = subprogress( progressCallback, (float)syncIndex / (float)count, 1.00f );
    ctx->initializeProgressMap( asyncBitSet );

    auto postLoad = [ctx, count, postLoadCallback]
    {
        SceneConstructor constructor;
        for ( auto index = 0ull; index < count; ++index )
            constructor.process( ctx->paths[index], ctx->results[index] );
        postLoadCallback( constructor.construct() );
    };

    if ( asyncBitSet.none() )
        return postLoad();

    ctx->asyncCount = asyncBitSet.count();
    for ( const auto index : asyncBitSet )
    {
        const auto& path = ctx->paths[index];
        const auto asyncFilter = findAsyncObjectLoadFilter( path );
        assert( asyncFilter );
        const auto asyncLoader = AsyncObjectLoad::getObjectLoader( *asyncFilter );
        assert( asyncLoader );
        const auto callback = ctx->progressCallbackFor( index );
        spdlog::info( "Async loading file {}", utf8string( path ) );
        asyncLoader( path, [ctx, index, postLoad, callback] ( Expected<LoadedObjects> result )
        {
            ctx->results[index] = std::move( result );
            reportProgress( callback, 1.00f );
            if ( ctx->asyncCount.fetch_sub( 1 ) == 1 )
                // that was the last file
                postLoad();
        }, callback );
    }
}

} // namespace MR::SceneLoad
