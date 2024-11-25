#pragma once
#include "MRMeshFwd.h"
#ifndef MR_PARSING_FOR_PB11_BINDINGS
#include "MRExpected.h"
#include "MRIOFilters.h"
#include "MRMeshLoadSettings.h"
#include "MROnInit.h"
#include "MRPointsLoadSettings.h"
#include "MRSaveSettings.h"
#include "MRLoadedObjects.h"

#include <filesystem>
#include <map>

#define MR_FORMAT_REGISTRY_EXTERNAL_DECL( API_ATTR, ProcName )                                               \
API_ATTR ProcName MR_CONCAT( get, ProcName )( const IOFilter& filter );                                      \
API_ATTR ProcName MR_CONCAT( get, ProcName )( const std::string& extension );                                \
API_ATTR void MR_CONCAT( set, ProcName )( const IOFilter& filter, ProcName proc, int8_t priorityScore = 0 ); \
API_ATTR const IOFilters& getFilters();

#define MR_FORMAT_REGISTRY_DECL( ProcName ) MR_FORMAT_REGISTRY_EXTERNAL_DECL( MRMESH_API, ProcName )

#define MR_FORMAT_REGISTRY_IMPL( ProcName )                                                         \
ProcName MR_CONCAT( get, ProcName )( const IOFilter& filter )                                       \
{                                                                                                   \
    return FormatRegistry<ProcName>::getProcessor( filter );                                        \
}                                                                                                   \
ProcName MR_CONCAT( get, ProcName )( const std::string& extension )                                 \
{                                                                                                   \
    return FormatRegistry<ProcName>::getProcessor( extension );                                     \
}                                                                                                   \
void MR_CONCAT( set, ProcName )( const IOFilter& filter, ProcName processor, int8_t priorityScore ) \
{                                                                                                   \
    FormatRegistry<ProcName>::setProcessor( filter, processor, priorityScore );                     \
}                                                                                                   \
const IOFilters& getFilters()                                                                       \
{                                                                                                   \
    return FormatRegistry<ProcName>::getFilters();                                                  \
}

namespace MR
{

MRMESH_API extern const IOFilters AllFilter;

/// format loader registry
/// NOTE: this is a singleton class, do NOT access it from header files
/// you might use MR_FORMAT_REGISTRY_DECL and MR_FORMAT_REGISTRY_IMPL macros to simplify the usage for most common cases
template <typename Processor>
class FormatRegistry
{
public:
    // get all registered filters
    static const IOFilters& getFilters()
    {
        return get_().filters_;
    }

    // get a registered loader for the filter
    static Processor getProcessor( const IOFilter& filter )
    {
        const auto& processors = get_().processors_;
        auto it = processors.find( filter );
        if ( it != processors.end() )
            return it->second;
        else
            return {};
    }

    // get a registered loader for the extension
    static Processor getProcessor( const std::string& extension )
    {
        const auto& processors = get_().processors_;
        // TODO: extension cache
        auto it = std::find_if( processors.begin(), processors.end(), [&extension] ( auto&& item )
        {
            const auto& [filter, _] = item;
            return filter.isSupportedExtension( extension );
        } );
        if ( it != processors.end() )
            return it->second;
        else
            return {};
    }

    // register or update a loader for the filter
    static void setProcessor( const IOFilter& filter, Processor processor, int8_t priorityScore = 0 )
    {
        auto& processors = get_().processors_;
        auto it = processors.find( filter );
        if ( it != processors.end() )
        {
            it->second = processor;
        }
        else
        {
            processors.emplace( filter, processor );

            auto& filters = get_().filterPriorityQueue_;
            filters.emplace( priorityScore, filter );
            get_().updateFilterList_();
        }
    }

private:
    FormatRegistry() = default;
    ~FormatRegistry() = default;

    static FormatRegistry<Processor>& get_()
    {
        static FormatRegistry<Processor> instance;
        return instance;
    }

    void updateFilterList_()
    {
        filters_.clear();
        filters_.reserve( filterPriorityQueue_.size() );
        for ( const auto& [_, filter] : filterPriorityQueue_ )
            filters_.emplace_back( filter );
    }

    std::map<IOFilter, Processor> processors_;
    std::multimap<int8_t, IOFilter> filterPriorityQueue_;
    std::vector<IOFilter> filters_;
};

namespace MeshLoad
{

/// \defgroup IOFormatsRegistryGroup IO Formats Registry
/// \ingroup IOGroup
/// \{

using MeshFileLoader = Expected<MR::Mesh>( * )( const std::filesystem::path&, const MeshLoadSettings& );
using MeshStreamLoader = Expected<MR::Mesh>( * )( std::istream&, const MeshLoadSettings& );

struct MeshLoader
{
    MeshFileLoader fileLoad{ nullptr };
    MeshStreamLoader streamLoad{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( MeshLoader )

/** 
 * \brief Register filter with loader function
 * \details loader function signature: Expected<Mesh> fromFormat( const std::filesystem::path& path, const MeshLoadSettings& settings );
 * example:
 * MR_ADD_MESH_LOADER( IOFilter("Name of filter (.ext)","*.ext"), fromFormat)
 */
#define MR_ADD_MESH_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::MeshLoad; setMeshLoader( filter, { static_cast<MeshFileLoader>( loader ), static_cast<MeshStreamLoader>( loader ) } ); };

#define MR_ADD_MESH_LOADER_WITH_PRIORITY( filter, loader, priority ) \
MR_ON_INIT { using namespace MR::MeshLoad; setMeshLoader( filter, { static_cast<MeshFileLoader>( loader ), static_cast<MeshStreamLoader>( loader ) }, priority ); };

/// \}

} // namespace MeshLoad

namespace MeshSave
{

using MeshFileSaver = Expected<void>( * )( const Mesh&, const std::filesystem::path&, const SaveSettings& );
using MeshStreamSaver = Expected<void>( * )( const Mesh&, std::ostream&, const SaveSettings& );

struct MeshSaver
{
    MeshFileSaver fileSave{ nullptr };
    MeshStreamSaver streamSave{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( MeshSaver )

#define MR_ADD_MESH_SAVER( filter, saver ) \
MR_ON_INIT { using namespace MR::MeshSave; setMeshSaver( filter, { static_cast<MeshFileSaver>( saver ), static_cast<MeshStreamSaver>( saver ) } ); };

#define MR_ADD_MESH_SAVER_WITH_PRIORITY( filter, saver, priority ) \
MR_ON_INIT { using namespace MR::MeshSave; setMeshSaver( filter, { static_cast<MeshFileSaver>( saver ), static_cast<MeshStreamSaver>( saver ) }, priority ); };

} // namespace MeshSave

namespace LinesLoad
{

using LinesFileLoader = Expected<Polyline3>( * )( const std::filesystem::path&, ProgressCallback );
using LinesStreamLoader = Expected<Polyline3>( * )( std::istream&, ProgressCallback );

struct LinesLoader
{
    LinesFileLoader fileLoad{ nullptr };
    LinesStreamLoader streamLoad{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( LinesLoader )

#define MR_ADD_LINES_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::LinesLoad; setLinesLoader( filter, { static_cast<LinesFileLoader>( loader ), static_cast<LinesStreamLoader>( loader ) } ); };

#define MR_ADD_LINES_LOADER_WITH_PRIORITY( filter, loader, priority ) \
MR_ON_INIT { using namespace MR::LinesLoad; setLinesLoader( filter, { static_cast<LinesFileLoader>( loader ), static_cast<LinesStreamLoader>( loader ) }, priority ); };

} // namespace LinesLoad

namespace LinesSave
{

using LinesFileSaver = Expected<void>( * )( const Polyline3&, const std::filesystem::path&, const SaveSettings& );
using LinesStreamSaver = Expected<void>( * )( const Polyline3&, std::ostream&, const SaveSettings& );

struct LinesSaver
{
    LinesFileSaver fileSave{ nullptr };
    LinesStreamSaver streamSave{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( LinesSaver )

#define MR_ADD_LINES_SAVER( filter, saver ) \
MR_ON_INIT { using namespace MR::LinesSave; setLinesSaver( filter, { static_cast<LinesFileSaver>( saver ), static_cast<LinesStreamSaver>( saver ) } ); };

#define MR_ADD_LINES_SAVER_WITH_PRIORITY( filter, saver, priority ) \
MR_ON_INIT { using namespace MR::LinesSave; setLinesSaver( filter, { static_cast<LinesFileSaver>( saver ), static_cast<LinesStreamSaver>( saver ) }, priority ); };

} // namespace LinesSave

namespace PointsLoad
{

using PointsFileLoader = Expected<PointCloud>( * )( const std::filesystem::path&, const PointsLoadSettings& );
using PointsStreamLoader = Expected<PointCloud>( * )( std::istream&, const PointsLoadSettings& );

struct PointsLoader
{
    PointsFileLoader fileLoad{ nullptr };
    PointsStreamLoader streamLoad{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( PointsLoader )

#define MR_ADD_POINTS_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::PointsLoad; setPointsLoader( filter, { static_cast<PointsFileLoader>( loader ), static_cast<PointsStreamLoader>( loader ) } ); };

} // namespace PointsLoad

namespace PointsSave
{

using PointsFileSaver = Expected<void>( * )( const PointCloud&, const std::filesystem::path&, const SaveSettings& );
using PointsStreamSaver = Expected<void>( * )( const PointCloud&, std::ostream&, const SaveSettings& );

struct PointsSaver
{
    PointsFileSaver fileSave{ nullptr };
    PointsStreamSaver streamSave{ nullptr };
};

MR_FORMAT_REGISTRY_DECL( PointsSaver )

#define MR_ADD_POINTS_SAVER( filter, saver ) \
MR_ON_INIT { using namespace MR::PointsSave; setPointsSaver( filter, { static_cast<PointsFileSaver>( saver ), static_cast<PointsStreamSaver>( saver ) } ); };

} // namespace PointsSave

namespace ImageLoad
{

using ImageLoader = Expected<Image>( * )( const std::filesystem::path& );

MR_FORMAT_REGISTRY_DECL( ImageLoader )

#define MR_ADD_IMAGE_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::ImageLoad; setImageLoader( filter, loader ); };

#define MR_ADD_IMAGE_LOADER_WITH_PRIORITY( filter, loader, priority ) \
MR_ON_INIT { using namespace MR::ImageLoad; setImageLoader( filter, loader, priority ); };

} // namespace ImageLoad

namespace ImageSave
{

using ImageSaver = Expected<void>( * )( const Image&, const std::filesystem::path& );

MR_FORMAT_REGISTRY_DECL( ImageSaver )

#define MR_ADD_IMAGE_SAVER( filter, saver ) \
MR_ON_INIT { using namespace MR::ImageSave; setImageSaver( filter, saver ); };

#define MR_ADD_IMAGE_SAVER_WITH_PRIORITY( filter, saver, priority ) \
MR_ON_INIT { using namespace MR::ImageSave; setImageSaver( filter, saver, priority ); };

} // namespace ImageSave

namespace ObjectLoad
{

using ObjectLoader = Expected<LoadedObjects>( * )( const std::filesystem::path&, const ProgressCallback& );

MR_FORMAT_REGISTRY_DECL( ObjectLoader )

#define MR_ADD_OBJECT_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::ObjectLoad; setObjectLoader( filter, loader ); };

} // namespace ObjectLoad

namespace ObjectSave
{

using ObjectSaver = Expected<void>( * )( const Object&, const std::filesystem::path&, const ProgressCallback& );

MR_FORMAT_REGISTRY_DECL( ObjectSaver )

#define MR_ADD_OBJECT_SAVER( filter, saver ) \
MR_ON_INIT { using namespace MR::ObjectSave; setObjectSaver( filter, saver ); };

} // namespace ObjectSave

namespace AsyncObjectLoad
{

using PostLoadCallback = std::function<void ( Expected<LoadedObjects> )>;
using ObjectLoader = void( * )( const std::filesystem::path&, PostLoadCallback, const ProgressCallback& );

MR_FORMAT_REGISTRY_DECL( ObjectLoader )

} // namespace AsyncObjectLoad

namespace SceneLoad
{

using SceneLoader = Expected<ObjectPtr>( * )( const std::filesystem::path&, std::string*, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( SceneLoader )

#define MR_ADD_SCENE_LOADER( filter, loader ) \
MR_ON_INIT { using namespace MR::SceneLoad; setSceneLoader( filter, loader ); };

#define MR_ADD_SCENE_LOADER_WITH_PRIORITY( filter, loader, priority ) \
MR_ON_INIT { using namespace MR::SceneLoad; setSceneLoader( filter, loader, priority ); };

} // namespace SceneLoad

namespace SceneSave
{

using SceneSaver = Expected<void>( * )( const Object&, const std::filesystem::path&, ProgressCallback );

MR_FORMAT_REGISTRY_DECL( SceneSaver )

#define MR_ADD_SCENE_SAVER( filter, saver ) \
MR_ON_INIT { using namespace MR::SceneSave; setSceneSaver( filter, saver ); };

#define MR_ADD_SCENE_SAVER_WITH_PRIORITY( filter, saver, priority ) \
MR_ON_INIT { using namespace MR::SceneSave; setSceneSaver( filter, saver, priority ); };

} // namespace SceneSave

} // namespace MR
#endif
