#include "MRVoxelsLoad.h"

#include "MRMesh/MRIOFormatsRegistry.h"
#include "MRMesh/MRProgressCallback.h"
#include "MRMesh/MRTimer.h"
#include "MRObjectVoxels.h"
#include "MRScanHelpers.h"
#include "MRVDBConversions.h"
#include "MRMesh/MRStringConvert.h"
#include "MRVDBFloatGrid.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRDirectory.h"
#include "MROpenVDBHelper.h"
#include "MRMesh/MRParallelFor.h"
#include "MROpenVDB.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"

#ifndef MRVOXELS_NO_TIFF
#include "MRMesh/MRTiffIO.h"
#endif // MRVOXELS_NO_TIFF

#include <openvdb/io/Stream.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Interpolation.h>

#include <cfloat>
#include <fstream>

namespace MR
{

namespace VoxelsLoad
{

Expected<RawParameters> findRawParameters( std::filesystem::path& path )
{
    if ( path.empty() )
        return unexpected( "Path is empty" );

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    if ( ext != ".raw" )
    {
        std::stringstream ss;
        ss << "Extension is not correct, expected \".raw\" current \"" << ext << "\"" << std::endl;
        return unexpected( fmt::format( "Extension is not correct, expected \".raw\" current \"{}\"", ext ) );
    }

    auto parentPath = path.parent_path();
    std::error_code ec;
    if ( parentPath.empty() )
        parentPath = ".";
    else if ( !std::filesystem::is_directory( parentPath, ec ) )
        return unexpected( utf8string( parentPath ) + " is not existing directory" );
    std::vector<std::filesystem::path> candidatePaths;
    for ( auto entry : Directory{ parentPath, ec } )
    {
        auto filename = entry.path().filename();
        auto pos = utf8string( filename ).find( utf8string( path.filename() ) );
        if ( pos != std::string::npos )
            candidatePaths.push_back( entry.path() );
    }
    if ( candidatePaths.empty() )
        return unexpected( "Cannot find file: " + utf8string( path.filename() ) );
    else if ( candidatePaths.size() > 1 )
        return unexpected( "More than one file exists: " + utf8string( path.filename() ) );

    RawParameters outParams;
    path = candidatePaths[0];
    auto filename = utf8string( path.filename() );
    auto wEndChar = filename.find("_");
    if ( wEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto wString = filename.substr( 1, wEndChar - 1 );
    outParams.dimensions.x = std::atoi( wString.c_str() );

    auto hEndChar = filename.find( "_", wEndChar + 1 );
    if ( hEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto hString = filename.substr( wEndChar + 2, hEndChar - ( wEndChar + 2 ) );
    outParams.dimensions.y = std::atoi( hString.c_str() );

    auto sEndChar = filename.find( "_", hEndChar + 1 );
    if ( sEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto sString = filename.substr( hEndChar + 2, sEndChar - ( hEndChar + 2 ) );
    outParams.dimensions.z = std::atoi( sString.c_str() );

    auto xvEndChar = filename.find( "_", sEndChar + 1 );
    if ( xvEndChar == std::string::npos )
        return unexpected( "Cannot parse filename: " + filename );
    auto xvString = filename.substr( sEndChar + 2, xvEndChar - ( sEndChar + 2 ) );
    outParams.voxelSize.x = float(std::atof( xvString.c_str() ) / 1000); // convert mm to meters

    if ( filename[xvEndChar + 1] == 'G' )
    {
        auto gtEndChar = filename.find( "_", xvEndChar + 1 );
        if ( gtEndChar != std::string::npos )
        {
            auto gtString = filename.substr( xvEndChar + 2, gtEndChar - ( xvEndChar + 2 ) );
            outParams.gridLevelSet = gtString == "1"; // convert mm to meters
        }
    }
    if ( filename[xvEndChar + 1] == 'F' ) // end of prefix
    {
        outParams.voxelSize.y = outParams.voxelSize.z = outParams.voxelSize.x;
    }
    else
    {
        auto yvEndChar = filename.find( "_", xvEndChar + 1 );
        if ( yvEndChar == std::string::npos )
            return unexpected( "Cannot parse filename: " + filename );
        auto yvString = filename.substr( xvEndChar + 1, yvEndChar - ( xvEndChar + 1 ) );
        outParams.voxelSize.y = float( std::atof( yvString.c_str() ) / 1000 ); // convert mm to meters

        auto zvEndChar = filename.find( "_", yvEndChar + 1 );
        if ( zvEndChar == std::string::npos )
            return unexpected( "Cannot parse filename: " + filename );
        auto zvString = filename.substr( yvEndChar + 1, zvEndChar - ( yvEndChar + 1 ) );
        outParams.voxelSize.z = float( std::atof( zvString.c_str() ) / 1000 ); // convert mm to meters

        auto gtEndChar = filename.find( "_", zvEndChar + 1 );
        if ( gtEndChar != std::string::npos )
        {
            auto gtString = filename.substr( zvEndChar + 2, gtEndChar - ( zvEndChar + 2 ) );
            outParams.gridLevelSet = gtString == "1"; // convert mm to meters
        }
    }
    outParams.scalarType = ScalarType::Float32;
    return outParams;
}

Expected<VdbVolume> fromRaw( const std::filesystem::path& path, const ProgressCallback& cb )
{
    auto filepathToOpen = path;
    auto expParams = findRawParameters( filepathToOpen );
    if ( !expParams )
        return unexpected( std::move( expParams.error() ) );
    return fromRaw( filepathToOpen, *expParams, cb );
}

Expected<std::vector<VdbVolume>> fromVdb( const std::filesystem::path& path, const ProgressCallback& cb /*= {} */ )
{
    if ( cb && !cb( 0.f ) )
        return unexpected( getCancelMessage( path ) );

    std::vector<VdbVolume> res;

    openvdb::GridPtrVecPtr grids;
    {
        // in order to load on Windows a file with Unicode symbols in the name, we need to open ifstream by ourselves,
        // because openvdb constructs it from std::string, which on Windows means "local codepage" and not Unicode
        std::ifstream file( path, std::ios::binary );
        if ( !file )
            return unexpected( "cannot open file for reading: " + utf8string( path ) );

        openvdb::initialize();
        openvdb::io::Stream stream( file, false );
        grids = stream.getGrids();
    }
    if ( grids )
    {
        auto& gridsRef = *grids;
        if ( grids->size() == 0 )
            unexpected( std::string( "Nothing to load" ) );

        bool anyLoaded = false;
        int size = int( gridsRef.size() );
        int i = 0;
        ProgressCallback scaledCb;
        if ( cb )
            scaledCb = [cb, &i, size] ( float v ) { return cb( ( i + v ) / size ); };
        for ( i = 0; i < size; ++i )
        {
            if ( !gridsRef[i] )
                continue;

            OpenVdbFloatGrid ovfg( std::move( *std::dynamic_pointer_cast< openvdb::FloatGrid >( gridsRef[i] ) ) );
            VdbVolume vdbVolume;
            vdbVolume.data = std::make_shared<OpenVdbFloatGrid>( std::move( ovfg ) );

            if ( !vdbVolume.data )
                continue;

            const auto dims = vdbVolume.data->evalActiveVoxelDim();
            const auto voxelSize = vdbVolume.data->voxelSize();
            for ( int j = 0; j < 3; ++j )
            {
                vdbVolume.dims[j] = dims[j];
                vdbVolume.voxelSize[j] = float( voxelSize[j] );
            }
            evalGridMinMax( vdbVolume.data, vdbVolume.min, vdbVolume.max );

            if ( scaledCb && !scaledCb( 0.1f ) )
                return unexpected( getCancelMessage( path ) );

            openvdb::math::Transform::Ptr transformPtr = std::make_shared<openvdb::math::Transform>();
            vdbVolume.data->setTransform( transformPtr );

            translateToZero( *vdbVolume.data );

            if ( cb && !cb( (1.f + i ) / size ) )
                return unexpected( getCancelMessage( path ) );

            res.emplace_back( std::move( vdbVolume ) );

            anyLoaded = true;
        }
        if ( !anyLoaded )
            unexpected( std::string( "No loaded grids" ) );
    }
    else
        unexpected( std::string( "Nothing to read" ) );

    if ( cb )
        cb( 1.f );

    return res;
}

inline Expected<std::vector<VdbVolume>> toSingleElementVector( VdbVolume&& v )
{
    std::vector<VdbVolume> ret;
    ret.push_back( std::move( v ) ); // Not using `return std::vector{ std::move( v ) }` because that would always copy `v`.
    return ret;
}

Expected<std::vector<VdbVolume>> vecFromRaw(  const std::filesystem::path& path, const ProgressCallback& cb )
{
    return fromRaw( path, cb ).and_then( toSingleElementVector );
}

Expected<std::vector<VdbVolume>> vecFromGav(  const std::filesystem::path& path, const ProgressCallback& cb )
{
    return fromGav( path, cb ).and_then( toSingleElementVector );
}

MR_FORMAT_REGISTRY_IMPL( VoxelsLoader )

Expected<std::vector<VdbVolume>> fromAnySupportedFormat( const std::filesystem::path& path, const ProgressCallback& cb /*= {} */ )
{
    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );
    ext = "*" + ext;

    auto loader = getVoxelsLoader( ext );
    if ( !loader )
        return unexpected( std::string( "unsupported file extension" ) );
    return loader( path, cb );
}

Expected<std::vector<std::shared_ptr<ObjectVoxels>>> toObjectVoxels( const std::vector<VdbVolume>& volumes, const std::filesystem::path& file, const ProgressCallback& callback )
{
    std::vector<std::shared_ptr<ObjectVoxels>> res;
    const auto size = volumes.size();
    for ( size_t i = 0; i < size; ++i )
    {
        auto cb = subprogress( callback, i, size );

        auto& volume = volumes[i];

        auto obj = std::make_shared<ObjectVoxels>();
        const std::string name = i > 1 ? fmt::format( "{} {}", utf8string( file.stem() ), (int)i ) : utf8string( file.stem() );
        obj->setName( name );

        obj->construct( volume );
        obj->setIsoValue( ( volume.min + volume.max ) / 2.f, cb );
        if ( !reportProgress( cb, 1.0f ) )
            return unexpected( getCancelMessage( file ) );

        res.emplace_back( obj );
    }
    return res;
}

LoadedObjects toObjects( std::vector<std::shared_ptr<ObjectVoxels>>&& voxels )
{
    LoadedObjects res;
    res.objs.reserve( voxels.size() );
    for ( auto&& objVoxels : voxels )
    {
        objVoxels->select( true );
        res.objs.emplace_back( std::move( objVoxels ) );
    }
    return res;
}

template <VoxelsLoader voxelsLoader>
Expected<LoadedObjects> toObjectLoader( const std::filesystem::path& path, const ProgressCallback& cb )
{
    return voxelsLoader( path, subprogress( cb, 0.f, 1.f / 3.f ) )
        .and_then( [&] ( auto&& volumes ) { return toObjectVoxels( volumes, path, subprogress( cb, 1.f / 3.f, 1.f ) ); } )
        .transform( toObjects );
}

#define MR_ADD_VOXELS_LOADER( filter, loader )                         \
MR_ON_INIT {                                                           \
    MR::VoxelsLoad::setVoxelsLoader( filter, loader );                 \
    /* additionally register the loader as an object loader */         \
    MR::ObjectLoad::setObjectLoader( filter, toObjectLoader<loader> ); \
};

MR_ADD_VOXELS_LOADER( IOFilter( "Raw (.raw)", "*.raw" ), vecFromRaw )
MR_ADD_VOXELS_LOADER( IOFilter( "Micro CT (.gav)", "*.gav" ), vecFromGav )
MR_ADD_VOXELS_LOADER( IOFilter( "OpenVDB (.vdb)", "*.vdb" ), fromVdb )

#ifndef MRVOXELS_NO_TIFF
struct TiffParams
{
    int bitsPerSample = 0;
    int samplesPerPixel = 0;
    int width = 0;
    int height = 0;

    bool operator==( const TiffParams& other ) const
    {
        return bitsPerSample == other.bitsPerSample &&
            samplesPerPixel == other.samplesPerPixel &&
            width == other.width &&
            height == other.height;
    }

    bool operator !=( const TiffParams& other ) const
    {
        return !( *this == other );
    }
};

Expected<VdbVolume> loadTiffDir( const LoadingTiffSettings& settings )
{
    std::error_code ec;
    if ( !std::filesystem::is_directory( settings.dir, ec ) )
        return unexpected( "Given path is not directory" );

    int filesNum = 0;
    std::vector<std::filesystem::path> files;
    for ( auto entry : Directory{ settings.dir, ec } )
    {
        if ( entry.is_regular_file( ec ) )
            ++filesNum;
    }
    files.reserve( filesNum );
    for ( auto entry : Directory{ settings.dir, ec } )
    {
        auto filePath = entry.path();
        if ( entry.is_regular_file( ec ) && isTIFFFile( filePath ) )
            files.push_back( filePath );
    }

    if ( files.size() < 2 )
        return unexpected( "Too few TIFF files in the directory" );

    sortScanFilesByName( files );

    auto tpExp = readTiffParameters( files.front() );
    if ( !tpExp.has_value() )
        return unexpected( tpExp.error() );

    auto& tp = *tpExp;

    SimpleVolumeMinMax outVolume;
    outVolume.dims = { tp.imageSize.x, tp.imageSize.y, 1 };
    outVolume.min = FLT_MAX;
    outVolume.max = FLT_MIN;

    outVolume.voxelSize = settings.voxelSize;
    outVolume.data.resize( size_t( outVolume.dims.x ) * outVolume.dims.y );

    TiffParameters localParams;
    RawTiffOutput output;
    output.size = ( tp.imageSize.x * tp.imageSize.y ) * sizeof( float );
    output.params = &localParams;
    output.min = &outVolume.min;
    output.max = &outVolume.max;
    FloatGrid grid;
    for ( size_t layerIndex = 0; layerIndex < files.size(); ++layerIndex )
    {
        output.bytes = ( uint8_t* )( outVolume.data.data() );
        auto readRes = readRawTiff( files[layerIndex], output );

        if ( !readRes.has_value() )
            return unexpected( readRes.error() );

        if ( localParams != tp )
            return unexpected( "Inconsistent TIFF files" );

        if ( !grid )
            grid = simpleVolumeToDenseGrid( outVolume );
        else
            putSimpleVolumeInDenseGrid( grid, Vector3i{0, 0, ( int ) layerIndex}, outVolume );

        if ( settings.cb && !settings.cb( float( layerIndex ) / files.size() ) )
            return unexpected( "Loading was cancelled" );
    }

    if ( settings.cb && !settings.cb( 1.0f ) )
        return unexpected( "Loading was cancelled" );

    if ( !grid )
        return unexpected( "No voxel data" );

    VdbVolume res;

    res.data = std::move( grid );
    res.dims = outVolume.dims;
    res.dims.z = static_cast<int>( files.size() );
    res.voxelSize = outVolume.voxelSize;
    res.min = outVolume.min;
    res.max = outVolume.max;

    if ( settings.gridType == GridType::LevelSet )
    {
        res.data->setGridClass( openvdb::GridClass::GRID_LEVEL_SET );
        res.data->denseFill( openvdb::CoordBBox( 0, 0, 0, res.dims.x, res.dims.y, 0 ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( 0, 0, 0, res.dims.x, 0, res.dims.z ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( 0, 0, 0, 0, res.dims.y, res.dims.z ), res.max, false );

        res.data->denseFill( openvdb::CoordBBox( 0, 0, res.dims.z, res.dims.x, res.dims.y, res.dims.z ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( 0, res.dims.y, 0, res.dims.x, res.dims.y, res.dims.z ), res.max, false );
        res.data->denseFill( openvdb::CoordBBox( res.dims.x, 0, 0, res.dims.x, res.dims.y, res.dims.z ), res.max, false );
    }

    return res;
}
#endif // MRVOXELS_NO_TIFF

Expected<VdbVolume> fromRaw( const std::filesystem::path& file, const RawParameters& params,
    const ProgressCallback& cb )
{
    MR_TIMER
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );
    return addFileNameInError( fromRaw( in, params, cb ), file );
}

Expected<VdbVolume> fromRaw( std::istream& in, const RawParameters& params,  const ProgressCallback& cb )
{
    MR_TIMER
    if ( params.dimensions.x <= 0 || params.dimensions.y <= 0 || params.dimensions.z <= 0 )
        return unexpected( "Wrong volume dimension parameter value" );

    if ( params.voxelSize.x <= 0 || params.voxelSize.y <= 0 || params.voxelSize.z <= 0 )
        return unexpected( "Wrong voxel size parameter value" );

    int unitSize = 0;
    switch ( params.scalarType )
    {
        case ScalarType::UInt8:
            unitSize = 1;
            break;
        case ScalarType::Int8:
            unitSize = 1;
            break;
        case ScalarType::UInt16:
            unitSize = 2;
            break;
        case ScalarType::Int16:
            unitSize = 2;
            break;
        case ScalarType::UInt32:
            unitSize = 4;
            break;
        case ScalarType::Int32:
            unitSize = 4;
            break;
        case ScalarType::Float32:
            unitSize = 4;
        break;
        case ScalarType::UInt64:
            unitSize = 8;
            break;
        case ScalarType::Int64:
            unitSize = 8;
            break;
        case ScalarType::Float64:
            unitSize = 8;
            break;
        case ScalarType::Float32_4:
            unitSize = 16;
            break;
        default:
            assert( false );
            return unexpected( "Wrong scalar type parameter value" );
    }

    SimpleVolumeMinMax outVolume;
    outVolume.dims = params.dimensions;
    outVolume.voxelSize = params.voxelSize;
    outVolume.data.resize( size_t( outVolume.dims.x ) * outVolume.dims.y * outVolume.dims.z );
    char* outPointer{ nullptr };
    std::vector<char> data;
    if ( params.scalarType == ScalarType::Float32 )
        outPointer = ( char* )outVolume.data.data();
    else
    {
        data.resize( outVolume.data.size() * unitSize );
        outPointer = data.data();
    }

    size_t xyDimsUnit = size_t( params.dimensions.x ) * params.dimensions.y * unitSize;
    for ( int z = 0; z < params.dimensions.z; ++z )
    {
        size_t shift = xyDimsUnit * z;
        if ( !in.read( outPointer + shift, xyDimsUnit ) )
            return unexpected( "Read error" );
        if ( cb )
            cb( ( z + 1.0f ) / float( params.dimensions.z ) );
    }

    if ( params.scalarType != ScalarType::Float32 )
    {
        int64_t min = 0;
        uint64_t max = 0;
        if ( params.scalarType == ScalarType::Int8 )
        {
            min = std::numeric_limits<int8_t>::lowest();
            max = std::numeric_limits<int8_t>::max();
        }
        else if ( params.scalarType == ScalarType::Int16 )
        {
            min = std::numeric_limits<int16_t>::lowest();
            max = std::numeric_limits<int16_t>::max();
        }
        else if ( params.scalarType == ScalarType::Int32 )
        {
            min = std::numeric_limits<int32_t>::lowest();
            max = std::numeric_limits<int32_t>::max();
        }
        else if ( params.scalarType == ScalarType::Int64 )
        {
            min = std::numeric_limits<int64_t>::lowest();
            max = std::numeric_limits<int64_t>::max();
        }
        else if ( params.scalarType == ScalarType::UInt8 )
            max = std::numeric_limits<uint8_t>::max();
        else if ( params.scalarType == ScalarType::UInt16 )
            max = std::numeric_limits<uint16_t>::max();
        else if ( params.scalarType == ScalarType::UInt32 )
            max = std::numeric_limits<uint32_t>::max();
        else if ( params.scalarType == ScalarType::UInt64 )
            max = std::numeric_limits<uint64_t>::max();
        auto converter = getTypeConverter( params.scalarType, max - min, min );
        for ( size_t i = 0; i < outVolume.data.size(); ++i )
        {
            float value = converter( &outPointer[i * unitSize] );
            outVolume.data[i] = value;
            outVolume.max = std::max( outVolume.max, value );
            outVolume.min = std::min( outVolume.min, value );
        }
    }
    else
    {
        auto minmaxIt = std::minmax_element( outVolume.data.begin(), outVolume.data.end() );
        outVolume.min = *minmaxIt.first;
        outVolume.max = *minmaxIt.second;
    }

    VdbVolume res;
    res.data = simpleVolumeToDenseGrid( outVolume );
    if ( params.gridLevelSet )
    {
        openvdb::tools::changeBackground( res.data->tree(), outVolume.max );
        res.data->setGridClass( openvdb::GRID_LEVEL_SET );
    }
    res.dims = outVolume.dims;
    res.voxelSize = outVolume.voxelSize;
    res.min = outVolume.min;
    res.max = outVolume.max;
    return res;
}

} // namespace VoxelsLoad

Expected<std::vector<std::shared_ptr<ObjectVoxels>>> makeObjectVoxelsFromFile( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    MR_TIMER

    return VoxelsLoad::fromAnySupportedFormat( file, subprogress( callback, 0.f, 1.f / 3.f ) )
        .and_then( [&] ( auto&& volumes ) { return VoxelsLoad::toObjectVoxels( volumes, file, subprogress( callback, 1.f / 3.f, 1.f ) ); } );
}

Expected<LoadedObjects> makeObjectFromVoxelsFile( const std::filesystem::path& file, const ProgressCallback& callback )
{
    return makeObjectVoxelsFromFile( file, callback ).transform( VoxelsLoad::toObjects );
}

} // namespace MR
