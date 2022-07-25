#include "MRPointsSave.h"
#include "MRTimer.h"
#include "MRVector3.h"
#include "MRColor.h"
#include "MRStringConvert.h"
#include "OpenCTM/openctm.h"
#include "MRStreamOperators.h"
#include <fstream>

namespace MR
{

const size_t blockSize = size_t( 1 ) << 16;

namespace PointsSave
{
const IOFilters Filters =
{
    {"PLY (.ply)",        "*.ply"},
    {"CTM (.ctm)",        "*.ctm"},
    {"PTS (.pts)",        "*.pts"}
};

tl::expected<void, std::string> toPly( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors /*= nullptr*/, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPly( points, out, colors, callback );
}

tl::expected<void, std::string> toPly( const PointCloud& points, std::ostream& out, const Vector<Color, VertId>* colors /*= nullptr*/, ProgressCallback callback )
{
    MR_TIMER;

    size_t numVertices = points.points.size();

    out << "ply\nformat binary_little_endian 1.0\ncomment MeshInspector.com\n"
        "element vertex " << numVertices << "\nproperty float x\nproperty float y\nproperty float z\n";
    if ( colors )
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "end_header\n";

    if ( !colors )
    {
        // write vertices
        static_assert( sizeof( points.points.front() ) == 12, "wrong size of Vector3f" );
        int blockIndex = 0;
        const float sizeAll = float( points.points.size() * sizeof( Vector3f ) );
        for ( size_t max = points.points.size() * sizeof( Vector3f ) / blockSize; blockIndex < max; ++blockIndex )
        {
            out.write( ( const char* )( points.points.data() ) + blockIndex * blockSize, blockSize );
            if ( callback && !callback( blockIndex * blockSize / sizeAll ) )
                return tl::make_unexpected( std::string( "Saving canceled" ) );
        }
        const size_t remnant = points.points.size() * sizeof( Vector3f ) - blockIndex * blockSize;
        if ( remnant )
            out.write( ( const char* )( points.points.data() ) + blockIndex * blockSize, remnant );
    }
    else
    {
        // write triangles
#pragma pack(push, 1)
        struct PlyColoredVert
        {
            Vector3f p;
            unsigned char r = 0, g = 0, b = 0;
        };
#pragma pack(pop)
        static_assert( sizeof( PlyColoredVert ) == 15, "check your padding" );

        PlyColoredVert cVert;
        for ( int v = 0; v < numVertices; ++v )
        {
            cVert.p = points.points[VertId( v )];
            const auto& c = ( *colors )[VertId( v )];
            cVert.r = c.r; cVert.g = c.g; cVert.b = c.b;
            out.write( ( const char* )&cVert, 15 );
            if ( callback && !callback( float( v ) / numVertices ) )
                return tl::make_unexpected( std::string( "Saving canceled" ) );
        }
    }

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in PLY-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

tl::expected<void, std::string> toCtm( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors /*= nullptr */,
                                                  const CtmSavePointsOptions& options /*= {}*/, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toCtm( points, out, colors, options, callback );
}

tl::expected<void, std::string> toCtm( const PointCloud& points, std::ostream& out, const Vector<Color, VertId>* colors /*= nullptr */,
                                                  const CtmSavePointsOptions& options /*= {}*/, ProgressCallback callback )
{
    MR_TIMER;

    class ScopedCtmConext
    {
        CTMcontext context_ = ctmNewContext( CTM_EXPORT );
    public:
        ~ScopedCtmConext()
        {
            ctmFreeContext( context_ );
        }
        operator CTMcontext()
        {
            return context_;
        }
    } context;

    ctmFileComment( context, options.comment );
    ctmCompressionMethod( context, CTM_METHOD_MG1 );
    ctmCompressionLevel( context, options.compressionLevel );

    const CTMfloat* normalsPtr = points.normals.empty() ? nullptr : (const CTMfloat*) points.normals.data();
    CTMuint aVertexCount = CTMuint( points.points.size() );

    std::vector<CTMuint> aIndices{0,0,0};

    ctmDefineMesh( context,
        (const CTMfloat*) points.points.data(), aVertexCount,
        aIndices.data(), 1, normalsPtr );

    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format" );

    std::vector<Vector4f> colors4f; // should be alive when save is performed
    if ( colors && colors->size() == points.points.size() )
    {
        colors4f.resize( colors->size() );
        for ( int i = 0; i < colors4f.size(); ++i )
            colors4f[i] = Vector4f( ( *colors )[VertId{i}] );

        ctmAddAttribMap( context, (const CTMfloat*) colors4f.data(), "Color" );
    }

    if ( ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( "Error encoding in CTM-format colors" );

    ctmSaveCustom( context, []( const void* buf, CTMuint size, void* data )
    {
        std::ostream& s = *reinterpret_cast<std::ostream*>( data );
        s.write( (const char*) buf, size );
        return s.good() ? size : 0;
    }, &out );

    if ( !out || ctmGetError( context ) != CTM_NONE )
        return tl::make_unexpected( std::string( "Error saving in CTM-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

tl::expected<void, std::string> toPts( const PointCloud& points, const std::filesystem::path& file, ProgressCallback callback )
{
    std::ofstream out( file, std::ofstream::binary );
    if ( !out )
        return tl::make_unexpected( std::string( "Cannot open file for writing " ) + utf8string( file ) );

    return toPts( points, out, callback );
}

tl::expected<void, std::string> toPts( const PointCloud& points, std::ostream& out, ProgressCallback callback )
{
    out << "BEGIN_Polyline\n";
    const float pointsNum = float( points.validPoints.count() );
    int pointIndex = 0;
    for ( auto v : points.validPoints )
    {
        out << points.points[v] << "\n";
        if ( callback && !callback( float( pointIndex ) / pointsNum ) )
            return tl::make_unexpected( std::string( "Saving canceled" ) );
        ++pointIndex;
    }
    out << "END_Polyline\n";

    if ( !out )
        return tl::make_unexpected( std::string( "Error saving in PTS-format" ) );

    if ( callback )
        callback( 1.f );
    return {};
}

tl::expected<void, std::string> toAnySupportedFormat( const PointCloud& points, const std::filesystem::path& file, const Vector<Color, VertId>* colors /*= nullptr */,
                                                      ProgressCallback callback )
{
    auto ext = file.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == u8".ply" )
        res = MR::PointsSave::toPly( points, file, colors, callback );
    else if ( ext == u8".ctm" )
        res = MR::PointsSave::toCtm( points, file, colors, {}, callback );
    else if ( ext == u8".pts" )
        res = MR::PointsSave::toPts( points, file, callback );
    return res;
}
tl::expected<void, std::string> toAnySupportedFormat( const PointCloud& points, std::ostream& out, const std::string& extension, const Vector<Color, VertId>* colors /*= nullptr */,
                                                      ProgressCallback callback )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    tl::expected<void, std::string> res = tl::make_unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".ply" )
        res = MR::PointsSave::toPly( points, out, colors, callback );
    else if ( ext == ".ctm" )
        res = MR::PointsSave::toCtm( points, out, colors, {}, callback );
    else if ( ext == ".pts" )
        res = MR::PointsSave::toPts( points, out, callback );
    return res;
}

}
}
