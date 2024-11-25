#include "MRMeshLoadObj.h"
#include "MRAffineXf3.h"
#include "MRBitSetParallelFor.h"
#include "MRBuffer.h"
#include "MRComputeBoundingBox.h"
#include "MRIOFormatsRegistry.h"
#include "MRIOParsing.h"
#include "MRImageLoad.h"
#include "MRMeshBuilder.h"
#include "MRObjectMesh.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRphmap.h"
#include "MRString.h"
#include "MRPch/MRFmt.h"
#include "MRPch/MRTBB.h"

#include <boost/algorithm/string/trim.hpp>
#include <boost/spirit/home/x3.hpp>

#include <map>

namespace
{
    using namespace MR;

    enum class ObjElement
    {
        Unknown,
        Vertex,
        Face,
        Object,
        TextureVertex,
        MaterialLibrary,
        MaterialName,
    };

    enum class MtlElement
    {
        Unknown,
        MaterialName,
        AmbientColor,
        DiffuseColor,
        SpecularColor,
        SpecularExponent,
        Dissolve,
        IlluminationModel,
        AmbientTexture,
        DiffuseTexture,
        SpecularTexture,
        SpecularExponentTexture,
        DissolveTexture,
    };

    template <typename T>
    T parseToken( std::string_view line );

    template <>
    ObjElement parseToken<ObjElement>( std::string_view line )
    {
        while ( !line.empty() && line[0] == ' ' )
            line.remove_prefix( 1 );

        assert( !line.empty() );
        switch ( line[0] )
        {
        case 'v':
            assert( line.size() >= 2 );
            switch ( line[1] )
            {
            case ' ':
                return ObjElement::Vertex;
            case 't':
                return ObjElement::TextureVertex;
            default:
                return ObjElement::Unknown;
            }
        case 'f':
            return ObjElement::Face;
        case 'o':
            return ObjElement::Object;
        case 'u':
            if ( line.starts_with( "usemtl" ) )
                return ObjElement::MaterialName;
            return ObjElement::Unknown;
        case 'm':
            if ( line.starts_with( "mtllib" ) )
                return ObjElement::MaterialLibrary;
            return ObjElement::Unknown;
        default:
            return ObjElement::Unknown;
        }
    }

    template <>
    MtlElement parseToken<MtlElement>( std::string_view line )
    {
        while ( !line.empty() && line[0] == ' ' )
            line.remove_prefix( 1 );

        assert( !line.empty() );
        switch ( line[0] )
        {
        case 'n':
            if ( line.starts_with( "newmtl" ) )
                return MtlElement::MaterialName;
            else
                return MtlElement::Unknown;
        case 'K':
            assert( line.size() >= 2 );
            switch ( line[1] )
            {
            case 'a':
                return MtlElement::AmbientColor;
            case 'd':
                return MtlElement::DiffuseColor;
            case 's':
                return MtlElement::SpecularColor;
            default:
                return MtlElement::Unknown;
            }
        case 'm':
            if ( line.starts_with( "map_" ) )
            {
                assert( line.size() >= 5 );
                switch ( line[4] )
                {
                case 'K':
                    assert( line.size() >= 6 );
                    switch ( line[5] )
                    {
                    case 'a':
                        return MtlElement::AmbientTexture;
                    case 'd':
                        return MtlElement::DiffuseTexture;
                    case 's':
                        return MtlElement::SpecularTexture;
                    default:
                        return MtlElement::Unknown;
                    }
                default:
                    return MtlElement::Unknown;
                }
            }
            else
            {
                return MtlElement::Unknown;
            }
        default:
            return MtlElement::Unknown;
        }
    }

    template <typename Element>
    struct ElementGroup
    {
        Element element{ Element() };
        size_t begin{ 0 };
        size_t end{ 0 };
    };

    template <typename Element>
    std::vector<ElementGroup<Element>> groupLines( const char* data, size_t, const std::vector<size_t>& newlines )
    {
        const auto lineCount = newlines.size() - 1;

        std::vector<ElementGroup<Element>> groups{ { Element(), 0, 0 } }; // emplace stub initial group
        for ( size_t li = 0; li < lineCount; li++ )
        {
            std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
            const auto element = parseToken<Element>( line );
            if ( element != groups.back().element )
            {
                groups.back().end = li;
                groups.push_back( { element, li, 0 } );
            }
        }
        groups.back().end = lineCount;
        return groups;
    }

    Expected<void> parseObjTextureVertex( const std::string_view& str, UVCoord& vt )
    {
        using namespace boost::spirit::x3;

        std::array<float, 3> coords{ 0.f, 0.f, 0.f };
        int i = 0;
        auto coord = [&] ( auto& ctx ) { coords[i++] = _attr( ctx ); };

        bool r = phrase_parse(
                str.begin(),
                str.end(),
                ( lit( "vt" ) >> float_[coord] >> -( float_[coord] >> -( float_[coord] ) ) ),
                ascii::space
        );

        static constexpr int MaxErrorStringLen = 80;
        if ( !r )
            return unexpected( "Failed to parse vertex in OBJ-file: " + std::string( trimRight( str.substr( 0, MaxErrorStringLen ) ) ) );

        vt = { coords[0], coords[1] };
        return {};
    }

    struct ObjFace
    {
        std::vector<int> vertices;
        std::vector<int> textures;
        std::vector<int> normals;
    };

    Expected<void> parseObjFace( const std::string_view& str, ObjFace& f )
    {
        using namespace boost::spirit::x3;

        auto v = [&] ( auto& ctx ) { f.vertices.emplace_back( _attr( ctx ) ); };
        auto vt = [&] ( auto& ctx ) { f.textures.emplace_back( _attr( ctx ) ); };
        auto vn = [&] ( auto& ctx ) { f.normals.emplace_back( _attr( ctx ) ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            // NOTE: actions are not being reverted after backtracking
            // https://github.com/boostorg/spirit/issues/378
            ( 'f' >> *( int_[v] >> -( '/' >> ( ( int_[vt] >> -( '/' >> int_[vn] ) ) | ( '/' >> int_[vn] ) ) ) ) ),
            ascii::space
        );
        if ( !r )
            return unexpected( "Failed to parse face in OBJ-file" );

        if ( f.vertices.empty() )
            return unexpected( "Invalid face vertex count in OBJ-file" );
        if ( !f.textures.empty() && f.textures.size() != f.vertices.size() )
            return unexpected( "Invalid face texture count in OBJ-file" );
        if ( !f.normals.empty() && f.normals.size() != f.vertices.size() )
            return unexpected( "Invalid face normal count in OBJ-file" );
        return {};
    }

    Expected<void> parseMtlColor( const std::string_view& str, Vector3f& color )
    {
        using namespace boost::spirit::x3;

        auto r = [&] ( auto& ctx ) { color.x = _attr( ctx ); };
        auto g = [&] ( auto& ctx ) { color.y = _attr( ctx ); };
        auto b = [&] ( auto& ctx ) { color.z = _attr( ctx ); };

        bool res = phrase_parse(
            str.begin(),
            str.end(),
            (
                char_( 'K' ) >> ( char_( 'a' ) | char_( 'd' ) | char_( 's' ) )
                >> (
                        ( float_[r] >> float_[g] >> float_[b] ) |
                        ( lit( "spectral" ) >> no_skip[+char_] ) |
                        ( lit( "xyz" ) >> float_ >> float_ >> float_ )
                    )
            ),
            space
        );
        if ( !res )
            return unexpected( "Failed to parse color in MTL-file" );

        return {};
    }

    Expected<void> parseMtlTexture( const std::string_view& str, std::string& textureFile )
    {
        using namespace boost::spirit::x3;

        auto file = [&] ( auto& ctx ) { textureFile = _attr( ctx ); };

        bool r = phrase_parse(
            str.begin(),
            str.end(),
            (
                lit( "map_") >> ( lit( "Ka" ) | lit( "Kd" ) | lit( "Ks" ) | lit( "Ns" ) | lit( "d" ) )
                >> *(
                        ( lit( "-blendu" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-blendv" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-cc" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-clamp" ) >> ( lit( "on" ) | lit( "off" ) ) ) |
                        ( lit( "-imfchan" ) >> ( char_( 'r' ) | char_( 'b' ) | char_( 'g' ) | char_( 'm' ) | char_( 'l' ) | char_( 'z' ) ) ) |
                        ( lit( "-mm" ) >> float_ >> float_ ) |
                        ( lit( "-bm" ) >> float_ ) |
                        ( lit( "-o" ) >> float_ >> float_ >> float_ ) |
                        ( lit( "-s" ) >> float_ >> float_ >> float_ ) |
                        ( lit( "-t" ) >> float_ >> float_ >> float_ ) |
                        ( lit( "-texres" ) >> float_ )
                    )
                >> no_skip[+char_][file]
            ),
            space
        );
        if ( !r )
            return unexpected( "Failed to parse texture in MTL-file" );

        boost::trim( textureFile );
        return {};
    }

    struct MtlMaterial
    {
        Vector3f diffuseColor = Vector3f::diagonal( -1.0f );
        std::string diffuseTextureFile;
        TextureId id;
    };

    using MtlLibrary = HashMap<std::string, MtlMaterial>;

    Expected<MtlLibrary> loadMtlLibrary( const std::filesystem::path& path )
    {
        std::ifstream mtlIn( path, std::ios::binary );
        if ( !mtlIn.is_open() )
            return unexpected( "unable to open MTL file" );

        const auto mtlContent = readCharBuffer( mtlIn );
        if ( !mtlContent )
            return unexpected( "Unable to open MTL file: " + mtlContent.error() );
        const auto* data = mtlContent->data();
        const auto mtlSize = mtlContent->size();

        const auto newlines = splitByLines( data, mtlSize );

        const auto groups = groupLines<MtlElement>( data, mtlSize, newlines );

        MtlLibrary result;
        std::string currentMaterialName;
        MtlMaterial currentMaterial;

        for ( const auto& group : groups )
        {
            std::string parseError;
            switch ( group.element )
            {
            case MtlElement::Unknown:
                break;
            case MtlElement::MaterialName:
            {
                const auto li = group.end - 1;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );

                if ( !currentMaterialName.empty() )
                {
                    result.emplace( std::move( currentMaterialName ), std::move( currentMaterial ) );
                    currentMaterial = {};
                }
                currentMaterialName = line.substr( 6, std::string_view::npos );
                boost::trim( currentMaterialName );
            }
            break;
            case MtlElement::DiffuseColor:
            {
                const auto li = group.end - 1;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );

                auto res = parseMtlColor( line, currentMaterial.diffuseColor );
                if ( !res.has_value() )
                    parseError = std::move( res.error() );
            }
            break;
            case MtlElement::DiffuseTexture:
            {
                const auto li = group.end - 1;
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );

                auto res = parseMtlTexture( line, currentMaterial.diffuseTextureFile );
                if ( !res.has_value() )
                    parseError = std::move( res.error() );
            }
            break;
            default:
                break;
            }
            if ( !parseError.empty() )
                return unexpected( parseError );
        }

        if ( !currentMaterialName.empty() )
            result.emplace( std::move( currentMaterialName ), std::move( currentMaterial ) );
        return result;
    }
}

namespace MR
{

namespace MeshLoad
{

Expected<std::vector<NamedMesh>> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                const ObjLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, file.parent_path(), settings ), file );
}

Expected<std::vector<NamedMesh>> fromSceneObjFile( std::istream& in, bool combineAllObjects, const std::filesystem::path& dir,
                                                                const ObjLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    auto data = readCharBuffer( in );
    if ( !data.has_value() )
        return unexpected( data.error() );

    if ( !reportProgress(settings.callback, 0.25f) )
        return unexpected( "Loading canceled" );
    // TODO: redefine callback

    ObjLoadSettings newSettings = settings;
    newSettings.callback = subprogress( settings.callback, 0.25f, 1.f );

    return fromSceneObjFile( data->data(), data->size(), combineAllObjects, dir, newSettings );
}

Expected<std::vector<NamedMesh>> fromSceneObjFile( const char* data, size_t size, bool combineAllObjects, const std::filesystem::path& dir,
                                                                const ObjLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    Vector<Vector3d, VertId> points;
    std::vector<UVCoord> textureVertices;
    std::vector<int> texCoords;
    Triangulation triangulation;
    VertUVCoords uvCoords;
    VertColors colors;
    bool hasColors = true; //assume that colors are present unless we find they are not
    Expected<MtlLibrary> mtl;
    std::string currentMaterialName;

    TextureId maxTextureId = TextureId( -1 );
    TextureId currentTextureId = TextureId(-1);
    Vector<TextureId, FaceId> texturePerFace;

    std::map<int, int> additions;
    additions[0] = 0;
    int originalPointCount = 0;

    auto finishObject = [&]() -> Expected<void>
    {
        MR_NAMED_TIMER( "finish object" )
        if ( !triangulation.empty() )
        {
            auto& result = res.emplace_back();
            result.name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & vs : triangulation )
            {
                minV = std::min( { minV, vs[0], vs[1], vs[2] } );
                maxV = std::max( { maxV, vs[0], vs[1], vs[2] } );
            }
            if ( maxV >= points.endId() )
                return unexpected( "vertex id is larger than total point coordinates" );
            for ( auto & vs : triangulation )
            {
                for ( int i = 0; i < 3; ++i )
                    vs[i] -= minV;
            }

            std::vector<MeshBuilder::VertDuplication> dups;
            if ( settings.customXf )
            {
                const auto box = computeBoundingBox( points, minV, maxV + 1 );
                // shift the reference frame to the center of bounding box for best relative precision of mesh point coordinates (they are stored as 32-bit floats),
                // with the exception when boundary box already contains the origin point (0,0,0)
                Vector3d pointOffset = box.contains( Vector3d{} ) ? Vector3d{} : box.center();
                result.xf = AffineXf3f::translation( Vector3f( pointOffset ) );
                VertCoords floatPoints;
                floatPoints.reserve( maxV - minV + 1 );
                for ( auto v = minV; v <= maxV; ++v )
                    floatPoints.emplace_back( points[v] - pointOffset );
                assert( floatPoints.size() == maxV - minV + 1 );
                result.mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                    std::move( floatPoints ), triangulation, &dups,
                    { .skippedFaceCount = settings.countSkippedFaces ? &result.skippedFaceCount : nullptr } );
            }
            else
            {
                result.mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                    VertCoords( begin( points ) + minV, begin( points ) + maxV + 1 ), triangulation, &dups,
                    { .skippedFaceCount = settings.countSkippedFaces ? &result.skippedFaceCount : nullptr } );
            }
            if ( !colors.empty() )
            {
                colors.resize( result.mesh.points.size() );
                for ( const auto& [src, dup] : dups )
                    colors[dup] = colors[src];

                result.colors = std::move( colors );
                colors = {};
            }
            hasColors = true; //assume that colors for next object are present unless we find they are not
            result.duplicatedVertexCount = int( dups.size() );
            triangulation.clear();

            VertHashMap dst2Src;
            dst2Src.reserve( dups.size() );
            for ( const auto& dup : dups )
                dst2Src.emplace( dup.dupVert, dup.srcVert );

            if ( mtl.has_value() && ! mtl->empty() )
            {
                auto materialIt = mtl->find( currentMaterialName );

                if ( materialIt == mtl->end() )
                {
                    materialIt = mtl->find( "default" );
                }
                if ( materialIt == mtl->end() )
                {
                    materialIt = mtl->begin();
                }
                
                if ( maxTextureId.valid() )
                {
                    result.textureFiles.resize( maxTextureId + 1 );
                    for ( const auto& [mtlName, material] : *mtl )
                    {
                        if ( !material.diffuseTextureFile.empty() )
                            result.textureFiles[material.id] = dir / material.diffuseTextureFile;
                    }
                }

                if ( materialIt->second.diffuseColor != Vector3f::diagonal(-1) )
                    result.diffuseColor = Color( materialIt->second.diffuseColor );

                result.texturePerFace = std::move( texturePerFace );

                if ( !texCoords.empty() )
                {
                    uvCoords.resize( points.size() );
                    tbb::parallel_for( tbb::blocked_range<VertId>( VertId( 0 ), VertId( points.size() ) ), [&] ( const tbb::blocked_range<VertId>& range )
                    {
                        for ( VertId i = range.begin(); i < range.end(); ++i )
                        {
                            if ( texCoords[i] < uvCoords.size() )
                                uvCoords[i] = textureVertices[texCoords[i]];
                        }
                    } );

                    assert( uvCoords.size() >= maxV );
                    result.uvCoords = { begin( uvCoords ) + minV, begin( uvCoords ) + maxV + 1 };
                    result.uvCoords.resize( result.mesh.points.size() );
                    for ( const auto& dup : dups )
                        result.uvCoords[dup.dupVert] = result.uvCoords[dup.srcVert];
                }
            }
        }
        currentObjName.clear();
        return {};
    };

    Timer timer( "split by lines" );
    const auto newlines = splitByLines( data, size );
    const auto lineCount = newlines.size() - 1;

    if ( !reportProgress( settings.callback, 0.4f ) )
        return unexpected( "Loading canceled" );

    timer.restart( "group element lines" );
    const auto groups = groupLines<ObjElement>( data, size, newlines );

    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpected( "Loading canceled" );

    auto parseVertices = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        assert ( end > begin );
        if ( hasColors && colors.empty() ) // check the presence of colors only once per object (parseVertices can be called many times)
        {
            // detect presence of colors from the first vertex
            Vector3d v;
            constexpr Vector3d cInvalidColor = { -1., -1., -1. };
            Vector3d c { cInvalidColor };
            std::string_view line( data + newlines[begin], newlines[begin + 1] - newlines[begin] );
            auto res = parseObjCoordinate( line, v, &c );
            if ( !res.has_value() )
            {
                parseError = std::move( res.error() );
                return;
            }
            hasColors = c != cInvalidColor;
        }

        const auto offset = points.endId();
        originalPointCount += int( end - begin );
        const size_t newSize = points.size() + ( end - begin );
        texCoords.resize( newSize, -1 );

        points.resize( newSize );
        if ( hasColors )
            colors.resize( newSize );
        uvCoords.resize( newSize );

        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            Vector3d v { noInit };
            Vector3d c { noInit };
            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjCoordinate( line, v, hasColors ? &c : nullptr );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                const auto n = offset + ( li - begin );
                points[n] = v;
                if ( hasColors )
                    colors[VertId( n )] = Color( c );
            }
        }, ctx );
    };

    auto parseTextureVertices = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        const auto offset = textureVertices.size();
        textureVertices.resize( textureVertices.size() + ( end - begin ) );

        tbb::task_group_context ctx;
        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            UVCoord vt;
            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjTextureVertex( line, vt );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }
                textureVertices[offset + ( li - begin )] = vt;
            }
        }, ctx );
    };

    auto parseFaces = [&] ( size_t begin, size_t end, std::string& parseError )
    {
        struct OrderedTriangulation
        {
            size_t offset;
            Triangulation t;
        };
        tbb::task_group_context ctx;
        tbb::enumerable_thread_specific<std::vector<OrderedTriangulation>> ordTrisPerThread;

        std::mutex mutex;

        int newPoints = 0;
        int lastAddition = 0;
        if ( !additions.empty() )
            lastAddition = additions.rbegin()->second;

        tbb::parallel_for( tbb::blocked_range<size_t>( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
        {
            auto& ordTris = ordTrisPerThread.local();
            auto& t = ordTris.emplace_back( OrderedTriangulation { range.begin(), {} } ).t;

            ObjFace f;
            // usually a face has 3 or 4 vertices
            for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
               elements->reserve( 4 );

            for ( auto li = range.begin(); li < range.end(); li++ )
            {
                for ( auto* elements : { &f.vertices, &f.textures, &f.normals } )
                    elements->clear();

                std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
                auto res = parseObjFace( line, f );
                if ( !res.has_value() )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = std::move( res.error() );
                    return;
                }

                auto& vs = f.vertices;
                for ( auto& v : vs )
                {
                    if ( --v < 0 )
                    {
                        v += originalPointCount + 1;

                        if ( v < 0 )
                        {
                            if ( ctx.cancel_group_execution() )
                                parseError = "Too negative vertex ID in OBJ-file";
                            return;
                        }
                    }
                   
                    auto it = additions.upper_bound( v );
                    int toAdd = 0;
                    if ( it == additions.end() )
                        toAdd = additions.rbegin()->second;
                    else if ( it != additions.begin() )
                        toAdd = (--it)->second;

                    if ( toAdd > 0 )
                        v += toAdd;
                   
                }
                if ( vs.size() < 3 )
                {
                    if ( ctx.cancel_group_execution() )
                        parseError = "Face with less than 3 vertices in OBJ-file";
                    return;
                }
                
                auto& vts = f.textures;
                if ( !vts.empty() )
                {
                    for ( auto& vt : vts )
                    {
                        if ( vt < 0 )
                            vt += int( textureVertices.size() ) + 1;

                        if ( --vt < 0 )
                        {
                            if ( ctx.cancel_group_execution() )
                                parseError = "Too negative texture vertex ID in OBJ-file";
                            return;
                        }
                    }

                    assert( vts.size() == vs.size() );
                    const std::lock_guard<std::mutex> lock( mutex );
                    for ( int j = 0; j < vs.size(); ++j )
                    {
                        if ( texCoords[vs[j]] == vts[j] )
                            continue;

                        if ( texCoords[vs[j]] < 0 )
                        {
                            texCoords[vs[j]] = vts[j];
                            continue;
                        }
                        points.push_back( points[ VertId( vs[j] ) ] );
                        texCoords.push_back( vts[j] );
                        vs[j] = int( points.size() ) - 1;
                        ++newPoints;
                    }
                }

                // TODO: make smarter triangulation based on point coordinates
                for ( int j = 1; j + 1 < vs.size(); ++j )
                    t.push_back( { VertId( vs[0] ), VertId( vs[j] ), VertId( vs[j+1] ) } );
            }
        }, ctx );

        additions.insert_or_assign( originalPointCount, lastAddition + newPoints );

        if ( !parseError.empty() )
            return;

        size_t trisSize = 0;
        std::vector<OrderedTriangulation> ordTrisAll;
        for ( auto& ordTris : ordTrisPerThread )
        {
            for ( auto& ordTri : ordTris )
            {
                trisSize += ordTri.t.size();
                ordTrisAll.emplace_back( std::move( ordTri ) );
            }
        }
        std::sort( ordTrisAll.begin(), ordTrisAll.end(),
                   [] ( auto&& a, auto&& b ) { return a.offset < b.offset; } );

        triangulation.reserve( triangulation.size() + trisSize );
        bool hasMaterial = mtl.has_value() && mtl.value().size() > 0;
        if ( hasMaterial )
            texturePerFace.reserve( triangulation.size() + trisSize );
        for ( auto& ordTri : ordTrisAll )
        {
            triangulation.vec_.insert( triangulation.vec_.end(), ordTri.t.vec_.begin(), ordTri.t.vec_.end() );
            if ( hasMaterial )
                texturePerFace.vec_.insert( texturePerFace.vec_.end(), ordTri.t.vec_.size(), currentTextureId );
        }
    };

    auto parseObject = [&] ( size_t, size_t end, std::string& ) -> Expected<void>
    {
        if ( combineAllObjects )
            return {};

        // finish previous object
        return finishObject().and_then( [&]() -> Expected<void>
        {
            const auto li = end - 1;
            std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
            currentObjName = line.substr( strlen( "o" ), std::string_view::npos );
            boost::trim( currentObjName );
            return {};
        } );
    };

    auto parseMaterialLibrary = [&] ( size_t, size_t end, std::string& )
    {
        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        // TODO: support multiple files
        std::string filename ( line.substr( strlen( "mtllib" ), std::string_view::npos ) );
        boost::trim( filename );
        mtl = loadMtlLibrary( dir / filename );
    };

    auto parseMaterialName = [&] ( size_t, size_t end, std::string& )
    {
        const auto li = end - 1;
        std::string_view line( data + newlines[li], newlines[li + 1] - newlines[li + 0] );
        currentMaterialName = line.substr( strlen( "usemtl" ), std::string_view::npos );
        boost::trim( currentMaterialName );
        if ( mtl.has_value() )
        {
            auto it = mtl->find( currentMaterialName );
            if ( it != mtl->end() )
            {
                if ( !it->second.id )
                    it->second.id = ++maxTextureId;
                currentTextureId = it->second.id;
            }
        }
    };

    timer.restart( "parse groups" );
    for ( const auto& group : groups )
    {
        std::string parseError;
        switch ( group.element )
        {
        case ObjElement::Unknown:
            break;
        case ObjElement::Vertex:
            parseVertices( group.begin, group.end, parseError );
            break;
        case ObjElement::Face:
            parseFaces( group.begin, group.end, parseError );
            break;
        case ObjElement::Object:
            if ( auto exp = parseObject( group.begin, group.end, parseError ); !exp )
                return unexpected( std::move( exp.error() ) );
            break;
        case ObjElement::TextureVertex:
            parseTextureVertices( group.begin, group.end, parseError );
            break;
        case ObjElement::MaterialLibrary:
            parseMaterialLibrary( group.begin, group.end, parseError );
            break;
        case ObjElement::MaterialName:
            parseMaterialName( group.begin, group.end, parseError );
            break;
        }
        if ( !parseError.empty() )
            return unexpected( parseError );

        if ( !reportProgress( subprogress( settings.callback, 0.5f, 1.f ), ( float )group.end / ( float )lineCount ) )
            return unexpected( "Loading canceled" );
    }

    if ( auto exp = finishObject(); !exp )
        return unexpected( std::move( exp.error() ) );

    return res;
}

Expected<LoadedObjects> loadObjectFromObj( const std::filesystem::path& file, const ProgressCallback& cb )
{
    return fromSceneObjFile( file, false, { .customXf = true, .countSkippedFaces = true, .callback = cb } )
    .transform( [&] ( std::vector<NamedMesh>&& results )
    {
        int totalSkippedFaceCount = 0;
        int totalDuplicatedVertexCount = 0;
        int holesCount = 0;
        LoadedObjects res;
        res.objs.resize( results.size() );
        for ( int i = 0; i < res.objs.size(); ++i )
        {
            auto& result = results[i];

            std::shared_ptr<ObjectMesh> objectMesh = std::make_shared<ObjectMesh>();
            if ( result.name.empty() )
                objectMesh->setName( utf8string( file.stem() ) );
            else
                objectMesh->setName( std::move( result.name ) );
            objectMesh->select( true );
            objectMesh->setMesh( std::make_shared<Mesh>( std::move( result.mesh ) ) );
            if ( result.diffuseColor )
                objectMesh->setFrontColor( *result.diffuseColor, false );

            objectMesh->setUVCoords( std::move( result.uvCoords ) );

            int numEmptyTexture = 0;
            for ( const auto& p : result.textureFiles )
            {
                if ( p.empty() )
                    numEmptyTexture++;
            }

            if ( numEmptyTexture != 0 && numEmptyTexture != result.textureFiles.size() )
            {
                res.warnings += " object has material with and without texture";
            }
            else if ( numEmptyTexture == 0 && result.textureFiles.size() != 0 )
            {
                bool crashTextureLoad = false;
                for ( const auto& p : result.textureFiles )
                {
                    auto image = ImageLoad::fromAnySupportedFormat( p );
                    if ( image.has_value() )
                    {
                        MeshTexture meshTexture;
                        meshTexture.resolution = std::move( image.value().resolution );
                        meshTexture.pixels = std::move( image.value().pixels );
                        meshTexture.filter = FilterType::Linear;
                        meshTexture.wrap = WrapType::Clamp;
                        objectMesh->addTexture( std::move( meshTexture ) );
                    }
                    else
                    {
                        crashTextureLoad = true;
                        objectMesh->setTextures( {} );
                        res.warnings += image.error();
                        break;
                    }
                }
                if ( !crashTextureLoad )
                {
                    objectMesh->setVisualizeProperty( true, MeshVisualizePropertyType::Texture, ViewportMask::all() );
                    objectMesh->setTexturePerFace( std::move( result.texturePerFace ) );
                }
            }

            if ( !result.colors.empty() )
            {
                objectMesh->setVertsColorMap( std::move( result.colors ) );
                objectMesh->setColoringType( ColoringType::VertsColorMap );
            }

            objectMesh->setXf( result.xf );

            res.objs[i] = std::dynamic_pointer_cast< Object >( objectMesh );

            holesCount += int( objectMesh->numHoles() );

            totalSkippedFaceCount += result.skippedFaceCount;
            totalDuplicatedVertexCount += result.duplicatedVertexCount;
        }

        if ( totalSkippedFaceCount )
        {
            if ( !res.warnings.empty() )
                res.warnings += '\n';
            res.warnings = fmt::format( "{} triangles were skipped as inconsistent with others.", totalSkippedFaceCount );
        }
        if ( totalDuplicatedVertexCount )
        {
            if ( !res.warnings.empty() )
                res.warnings += '\n';
            res.warnings += fmt::format( "{} vertices were duplicated to make them manifold.", totalDuplicatedVertexCount );
        }
        if ( holesCount )
        {
            if ( !res.warnings.empty() )
                res.warnings += '\n';
            res.warnings += fmt::format( "The objects contain {} holes. Please consider using Fill Holes tool.", holesCount );
        }
        return res;
    } );
}

MR_ADD_OBJECT_LOADER( IOFilter( "3D model object (.obj)", "*.obj" ), loadObjectFromObj )

} //namespace MeshLoad

} //namespace MR
