#include "MRMeshLoadObj.h"
#include "MRStringConvert.h"
#include "MRMeshBuilder.h"
#include "MRTimer.h"
#include "MRBuffer.h"

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/qi.hpp>

namespace
{
    using namespace MR;

    tl::expected<Vector3f, std::string> parse_obj_vertex( const std::string_view& str )
    {
        namespace qi = boost::spirit::qi;
        namespace ascii = boost::spirit::ascii;

        using boost::phoenix::ref;

        Vector3f v;
        bool r = qi::phrase_parse(
            str.begin(),
            str.end(),
            ( 'v' >> qi::float_[ref( v.x ) = qi::_1] >> qi::float_[ref( v.y ) = qi::_1] >> qi::float_[ref( v.z ) = qi::_1] ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse vertex" );
        return v;
    }

    struct index_accumulator
    {
        std::vector<int> indices;

        index_accumulator& operator +=( int v )
        {
            indices.emplace_back( v );
            return *this;
        }
    };

    struct obj_face_indices
    {
        index_accumulator vertices;
        index_accumulator textures;
        index_accumulator normals;
    };

    tl::expected<obj_face_indices, std::string> parse_obj_face( const std::string_view& str )
    {
        namespace qi = boost::spirit::qi;
        namespace ascii = boost::spirit::ascii;

        using boost::phoenix::ref;

        obj_face_indices vs;
        for ( auto ia : { &vs.vertices, &vs.textures, &vs.normals } )
            ia->indices.reserve( 4 );
        bool r = qi::phrase_parse(
            str.begin(),
            str.end(),
            (
                'f' >>
                *( qi::int_[ref( vs.vertices ) += qi::_1]
                    >> -(
                        ( '/' >> qi::int_[ref( vs.textures ) += qi::_1] )
                        |
                        ( '/' >> qi::int_[ref( vs.textures ) += qi::_1] >> '/' >> qi::int_[ref( vs.normals ) += qi::_1] )
                        |
                        ( "//" >> qi::int_[ref( vs.normals ) += qi::_1] )
                    )
                )
            ),
            ascii::space
        );
        if ( !r )
            return tl::make_unexpected( "Failed to parse face" );
        // TODO: checks
        return vs;
    }
}

namespace MR
{

namespace MeshLoad
{

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( const std::filesystem::path& file, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    std::ifstream in( file );
    if ( !in )
        return tl::make_unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromSceneObjFile( in, combineAllObjects, callback ), file );
}

tl::expected<std::vector<NamedMesh>, std::string> fromSceneObjFile( std::istream& in, bool combineAllObjects,
                                                                    ProgressCallback callback )
{
    MR_TIMER

    std::vector<NamedMesh> res;
    std::string currentObjName;
    std::vector<Vector3f> points;
    Triangulation t;

    auto finishObject = [&]() 
    {
        MR_NAMED_TIMER( "finish object" )
        if ( !t.empty() )
        {
            res.emplace_back();
            res.back().name = std::move( currentObjName );

            // copy only minimal span of vertices for this object
            VertId minV(INT_MAX), maxV(-1);
            for ( const auto & vs : t )
            {
                minV = std::min( { minV, vs[0], vs[1], vs[2] } );
                maxV = std::max( { maxV, vs[0], vs[1], vs[2] } );
            }
            for ( auto & vs : t )
            {
                for ( int i = 0; i < 3; ++i )
                    vs[i] -= minV;
            }

            res.back().mesh = Mesh::fromTrianglesDuplicatingNonManifoldVertices(
                VertCoords( points.begin() + minV, points.begin() + maxV + 1 ), t );
            t.clear();
        }
        currentObjName.clear();
    };

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const auto streamSize = posEnd - posStart;

    Buffer<char> data;
    {
        MR_NAMED_TIMER( "resize buffer" )
        data.resize( streamSize + 1 );
    }
    {
        MR_NAMED_TIMER( "read data" )
        in.read( data.data(), streamSize );
    }
    if ( !in )
        return tl::make_unexpected( std::string( "OBJ-format read error" ) );
    if ( data[streamSize - 1] != '\n' )
        data[streamSize] = '\n';
    else
        data.resize( streamSize );

    std::vector<size_t> newlines{ 0 };
    for ( size_t i = 0; i < data.size(); i++ )
        if ( data[i] == '\n' )
            newlines.emplace_back( i + 1 );

    std::vector<int> vs;
    for ( int i = 0; i + 1 < newlines.size(); ++i )
    {
        std::string_view line( data.data() + newlines[i], newlines[i + 1] - newlines[i + 0] );
        if ( line[0] == 'v' && line[1] != 'n' /*normals*/ && line[1] != 't' /*texture coordinates*/ )
        {
            auto v = parse_obj_vertex( line );
            if ( !v.has_value() )
                return tl::make_unexpected( v.error() );
            points.emplace_back( *v );
        }
        else if ( line[0] == 'f' )
        {
            auto is = parse_obj_face( line );
            if ( !is.has_value() )
                return tl::make_unexpected( is.error() );

            auto vs = std::move( is->vertices.indices );
            for ( auto& v : vs )
            {
                if ( v < 0 )
                {
                    v += (int)points.size() + 1;
                    if ( v <= 0 )
                        return tl::make_unexpected( std::string( "Too negative vertex ID in OBJ-file" ) );
                }
            }
            if ( vs.size() < 3 )
                return tl::make_unexpected( std::string( "Face with less than 3 vertices in OBJ-file" ) );

            // TODO: make smarter triangulation based on point coordinates
            for ( int j = 1; j + 1 < vs.size(); ++j )
                t.push_back( { VertId( vs[0]-1 ), VertId( vs[j]-1 ), VertId( vs[j+1]-1 ) } );
        }
        else if ( line[0] == 'o' )
        {
            if ( !combineAllObjects )
                finishObject();
            // next object
            currentObjName = line.substr( 1, std::string::npos );
            while ( !currentObjName.empty() && currentObjName[0] == ' ' )
                currentObjName.erase( currentObjName.begin() );
            {

            }
        }
        else
        {
            // skip unknown line
        }
        if ( callback && !(i & 0x3FF) )
        {
            const float progress = float( i + 1 ) / float( newlines.size() );
            if ( !callback( progress ) )
                return tl::make_unexpected( std::string( "Loading canceled" ));
        }
    }

    finishObject();
    return res;
}

} //namespace MeshLoad

} //namespace MR
