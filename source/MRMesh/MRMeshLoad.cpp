#include "MRMeshLoad.h"
#include "MRMeshBuilder.h"
#include "MRIdentifyVertices.h"
#include "MRMesh.h"
#include "MRphmap.h"
#include "MRTimer.h"
#include "miniply.h"
#include "MRIOFormatsRegistry.h"
#include "MRStringConvert.h"
#include "MRMeshLoadObj.h"
#include "MRColor.h"
#include "MRPch/MRTBB.h"
#include "MRProgressReadWrite.h"

#include <array>
#include <future>

#ifndef MRMESH_NO_OPENCTM
#include "OpenCTM/openctm.h"
#endif

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
#include <tinyxml2.h>
#endif

namespace MR
{

namespace MeshLoad
{

Expected<Mesh, std::string> fromMrmesh( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromMrmesh( in, settings ), file );
}

Expected<Mesh, std::string> fromMrmesh( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    Mesh mesh;
    auto readRes = mesh.topology.read( in, subprogress( settings.callback, 0.f, 0.5f) );
    if ( !readRes.has_value() )
    {
        std::string error = readRes.error();
        if ( error != "Loading canceled" )
            error = "Error reading topology from mrmesh - file:\n" + error;
        return unexpected( error );
    }

    // read points
    std::uint32_t numPoints;
    in.read( (char*)&numPoints, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of points from mrmesh-file" ) );
    mesh.points.resize( numPoints );
    if ( !readByBlocks( in, ( char* )mesh.points.data(), mesh.points.size() * sizeof( Vector3f ), subprogress( settings.callback, 0.5f, 1.f ) ) )
        return unexpected( std::string( "Loading canceled" ) );

    if ( !in )
        return unexpected( std::string( "Error reading  points from mrmesh-file" ) );

    return std::move( mesh );
}

Expected<Mesh, std::string> fromOff( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromOff( in, settings ), file );
}

Expected<Mesh, std::string> fromOff( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER
    std::string header;
    in >> header;
    if ( !in || header != "OFF" )
        return unexpected( std::string( "File is not in OFF-format" ) );

    int numPoints, numPolygons, numUnused;
    in >> numPoints >> numPolygons >> numUnused;
    if ( !in || numPoints <= 0 || numPolygons <= 0 || numUnused != 0 )
        return unexpected( std::string( "Unsupported OFF-format" ) );

    std::vector<Vector3f> points;
    points.reserve( numPoints );

    for ( int i = 0; i < numPoints; ++i )
    {
        float x, y, z;
        in >> x >> y >> z;
        if ( !in )
            return unexpected( std::string( "Points read error" ) );
        points.emplace_back( x, y, z );
        if ( settings.callback && !( i & 0x3FF ) && !subprogress( settings.callback, 0.f, 0.5f )( float( i ) / numPoints ) )
            return unexpected( std::string( "Loading canceled" ) );
    }

    Triangulation t;
    t.reserve( numPolygons );

    for ( int i = 0; i < numPolygons; ++i )
    {
        int k, a, b, c;
        in >> k >> a >> b >> c;
        if ( !in || k != 3 )
            return unexpected( std::string( "Polygons read error" ) );
        t.push_back( { VertId( a ), VertId( b ), VertId( c ) } );
        if ( settings.callback && !( i & 0x3FF ) && !subprogress( settings.callback, 0.5f, 1.f )( float( i ) / numPolygons ) )
            return unexpected( std::string( "Loading canceled" ) );
    }

    return Mesh::fromTriangles( std::move( points ), t );
}

Expected<Mesh, std::string> fromObj( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ios::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromObj( in, settings ), file );
}

Expected<Mesh, std::string> fromObj( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    auto objs = fromSceneObjFile( in, true, {}, settings.callback );
    if ( !objs.has_value() )
        return unexpected( objs.error() );
    if ( objs->size() != 1 )
        return unexpected( "OBJ-file is empty" );

    return std::move( (*objs)[0].mesh );
}

Expected<MR::Mesh, std::string> fromAnyStl( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromAnyStl( in, settings ), file );
}

Expected<MR::Mesh, std::string> fromAnyStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    auto pos = in.tellg();
    auto resBin = fromBinaryStl( in, settings );
    if ( resBin.has_value() || resBin.error() == "Loading canceled" )
        return resBin;
    in.clear();
    in.seekg( pos );
    auto resAsc = fromASCIIStl( in, settings );
    if ( resAsc.has_value() )
        return resAsc;
    return unexpected( resBin.error() + '\n' + resAsc.error() );
}

Expected<Mesh, std::string> fromBinaryStl( const std::filesystem::path & file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromBinaryStl( in, settings ), file );
}

Expected<Mesh, std::string> fromBinaryStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    char header[80];
    in.read( header, 80 );

    std::uint32_t numTris;
    in.read( (char*)&numTris, 4 );
    if ( !in )
        return unexpected( std::string( "Error reading the number of triangles from STL-file" ) );

    auto posCur = in.tellg();
    in.seekg( 0, std::ios_base::end );
    auto posEnd = in.tellg();
    in.seekg( posCur );
    if ( posEnd - posCur < 50 * std::istream::pos_type( numTris ) )
        return unexpected( std::string( "Binary STL-file is too short" ) );

    MeshBuilder::VertexIdentifier vi;
    vi.reserve( numTris );

    #pragma pack(push, 1)
    struct StlTriangle
    {
        Vector3f normal;
        Vector3f vert[3];
        std::uint16_t attr;
    };
    #pragma pack(pop)
    static_assert( sizeof( StlTriangle ) == 50, "check your padding" );

    const auto itemsInBuffer = std::min( numTris, 32768u );
    std::vector<StlTriangle> buffer( itemsInBuffer ), nextBuffer( itemsInBuffer );
    std::vector<Triangle3f> chunk( itemsInBuffer );

    // first chunk
    in.read( (char*)buffer.data(), sizeof(StlTriangle) * itemsInBuffer );
    if ( !in  )
        return unexpected( std::string( "Binary STL read error" ) );

    size_t readBytes = 0;
    const float streamSize = float( posEnd - posCur );

    for ( ;; )
    {
        tbb::task_group taskGroup;
        bool hasTask = false;
        if ( vi.numTris() + buffer.size() < numTris )
        {
            const auto itemsInNextChuck = std::min( numTris - (std::uint32_t)( vi.numTris() + buffer.size() ), itemsInBuffer );
            nextBuffer.resize( itemsInNextChuck );
            hasTask = true;
            const size_t size = sizeof( StlTriangle ) * nextBuffer.size();
            taskGroup.run( [&in, &nextBuffer, size] ()
            {
                in.read( ( char* )nextBuffer.data(), size );
            } );
            readBytes += size;
        }

        chunk.resize( buffer.size() );
        for ( int i = 0; i < buffer.size(); ++i )
            for ( int j = 0; j < 3; ++j )
                chunk[i][j] = buffer[i].vert[j];
        vi.addTriangles( chunk );

        if ( !hasTask )
            break;
        taskGroup.wait();
        if ( reportProgress( settings.callback , readBytes / streamSize ) )
            return unexpected( std::string( "Loading canceled" ) );
        if ( !in  )
            return unexpected( std::string( "Binary STL read error" ) );
        buffer.swap( nextBuffer );
    }

//     #pragma warning(disable: 4244)
//     std::cout <<
//         "tris = " << numTris << "\n"
//         "verts = " << hmap.size() << "\n"
//         "bucket_count = " << hmap.bucket_count() << "\n"
//         "subcnt = " << hmap.subcnt() << "\n"
//         "load_factor = " << hmap.load_factor() << "\n"
//         "max_load_factor = " << hmap.max_load_factor() << "\n";

    auto t = vi.takeTriangulation();
    if ( settings.duplicatedVertexCount )
    {
        std::vector<MeshBuilder::VertDuplication> dups;
        const auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( vi.takePoints(), t, &dups );
        *settings.duplicatedVertexCount = int( dups.size() );
        return res;
    }
    else
        return Mesh::fromTrianglesDuplicatingNonManifoldVertices( vi.takePoints(), t );
}

Expected<Mesh, std::string> fromASCIIStl( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromASCIIStl( in, settings ), file );
}

Expected<Mesh, std::string> fromASCIIStl( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER;

    using HMap = ParallelHashMap<Vector3f, VertId>;
    HMap hmap;
    VertCoords points;
    Triangulation t;

    std::string line;
    std::string prefix;
    Vector3f point;
    ThreeVertIds currTri;
    int triPos = 0;
    bool solidFound = false;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );

    for ( int i = 0; std::getline( in, line ); ++i )
    {
        std::istringstream iss( line );
        if ( !( iss >> prefix ) )
            break;

        if ( !solidFound )
        {
            if ( prefix == "solid" )
                solidFound = true;
            else
                break;
        }

        if ( prefix == "outer" )
        {
            triPos = 0;
            continue;
        }
        if ( prefix == "vertex" )
        {
            double x, y, z; // double is used to correctly open coordinates like 1e-55 which are under of float-precision
            if ( !( iss >> x >> y >> z ) )
                break;
            point = Vector3f{ Vector3d{ x, y, z } };

            VertId& id = hmap[point];
            if ( !id.valid() )
            {
                id = VertId( points.size() );
                points.push_back( point );
            }
            currTri[triPos] = id;
            ++triPos;
            continue;
        }
        if ( prefix == "endloop" )
        {
            t.push_back( currTri );
            continue;
        }
        if ( settings.callback && !( i & 0x3FF ) )
        {
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( !settings.callback( progress ) )
                return unexpected( std::string( "Loading canceled" ) );
        }
    }

    if ( !solidFound )
        return unexpected( std::string( "Failed to find 'solid' prefix in ascii STL" ) );


    if ( settings.duplicatedVertexCount )
    {
        std::vector<MeshBuilder::VertDuplication> dups;
        const auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( points ), t, &dups );
        *settings.duplicatedVertexCount = int( dups.size() );
        return res;
    }
    return Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( points ), t );
}

Expected<Mesh, std::string> fromPly( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromPly( in, settings ), file );
}

Expected<Mesh, std::string> fromPly( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    miniply::PLYReader reader( in );
    if ( !reader.valid() )
        return unexpected( std::string( "PLY file open error" ) );

    uint32_t indecies[3];
    bool gotVerts = false, gotFaces = false;

    std::vector<unsigned char> colorsBuffer;
    Mesh res;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );
    const float streamSize = float( posEnd - posStart );
    
    for ( int i = 0; reader.has_element() && ( !gotVerts || !gotFaces ); reader.next_element(), ++i )
    {
        if ( reader.element_is(miniply::kPLYVertexElement) && reader.load_element() )
        {
            auto numVerts = reader.num_rows();
            if ( reader.find_pos( indecies ) )
            {
                Timer t( "extractPoints" );
                res.points.resize( numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::Float, res.points.data() );
                gotVerts = true;
            }
            if ( settings.colors && reader.find_color( indecies ) )
            {
                Timer t( "extractColors" );
                colorsBuffer.resize( 3 * numVerts );
                reader.extract_properties( indecies, 3, miniply::PLYPropertyType::UChar, colorsBuffer.data() );
            }
            const float progress = float( in.tellg() - posStart ) / streamSize;
            if ( reportProgress( settings.callback, progress ) )
                return unexpected( std::string( "Loading canceled" ) );
            continue;
        }

        const auto posLast = in.tellg();
        if ( reader.element_is(miniply::kPLYFaceElement) && reader.load_element() && reader.find_indices(indecies) )
        {
            bool polys = reader.requires_triangulation( indecies[0] );
            if ( polys && !gotVerts )
                return unexpected( std::string( "PLY file open: need vertex positions to triangulate faces" ) );

            Triangulation tris;
            if (polys) 
            {
                Timer t( "extractTriangles" );
                auto numIndices = reader.num_triangles( indecies[0] );
                tris.resize( numIndices );
                reader.extract_triangles( indecies[0], &res.points.front().x, (std::uint32_t)res.points.size(), miniply::PLYPropertyType::Int, &tris.front() );
            }
            else 
            {
                Timer t( "extractTriples" );
                auto numIndices = reader.num_rows();
                tris.resize( numIndices );
                reader.extract_list_property( indecies[0], miniply::PLYPropertyType::Int, &tris.front() );
            }
            const auto posCurent = in.tellg();
            // suppose  that reading is 10% of progress and building mesh is 90% of progress
            if ( reportProgress( settings.callback, ( float( posLast ) + ( posCurent - posLast ) * 0.1f - posStart ) / streamSize ) )
                return unexpected( std::string( "Loading canceled" ) );
            bool isCanceled = false;
            ProgressCallback partedProgressCb = settings.callback ? [callback = settings.callback, posLast, posCurent, posStart, streamSize, &isCanceled] ( float v )
            {
                const bool res = callback( ( float( posLast ) + ( posCurent - posLast ) * ( 0.1f + v * 0.9f ) - posStart ) / streamSize );
                isCanceled |= !res;
                return res;
            } : settings.callback;
            res.topology = MeshBuilder::fromTriangles( tris, {}, partedProgressCb );
            if ( settings.callback && ( !settings.callback( float( posCurent - posStart ) / streamSize ) || isCanceled ) )
                return unexpected( std::string( "Loading canceled" ) );
            gotFaces = true;
        }
    }

    if ( !reader.valid() )
        return unexpected( std::string( "PLY file read or parse error" ) );

    if ( !gotVerts )
        return unexpected( std::string( "PLY file does not contain vertices" ) );

    if ( settings.colors && !colorsBuffer.empty() )
    {
        settings.colors->resize( res.points.size() );
        for ( VertId i{ 0 }; i < res.points.size(); ++i )
        {
            int ind = 3 * i;
            ( *settings.colors )[i] = Color( colorsBuffer[ind], colorsBuffer[ind + 1], colorsBuffer[ind + 2] );
        }
    }

    return res;
}

#ifndef MRMESH_NO_OPENCTM

Expected<Mesh, std::string> fromCtm( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( fromCtm( in, settings ), file );
}

Expected<Mesh, std::string> fromCtm( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    class ScopedCtmConext 
    {
        CTMcontext context_ = ctmNewContext( CTM_IMPORT );
    public:
        ~ScopedCtmConext() { ctmFreeContext( context_ ); }
        operator CTMcontext() { return context_; }
    } context;


    struct LoadData
    {
        std::function<bool( float )> callbackFn{};
        std::istream* stream;
        bool wasCanceled{ false };
    } loadData;
    loadData.stream = &in;

    const auto posStart = in.tellg();
    in.seekg( 0, std::ios_base::end );
    const auto posEnd = in.tellg();
    in.seekg( posStart );

    if ( settings.callback )
    {
        loadData.callbackFn = [callback = settings.callback, posStart, sizeAll = float( posEnd - posStart ), &in] ( float )
        {
            float progress = float( in.tellg() - posStart ) / sizeAll;
            return callback( progress );
        };
    }

    ctmLoadCustom( context, []( void * buf, CTMuint size, void * data )
    {
        LoadData& loadData = *reinterpret_cast<LoadData*>( data );
        auto& stream = *loadData.stream;
        auto pos = stream.tellg();
        loadData.wasCanceled |= !readByBlocks( stream, ( char* )buf, size, loadData.callbackFn, 1u << 12 );
        if ( loadData.wasCanceled )
            return 0u;
        return (CTMuint)( stream.tellg() - pos );
    }, &loadData );

    auto vertCount = ctmGetInteger( context, CTM_VERTEX_COUNT );
    auto triCount  = ctmGetInteger( context, CTM_TRIANGLE_COUNT );
    auto vertices  = ctmGetFloatArray( context, CTM_VERTICES );
    auto indices   = ctmGetIntegerArray( context, CTM_INDICES );
    if ( loadData.wasCanceled )
        return unexpected( "Loading canceled" );
    if ( ctmGetError(context) != CTM_NONE )
        return unexpected( "Error reading CTM format" );

    // even if we save false triangle (0,0,0) in MG2 format, it can be open as triangle (i,i,i)
    if ( triCount == 1 && indices[0] == indices[1] && indices[0] == indices[2] )
    {
        // CTM file is representing points, but it was written with the library requiring the presence of at least one triangle
        triCount = 0;
    }

    if ( settings.colors )
    {
        auto colorAttrib = ctmGetNamedAttribMap( context, "Color" );
        if ( colorAttrib != CTM_NONE )
        {
            auto colorArray = ctmGetFloatArray( context, colorAttrib );
            settings.colors->resize( vertCount );
            for ( VertId i{ 0 }; CTMuint( i ) < vertCount; ++i )
            {
                auto j = 4 * i;
                (*settings.colors)[i] = Color( colorArray[j], colorArray[j + 1], colorArray[j + 2], colorArray[j + 3] );
            }
        }
    }

    Mesh mesh;
    mesh.points.resize( vertCount );
    for ( VertId i{0}; i < (int)vertCount; ++i )
        mesh.points[i] = Vector3f( vertices[3*i], vertices[3*i+1], vertices[3*i+2] );

    Triangulation t;
    t.reserve( triCount );
    for ( FaceId i{0}; i < (int)triCount; ++i )
        t.push_back( { VertId( (int)indices[3*i] ), VertId( (int)indices[3*i+1] ), VertId( (int)indices[3*i+2] ) } );

    mesh.topology = MeshBuilder::fromTriangles( t );

    return mesh;
}
#endif

#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
Expected<Mesh, std::string> from3mfModel( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    std::ifstream in( file, std::ifstream::binary );
    if ( !in )
        return unexpected( std::string( "Cannot open file for reading " ) + utf8string( file ) );

    return addFileNameInError( from3mfModel( in, settings ), file );
}

Expected<Mesh, std::string> from3mfModel( std::istream& in, const MeshLoadSettings& settings /*= {}*/ )
{
    MR_TIMER

    // find size
    in.seekg( 0, std::ios_base::end );
    size_t size = in.tellg();
    in.seekg( 0 );
    // read to char vector
    std::vector<char> docStr( size + 1 );
    in.read( docStr.data(), size );
    if ( in.fail() || in.bad() )
        return unexpected( std::string( "3DF model file read error" ) );

    tinyxml2::XMLDocument doc;
    if ( tinyxml2::XML_SUCCESS != doc.Parse( docStr.data(), docStr.size() ) )
        return unexpected( std::string( "3DF model file parse error" ) );

    auto rootNode = doc.FirstChildElement();
    if ( !rootNode || std::string( rootNode->Name() ) != "model" )
        return unexpected( std::string( "3DF model root node is not 'model' but '" ) + std::string( rootNode->Name() ) + "'" );

    auto resourcesNode = rootNode->FirstChildElement( "resources" );
    if ( !resourcesNode )
        return unexpected( std::string( "3DF model 'resources' node not found" ) );

    auto objectNode = resourcesNode->FirstChildElement( "object" );
    if ( !resourcesNode )
        return unexpected( std::string( "3DF model 'object' node not found" ) );

    auto meshNode = objectNode->FirstChildElement( "mesh" );
    if ( !objectNode )
        return unexpected( std::string( "3DF model 'mesh' node not found" ) );

    auto verticesNode = meshNode->FirstChildElement( "vertices" );
    if ( !verticesNode )
        return unexpected( std::string( "3DF model 'vertices' node not found" ) );

    VertCoords vertexCoordinates;
    for ( auto vertexNode = verticesNode->FirstChildElement( "vertex" ); vertexNode; vertexNode = vertexNode->NextSiblingElement( "vertex" ) )
    {
        Vector3f p;
        if ( tinyxml2::XML_SUCCESS != vertexNode->QueryFloatAttribute( "x", &p.x ) )
            return unexpected( std::string( "3DF model vertex node does not have 'x' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != vertexNode->QueryFloatAttribute( "y", &p.y ) )
            return unexpected( std::string( "3DF model vertex node does not have 'y' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != vertexNode->QueryFloatAttribute( "z", &p.z ) )
            return unexpected( std::string( "3DF model vertex node does not have 'z' attribute" ) );
        vertexCoordinates.push_back( p );
    }

    if ( !reportProgress( settings.callback, 0.25f ) )
        return unexpected( std::string( "Loading canceled" ) );

    auto trianglesNode = meshNode->FirstChildElement( "triangles" );
    if ( !trianglesNode )
        return unexpected( std::string( "3DF model 'triangles' node not found" ) );

    Triangulation tris;
    for ( auto triangleNode = trianglesNode->FirstChildElement( "triangle" ); triangleNode; triangleNode = triangleNode->NextSiblingElement( "triangle" ) )
    {
        int vs[3];
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "v1", &vs[0] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'v1' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "v2", &vs[1] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'v2' attribute" ) );
        if ( tinyxml2::XML_SUCCESS != triangleNode->QueryIntAttribute( "v3", &vs[2] ) )
            return unexpected( std::string( "3DF model triangle node does not have 'v3' attribute" ) );
        tris.push_back( { VertId( vs[0] ), VertId( vs[1] ), VertId( vs[2] ) } );
    }

    if ( !reportProgress( settings.callback, 0.5f ) )
        return unexpected( std::string( "Loading canceled" ) );

    if ( settings.duplicatedVertexCount )
    {
        std::vector<MeshBuilder::VertDuplication> dups;
        const auto res = Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( vertexCoordinates ), tris, &dups );
        *settings.duplicatedVertexCount = int( dups.size() );
        return res;
    }
    else
        return Mesh::fromTrianglesDuplicatingNonManifoldVertices( std::move( vertexCoordinates ), tris );
}
#endif

Expected<Mesh, std::string> fromAnySupportedFormat( const std::filesystem::path& file, const MeshLoadSettings& settings /*= {}*/ )
{
    auto ext = utf8string( file.extension() );
    for ( auto & c : ext )
        c = (char)tolower( c );

    ext = "*" + ext;

    Expected<MR::Mesh, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    auto filters = getFilters();
    auto itF = std::find_if( filters.begin(), filters.end(), [ext]( const IOFilter& filter )
    {
        return filter.extension == ext;
    } );
    if ( itF == filters.end() )
        return res;

    auto loader = getMeshLoader( *itF );
    if ( !loader )
        return res;
    return loader( file, settings );
}

Expected<Mesh, std::string> fromAnySupportedFormat( std::istream& in, const std::string& extension, const MeshLoadSettings& settings /*= {}*/ )
{
    auto ext = extension;
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<MR::Mesh, std::string> res = unexpected( std::string( "unsupported file extension" ) );
    auto filters = getFilters();
    auto itF = std::find_if( filters.begin(), filters.end(), [ext] ( const IOFilter& filter )
    {
        return filter.extension == ext;
    } );
    if ( itF == filters.end() )
        return res;

    auto loader = getMeshStreamLoader( *itF );
    if ( !loader )
        return res;

    return loader( in, settings );
}

/*
MeshLoaderAdder __meshLoaderAdder( NamedMeshLoader{IOFilter( "MrMesh (.mrmesh)", "*.mrmesh" ),MeshLoader{static_cast<Expected<MR::Mesh, std::string>(*)(const std::filesystem::path&,VertColors*)>(fromMrmesh)}} );
*/

MR_ADD_MESH_LOADER( IOFilter( "MeshInspector (.mrmesh)", "*.mrmesh" ), fromMrmesh )
MR_ADD_MESH_LOADER( IOFilter( "Stereolithography (.stl)", "*.stl" ), fromAnyStl )
MR_ADD_MESH_LOADER( IOFilter( "Object format file (.off)", "*.off" ), fromOff )
MR_ADD_MESH_LOADER( IOFilter( "3D model object (.obj)", "*.obj" ), fromObj )
MR_ADD_MESH_LOADER( IOFilter( "Polygon File Format (.ply)", "*.ply" ), fromPly )
#ifndef MRMESH_NO_OPENCTM
MR_ADD_MESH_LOADER( IOFilter( "Compact triangle-based mesh (.ctm)", "*.ctm" ), fromCtm )
#endif
#if !defined( __EMSCRIPTEN__ ) && !defined( MRMESH_NO_XML )
MR_ADD_MESH_LOADER( IOFilter( "3D Manufacturing Format (.model)", "*.model" ), from3mfModel )
#endif

} //namespace MeshLoad

} //namespace MR
