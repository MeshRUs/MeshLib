#include "MRImageLoad.h"
#include "MRBuffer.h"
#include "MRFile.h"
#include "MRImage.h"
#include "MRStringConvert.h"

#include "MRExpected.h"

#include <filesystem>
#include <fstream>
#include <string>

#ifndef MRMESH_NO_PNG
#ifdef __EMSCRIPTEN__
#include <png.h>
#else
#include <libpng16/png.h>
#endif
#endif

#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_JPEG
#include <turbojpeg.h>
#endif
#endif

namespace MR
{
namespace ImageLoad
{

const IOFilters Filters =
{
#ifndef MRMESH_NO_PNG
    {"Portable Network Graphics (.png)",  "*.png"},
#endif
#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_JPEG
    {"JPEG (.jpg)",  "*.jpg"},
    {"JPEG (.jpeg)", "*.jpeg"},
#endif
#endif
};

#ifndef MRMESH_NO_PNG
struct ReadPng
{
    ReadPng( std::istream& in )
    {
        pngPtr = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        if ( !pngPtr )
            return;
        infoPtr = png_create_info_struct( pngPtr );

        png_set_read_fn( pngPtr, ( png_voidp )&in, userReadData );
    }

    ~ReadPng()
    {
        if ( pngPtr )
            png_destroy_read_struct( &pngPtr, &infoPtr, NULL );
    }

    static void userReadData( png_structp pngPtr, png_bytep data, png_size_t length )
    {
        png_voidp a = png_get_io_ptr( pngPtr );
        ( ( std::istream* )a )->read( ( char* )data, length );
    }

    png_structp pngPtr{ nullptr };
    png_infop infoPtr{ nullptr };
    png_colorp palette{ nullptr };
    int paletteSize{ 0 };
    png_bytep alphaPalette{ nullptr };
    int alphaPaletteSize{ 0 };
};

Expected<Image, std::string> fromPng( const std::filesystem::path& path )
{
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return unexpected( "Cannot open file " + utf8string( path ) );

    return addFileNameInError( fromPng( in ), path );
}

Expected<MR::Image, std::string> fromPng( std::istream& in )
{
    ReadPng png( in );

    if ( !png.pngPtr )
        return unexpected( std::string( "Cannot read png" ) );

    if ( !png.infoPtr )
        return unexpected( std::string( "Cannot create png info" ) );

    Image result;

    unsigned w{ 0 }, h{ 0 };
    int depth{ 0 }; // 8
    int colorType{ 0 }; // PNG_COLOR_TYPE_RGBA
    int interlace{ 0 }; // PNG_INTERLACE_NONE
    int compression{ 0 }; // PNG_COMPRESSION_TYPE_DEFAULT
    int filter{ 0 }; // PNG_FILTER_TYPE_DEFAULT
    // Read info
    png_read_info( png.pngPtr, png.infoPtr );
    png_get_IHDR(
      png.pngPtr,
      png.infoPtr,
      &w, &h,
      &depth,
      &colorType,
      &interlace,
      &compression,
      &filter
    );

    result.resolution = Vector2i{ int( w ),int( h ) };
    result.pixels.resize( result.resolution.x * result.resolution.y );

    std::vector<unsigned char*> ptrs( result.resolution.y );
    if ( colorType == PNG_COLOR_TYPE_RGBA )
    {
        for ( int i = 0; i < result.resolution.y; ++i )
            ptrs[result.resolution.y - i - 1] = ( unsigned char* )( result.pixels.data() + result.resolution.x * i );
        png_read_image( png.pngPtr, ptrs.data() );
    }
    else if ( colorType == PNG_COLOR_TYPE_RGB )
    {
        std::vector<Vector3<unsigned char>> rawPixels( result.resolution.x * result.resolution.y );
        for ( int i = 0; i < result.resolution.y; ++i )
            ptrs[result.resolution.y - i - 1] = ( unsigned char* )( rawPixels.data() + result.resolution.x * i );
        png_read_image( png.pngPtr, ptrs.data() );
        for ( size_t i = 0; i < result.pixels.size(); ++i )
            result.pixels[i] = Color( rawPixels[i] );
    }
    else if ( colorType == PNG_COLOR_TYPE_PALETTE )
    {
        png_get_PLTE( png.pngPtr, png.infoPtr, &png.palette, &png.paletteSize );
        png_get_tRNS( png.pngPtr, png.infoPtr, &png.alphaPalette, &png.alphaPaletteSize, nullptr );

        std::vector<Color> palette( png.paletteSize );
        for ( int i = 0; i < png.paletteSize; ++i )
        {
            palette[i] = Color( png.palette[i].red, png.palette[i].green, png.palette[i].blue );
            if ( png.alphaPalette && i < png.alphaPaletteSize )
                palette[i].a = png.alphaPalette[i];
        }

        std::vector<unsigned char> rawPixels( result.resolution.x * result.resolution.y );
        for ( int i = 0; i < result.resolution.y; ++i )
            ptrs[result.resolution.y - i - 1] = ( unsigned char* )( rawPixels.data() + result.resolution.x * i );

        png_read_image( png.pngPtr, ptrs.data() );
        for ( int i = 0; i < result.resolution.y; ++i )
            for ( int j = 0; j < result.resolution.x; ++j )
            {
                result.pixels[i * result.resolution.x + j] = palette[rawPixels[i * result.resolution.x + j]];
            }
    }
    else
        return unexpected( "Unsupported png color type" );

    png_read_end( png.pngPtr, NULL );
    return result;
}

#endif

#ifndef __EMSCRIPTEN__

#ifndef MRMESH_NO_JPEG
struct JpegReader
{
    JpegReader()
    {
        tjInstance = tjInitDecompress();
    }
    ~JpegReader()
    {
        if ( tjInstance )
            tjDestroy( tjInstance );
    }
    tjhandle tjInstance{ nullptr };
};

Expected<Image, std::string> fromJpeg( const std::filesystem::path& path )
{
    std::ifstream in( path, std::ios::binary );
    if ( !in )
        return unexpected( "Cannot open file " + utf8string( path ) );

    return addFileNameInError( fromJpeg( in ), path );
}

Expected<MR::Image, std::string> fromJpeg( std::istream& in )
{
    std::error_code ec;
    in.seekg( 0, std::ios::end );
    size_t fileSize = in.tellg();
    in.seekg( 0 );

    Buffer<char> buffer( fileSize );
    in.read( buffer.data(), ( ptrdiff_t )buffer.size() );
    if ( !in )
        return unexpected( "Cannot read file" );

    JpegReader reader;
    if ( !reader.tjInstance )
        return unexpected( "Cannot initialize JPEG decompressor" );

    int width, height, jpegSubsamp, jpegColorspace;
    auto res = tjDecompressHeader3( reader.tjInstance, ( const unsigned char* )buffer.data(), ( unsigned long )buffer.size(), &width, &height, &jpegSubsamp, &jpegColorspace );
    if ( res != 0 )
        return unexpected( "Failed to decompress JPEG header" );

    Image image;
    image.pixels.resize( width * height );
    image.resolution = { width, height };
    res = tjDecompress2( reader.tjInstance, ( const unsigned char* )buffer.data(), ( unsigned long )buffer.size(), reinterpret_cast< unsigned char* >( image.pixels.data() ), width, 0, height, TJPF_RGBA, TJFLAG_BOTTOMUP );
    if ( res != 0 )
        return unexpected( "Failed to decompress JPEG file" );

    return image;
}

#endif

#endif

Expected<Image, std::string> fromAnySupportedFormat( const std::filesystem::path& file )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<Image, std::string> res = unexpected( std::string( "unsupported file extension" ) );
#ifndef MRMESH_NO_PNG
    if ( ext == ".png" )
        return MR::ImageLoad::fromPng( file );
#endif
#ifndef __EMSCRIPTEN__
#ifndef MRMESH_NO_JPEG
    if ( ext == ".jpg" || ext == ".jpeg" )
        return MR::ImageLoad::fromJpeg( file );
#endif
#endif
    return res;
}

}

}
