#include "MRGcodeLoad.h"
#include "MRStringConvert.h"
#include "MRTimer.h"
#include "MRIOFormatsRegistry.h"
#include <fstream>
#include <sstream>

namespace MR::GcodeLoad
{
Expected<GcodeSource> fromGcode( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    std::ifstream filestream( file );
    return fromGcode( filestream, callback );
}

Expected<MR::GcodeSource> fromGcode( std::istream& in, ProgressCallback /*= {} */ )
{
    MR_TIMER
    std::vector<std::string> res;
    while ( in )
    {
        std::string s;
        std::getline( in, s );
        if ( !s.empty() )
            res.push_back( std::move( s ) );
    }
    return res;
}

Expected<GcodeSource> fromAnySupportedFormat( const std::filesystem::path& file, ProgressCallback callback /*= {} */ )
{
    auto ext = utf8string( file.extension() );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<std::vector<std::string>> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".gcode" || ext == ".txt" || ext == ".nc" )
        res = fromGcode( file, callback );
    return res;
}

Expected<MR::GcodeSource> fromAnySupportedFormat( std::istream& in, const std::string& extension, ProgressCallback callback /*= {} */ )
{
    auto ext = extension.substr( 1 );
    for ( auto& c : ext )
        c = ( char )tolower( c );

    Expected<GcodeSource> res = unexpected( std::string( "unsupported file extension" ) );
    if ( ext == ".gcode" || ext == ".txt" || ext == ".nc" )
        res = fromGcode( in, callback );
    return res;
}

MR_ADD_GCODE_LOADER( IOFilter( "G-code", "*.gcode" ), fromGcode )
MR_ADD_GCODE_LOADER( IOFilter( "Numerical Control", "*.nc" ), fromGcode )
MR_ADD_GCODE_LOADER( IOFilter( "Text file", "*.txt" ), fromGcode )

}
