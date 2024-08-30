#include "MRPch/MRSpdlog.h"
#include "MRMesh/MRStringConvert.h"
#include "MRViewer.h"
#include "MRCommandLoop.h"

#ifdef __APPLE__

// callback function from .mm
// accumulates file names until called with null, then handles the list
extern "C" void handle_load_message( const char* filePath )
{
    static std::vector<std::filesystem::path> filePaths;
    static std::string joined;
    if ( filePath )
    {
        filePaths.push_back( MR::pathFromUtf8( filePath ) );
        if ( !joined.empty() )
            joined += ',';
        joined += filePath;
    }
    else
    {
        spdlog::info( "Open file(s) requested: {}", joined );
        MR::CommandLoop::appendCommand( [paths = filePaths] ()
        {
            MR::getViewerInstance().openFilesRequested( paths );
        }, MR::CommandLoop::StartPosition::AfterPluginInit );
        joined.clear();
        filePaths.clear();
    }
}

#endif
