#include "MRFileDialog.h"
#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRMesh/MRConfig.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystem.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRWasm.h"
#include "MRColorTheme.h"
#include <GLFW/glfw3.h>
#include <clocale>

#ifndef _WIN32
  #ifndef __EMSCRIPTEN__
    #include <gtkmm.h>
  #endif
#else
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
#ifndef __EMSCRIPTEN__
#include <GLFW/glfw3native.h>
#endif

#ifdef __EMSCRIPTEN__
namespace
{
static std::function<void( const std::vector<std::filesystem::path>& )> sDialogFilesCallback;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE int emsOpenFiles( int count, const char** filenames )
{
    if ( !sDialogFilesCallback )
        return 1;
    std::vector<std::filesystem::path> paths( count );
    for ( int i = 0; i < count; ++i )
    {
        paths[i] = MR::pathFromUtf8( filenames[i] );
    }
    sDialogFilesCallback( paths );
    sDialogFilesCallback = {};
    return 0;
}

EMSCRIPTEN_KEEPALIVE int emsSaveFile( const char* filename )
{
    if ( !sDialogFilesCallback )
        return 1;

    std::filesystem::path savePath = std::string( filename );
    sDialogFilesCallback( { savePath } );

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( save_file( UTF8ToString( $0 ) ), filename );
#pragma clang diagnostic pop
    return 0;
}

EMSCRIPTEN_KEEPALIVE int emsOpenDirectory( const char* dirname )
{
    if ( !sDialogFilesCallback )
        return 1;

    const std::filesystem::path path = std::string( dirname );
    sDialogFilesCallback( { path } );
    sDialogFilesCallback = {};

    return 0;
}

}

#endif

namespace
{

struct FileDialogParameters : MR::FileParameters
{
    bool folderDialog{false}; // open dialog only
    bool multiselect{true};   // open dialog only
    bool saveDialog{false};   // true for save dialog, false for open
};

#if !defined( __EMSCRIPTEN__ ) && !defined( _WIN32 )
const std::string cLastUsedDirKey = "lastUsedDir";

std::string getCurrentFolder( const FileDialogParameters& params )
{
    if ( !params.baseFolder.empty() )
        return MR::utf8string( params.baseFolder );

    auto& cfg = MR::Config::instance();
    if ( cfg.hasJsonValue( cLastUsedDirKey ) )
    {
        auto lastUsedDir = cfg.getJsonValue( cLastUsedDirKey );
        if ( lastUsedDir.isString() )
            return lastUsedDir.asString();
    }

    return MR::utf8string( MR::GetHomeDirectory() );
}
#endif

#ifdef  _WIN32
std::vector<std::filesystem::path> windowsDialog( const FileDialogParameters& params = {} )
{
    std::vector<std::filesystem::path> res;
    //<SnippetRefCounts>
    HRESULT hr = CoInitializeEx( NULL, COINIT_APARTMENTTHREADED |
        COINIT_DISABLE_OLE1DDE );

    std::vector<COMDLG_FILTERSPEC> filters;
    std::vector<std::pair<std::wstring, std::wstring>> filtersWCopy;

    if( SUCCEEDED( hr ) )
    {
        IFileDialog* pFileOpen;

        // Create the FileOpenDialog object.
        hr = CoCreateInstance( params.saveDialog ? CLSID_FileSaveDialog : CLSID_FileOpenDialog,
            NULL, CLSCTX_ALL,
            params.saveDialog ? IID_IFileSaveDialog : IID_IFileOpenDialog,
            reinterpret_cast<void**>(&pFileOpen) );

        if( SUCCEEDED( hr ) )
        {
            if( !params.baseFolder.empty() )
            {
                IShellItem* pItem;
                hr = SHCreateItemFromParsingName( params.baseFolder.c_str(), NULL, IID_PPV_ARGS( &pItem ) );
                if( SUCCEEDED( hr ) )
                {
                    pFileOpen->SetFolder( pItem );
                    pItem->Release();
                }
            }

            // Set the options on the dialog.
            DWORD dwFlags;
            // Before setting, always get the options first in order
            // not to override existing options.
            pFileOpen->GetOptions( &dwFlags );
            pFileOpen->SetOptions( dwFlags | (params.multiselect ? FOS_ALLOWMULTISELECT : 0) |
                (params.folderDialog ? FOS_PICKFOLDERS : 0) );

            if( !params.filters.empty() )
            {
                unsigned filtersSize = unsigned( params.filters.size() );
                filtersWCopy.resize( filtersSize );
                filters.resize( filtersSize );

                for( unsigned i = 0; i < filtersSize; ++i )
                {
                    const auto& [nameU8, filterU8] = params.filters[i];
                    auto& [nameW, filterW] = filtersWCopy[i];
                    nameW = MR::utf8ToWide( nameU8.c_str() );
                    filterW = MR::utf8ToWide( filterU8.c_str() );

                    filters[i].pszName = nameW.c_str();
                    filters[i].pszSpec = filterW.c_str();
                }

                pFileOpen->SetFileTypes( filtersSize, filters.data() );
                pFileOpen->SetDefaultExtension( L"" );
            }

            if ( !params.fileName.empty() )
            {
                auto filenameW = MR::utf8ToWide( params.fileName.c_str() );
                pFileOpen->SetFileName( filenameW.c_str() );
            }

            // Show the Open dialog box.
            hr = pFileOpen->Show( glfwGetWin32Window( glfwGetCurrentContext() ) );

            // Get the file name from the dialog box.
            if( SUCCEEDED( hr ) )
            {
                if( !params.saveDialog )
                {
                    IFileOpenDialog* fileOpenDialog = (IFileOpenDialog*) pFileOpen;
                    IShellItemArray* pItems;
                    hr = fileOpenDialog->GetResults( &pItems );
                    if( SUCCEEDED( hr ) )
                    {
                        DWORD itemCount;
                        hr = pItems->GetCount( &itemCount );
                        if( SUCCEEDED( hr ) )
                        {
                            res.reserve( itemCount );
                            for( DWORD i = 0; i < itemCount; ++i )
                            {
                                IShellItem* item;
                                hr = pItems->GetItemAt( i, &item );
                                if( SUCCEEDED( hr ) )
                                {
                                    PWSTR pszFilePath;
                                    hr = item->GetDisplayName( SIGDN_FILESYSPATH, &pszFilePath );
                                    if( SUCCEEDED( hr ) )
                                    {
                                        res.push_back( pszFilePath );
                                        CoTaskMemFree( pszFilePath );
                                    }
                                    item->Release();
                                }
                            }
                            pItems->Release();
                        }
                    }
                }
                else
                {
                    IFileSaveDialog* fileSaveDialog = (IFileSaveDialog*) pFileOpen;
                    IShellItem* pItem;
                    hr = fileSaveDialog->GetResult( &pItem );

                    if( SUCCEEDED( hr ) )
                    {
                        PWSTR pszFilePath;
                        hr = pItem->GetDisplayName( SIGDN_FILESYSPATH, &pszFilePath );
                        if( SUCCEEDED( hr ) )
                        {
                            res.push_back( pszFilePath );
                            CoTaskMemFree( pszFilePath );
                        }

                        pItem->Release();
                    }
                }
            }
            pFileOpen->Release();
        }
        CoUninitialize();
    }

    return res;
}
#else
#ifndef __EMSCRIPTEN__
std::string gtkDialogTitle( Gtk::FileChooserAction action, bool multiple = false )
{
    switch ( action )
    {
    case Gtk::FILE_CHOOSER_ACTION_OPEN:
        return multiple ? "Open Files" : "Open File";
    case Gtk::FILE_CHOOSER_ACTION_SAVE:
        return "Save File";
    case Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER:
        return multiple ? "Open Folders" : "Open Folder";
    case Gtk::FILE_CHOOSER_ACTION_CREATE_FOLDER:
        return "Save Folder";
    }
    assert( false );
    return {};
}

std::vector<std::filesystem::path> gtkDialog( const FileDialogParameters& params = {} )
{
    // Gtk has a nasty habit of overriding the locale to "".
    std::string locale = std::setlocale( LC_ALL, nullptr );
    auto kit = Gtk::Application::create();
    std::setlocale( LC_ALL, locale.c_str() );

    Gtk::FileChooserAction action;
    if ( params.folderDialog )
        action = params.saveDialog ? Gtk::FILE_CHOOSER_ACTION_CREATE_FOLDER : Gtk::FILE_CHOOSER_ACTION_SELECT_FOLDER;
    else
        action = params.saveDialog ? Gtk::FILE_CHOOSER_ACTION_SAVE : Gtk::FILE_CHOOSER_ACTION_OPEN;
#if defined( __APPLE__ )
    const auto dialogPtr = Gtk::FileChooserNative::create(gtkDialogTitle( action, params.multiselect ), action );
    auto& dialog = *dialogPtr.get();
#else
    Gtk::FileChooserDialog dialog( gtkDialogTitle( action, params.multiselect ), action );
#endif
    dialog.set_select_multiple( params.multiselect );

#if !defined( __APPLE__ )
    dialog.add_button( Gtk::Stock::CANCEL, Gtk::RESPONSE_CANCEL );
    dialog.add_button( params.saveDialog ? Gtk::Stock::SAVE : Gtk::Stock::OPEN, Gtk::RESPONSE_ACCEPT );
#endif

    for ( const auto& filter: params.filters )
    {
        auto filterText = Gtk::FileFilter::create();
        filterText->set_name( filter.name );
        size_t separatorPos = 0;
        for (;;)
        {
            auto nextSeparatorPos = filter.extensions.find( ";", separatorPos );
            auto ext = filter.extensions.substr( separatorPos, nextSeparatorPos - separatorPos );
#if defined( __APPLE__ )
            if ( ext == "*.*" )
                ext = "*";
#endif
            filterText->add_pattern( ext );
            if ( nextSeparatorPos == std::string::npos )
                break;
            separatorPos = nextSeparatorPos + 1;
        }
        dialog.add_filter( filterText );
    }

    dialog.set_current_folder( getCurrentFolder( params ) );

    if ( !params.fileName.empty() )
        dialog.set_current_name( params.fileName );

    if ( params.saveDialog )
        dialog.set_do_overwrite_confirmation( true );

    std::vector<std::filesystem::path> results;
    auto onResponse = [&] ( int responseId )
    {
        if ( responseId == Gtk::RESPONSE_ACCEPT )
        {
            for ( const auto& filename : dialog.get_filenames() )
            {
                std::filesystem::path filepath( filename );
                if ( params.saveDialog && !filepath.has_extension() )
                {
                    const std::string filterName = dialog.get_filter()->get_name();
                    for ( const auto& filter: params.filters )
                    {
                        if ( filterName == filter.name )
                        {
                            filepath.replace_extension( filter.extensions.substr( 1 ) );
                            break;
                        }
                    }
                }
                results.emplace_back( std::move( filepath ) );
            }

            auto& cfg = MR::Config::instance();
            cfg.setJsonValue( cLastUsedDirKey, dialog.get_current_folder() );
        }
        else if ( responseId != Gtk::RESPONSE_CANCEL )
        {
            spdlog::warn( "GTK dialog failed" );
        }
#if defined( __APPLE__ )
        // on macOS the main window remains unfocused after the file dialog is closed
        MR::CommandLoop::appendCommand( []
        {
            glfwFocusWindow( MR::Viewer::instance()->window );
        } );
#endif
    };
#if defined( __APPLE__ )
    onResponse( dialog.run() );
#else // __APPLE__
    dialog.signal_response().connect([&] ( int responseId )
    {
        onResponse( responseId );
        dialog.hide();
    });
    kit->run( dialog );
#endif // __APPLE__

    return results;
}
#else
std::string webAccumFilter( const MR::IOFilters& filters )
{
    std::string accumFilter;
    for (  const auto& filter : filters )
    {
        size_t separatorPos = 0;
        for (;;)
        {
            auto nextSeparatorPos = filter.extensions.find( ";", separatorPos );
            auto ext = filter.extensions.substr( separatorPos, nextSeparatorPos - separatorPos );
            accumFilter += ( ext.substr( 1 ) + ", " );
            if ( nextSeparatorPos == std::string::npos )
                break;
            separatorPos = nextSeparatorPos + 1;
        }
    }
    accumFilter = accumFilter.substr( 0, accumFilter.size() - 2 );
    return accumFilter;
}
#endif
#endif
}

namespace MR
{

std::filesystem::path openFileDialog( const FileParameters& params )
{
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = false;
    if ( parameters.filters.empty() )
        parameters.filters = { { "All files", "*.*" } };

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
        return results[0];
    return {};
}

void openFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params /*= {} */ )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( openFileDialog( params ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
            callback( paths[0] );
    };
    std::string accumFilter = webAccumFilter( params.filters );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( open_files_dialog_popup( UTF8ToString( $0 ), $1), accumFilter.c_str(), false);
#pragma clang diagnostic pop
#endif
}

std::vector<std::filesystem::path> openFilesDialog( const FileParameters& params )
{
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = true;
    parameters.saveDialog = false;
    if ( parameters.filters.empty() )
        parameters.filters = { { "All files", "*.*" } };

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    return results;
}

void openFilesDialogAsync( std::function<void( const std::vector<std::filesystem::path>& )> callback, const FileParameters& params /*= {} */ )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( openFilesDialog( params ) );
#else
    sDialogFilesCallback = callback;
    std::string accumFilter = webAccumFilter( params.filters );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( open_files_dialog_popup( UTF8ToString( $0 ), $1), accumFilter.c_str(), true);
#pragma clang diagnostic pop
#endif
}

std::filesystem::path openFolderDialog( std::filesystem::path baseFolder )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();

    FileDialogParameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = false;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
        return results[0];
    return {};
}
    
void openFolderDialogAsync( std::function<void ( const std::filesystem::path& )> callback, std::filesystem::path baseFolder )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( openFolderDialog( std::move( baseFolder ) ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
            callback( paths[0] );
    };
    (void)baseFolder;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( open_directory_dialog_popup());
#pragma clang diagnostic pop
#endif
}

std::vector<std::filesystem::path> openFoldersDialog( std::filesystem::path baseFolder )
{
    // Windows dialog does not support forward slashes between folders
    baseFolder.make_preferred();

    FileDialogParameters parameters;
    parameters.baseFolder = baseFolder;
    parameters.folderDialog = true;
    parameters.multiselect = true;
    parameters.saveDialog = false;

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    return results;
}

std::filesystem::path saveFileDialog( const FileParameters& params /*= {} */ )
{
    FileDialogParameters parameters{ params };
    parameters.folderDialog = false;
    parameters.multiselect = false;
    parameters.saveDialog = true;
    if ( parameters.filters.empty() )
        parameters.filters = { { "All files", "*.*" } };

    std::vector<std::filesystem::path> results;
#if defined( _WIN32 )
    results = windowsDialog( parameters );
#elif !defined( __EMSCRIPTEN__ )
    results = gtkDialog( parameters );
#endif
    if ( results.size() == 1 )
        return results[0];
    return {};
}

void saveFileDialogAsync( std::function<void( const std::filesystem::path& )> callback, const FileParameters& params /*= {} */ )
{
    assert( callback );
#ifndef __EMSCRIPTEN__
    callback( saveFileDialog( params ) );
#else
    sDialogFilesCallback = [callback] ( const std::vector<std::filesystem::path>& paths )
    {
        if ( !paths.empty() )
            callback( paths[0] );
    };
    auto filters = params.filters;
    filters.erase( std::remove_if( filters.begin(), filters.end(), [] ( const auto& filter )
    {
        return filter.extensions == "*.*";
    } ), filters.end() );
    std::string accumFilter = webAccumFilter( params.filters );
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
    EM_ASM( download_file_dialog_popup( UTF8ToString( $0 ), UTF8ToString( $1 )), params.fileName.c_str(), accumFilter.c_str() );
#pragma clang diagnostic pop
#endif
}

}
