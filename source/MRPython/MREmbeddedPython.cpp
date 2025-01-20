#include "MREmbeddedPython.h"
#include "MRPython.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRPch/MRSpdlog.h"
#include "MRUnifiedPythonStream.h"

#include <pybind11/embed.h>

#include <fstream>

namespace MR
{

EmbeddedPython::Config EmbeddedPython::pythonConfig{};

bool EmbeddedPython::isAvailable()
{
    EmbeddedPython &self = instance_();
    return self.available_ && !self.shutdownCalled_;
}

void EmbeddedPython::shutdown()
{
    if ( !isAvailable() )
        return;

    EmbeddedPython &self = instance_();
    self.shutdownCalled_ = true;

    if ( !self.interpreterThread_.joinable() )
        return; // Nothing to do.

    { // Tell the thread to stop.
        self.stopInterpreterThread_ = true;
        self.cv_.notify_all();
    }

    self.interpreterThread_.join();
}

bool EmbeddedPython::runString( std::string pythonString, std::function<void( bool success )> onDoneAsync )
{
    if ( !isAvailable() )
        return false;

    EmbeddedPython &self = instance_();
    self.ensureInterpreterThreadIsRunning_();

    // Negotiate with the interpreter thread.
    std::unique_lock guard( self.cvMutex_ );
    self.cv_.wait( guard, [&]{ return self.state_ == State::idle; } );
    self.queuedSource_ = std::move( pythonString );
    self.state_ = State::running;
    self.onDoneAsync_ = std::move( onDoneAsync );
    self.cv_.notify_all();
    if ( self.onDoneAsync_ )
    {
        return true;
    }
    else
    {
        self.cv_.wait( guard, [&]{ return self.state_ == State::finishing; } );
        self.state_ = State::idle;
        self.cv_.notify_all();
        return self.lastRunSuccessful_;
    }
}

bool EmbeddedPython::runScript( const std::filesystem::path& path )
{
    if ( !isAvailable() || !isPythonScript( path ) )
        return false;

    std::ifstream ifs( path );
    std::ostringstream oss;
    oss << ifs.rdbuf();
    ifs.close();
    std::string str = oss.str();
    return runString( str );
}

bool EmbeddedPython::isPythonScript( const std::filesystem::path& path )
{
    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        return false;
    if ( !std::filesystem::is_regular_file( path, ec ) )
        return false;

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = (char)std::tolower( (unsigned char)c );

    if ( ext != ".py" )
        return false;

    return true;
}

EmbeddedPython::EmbeddedPython()
{
    available_ = !Py_IsInitialized();
}

EmbeddedPython::~EmbeddedPython()
{
    shutdown();
}

bool EmbeddedPython::init_()
{
    if ( !isAvailable() )
        return true;

    // Initialize our patched pybind11.
    #ifdef PYBIND11_NONLIMITEDAPI_H_
    pybind11::non_limited_api::EnsureSharedLibraryIsLoaded(true, "meshlib", SystemPath::getExecutableDirectory().value() / "meshlib");
    #endif

    PyConfig *config = pybind11::non_limited_api::PyConfig_new();
    MR_FINALLY{ pybind11::non_limited_api::PyConfig_delete( config ); };
    pybind11::non_limited_api::PyConfig_InitPythonConfig( config );
    MR_FINALLY{ pybind11::non_limited_api::PyConfig_Clear( config ); };

    pybind11::non_limited_api::PyConfig_set_site_import( config, pythonConfig.siteImport ? 1 : 0 );

    if ( !pythonConfig.home.empty() )
    {
        const auto homeW = utf8ToWide( pythonConfig.home.c_str() );
        pybind11::non_limited_api::PyStatus_delete( pybind11::non_limited_api::PyConfig_SetString( config, pybind11::non_limited_api::PyConfig_home_ptr( config ), homeW.c_str() ) );
    }

    for ( const auto& mod : PythonExport::instance().modules() )
        PyImport_AppendInittab( mod.first.c_str(), mod.second.initFncPointer );

    pybind11::non_limited_api::PyStatus_ *status = nullptr;
    MR_FINALLY{ pybind11::non_limited_api::PyStatus_delete( status ); };
    if ( pythonConfig.argv.empty() )
    {
        pybind11::non_limited_api::PyConfig_set_parse_argv( config, 0 );
        pybind11::non_limited_api::PyConfig_set_install_signal_handlers( config, 0 );
        status = pybind11::non_limited_api::PyConfig_SetBytesArgv( config, 0, NULL );
    }
    else
    {
        pybind11::non_limited_api::PyConfig_set_isolated( config, 1 );
        std::vector<char *> argv;
        for ( auto& str : pythonConfig.argv )
            argv.push_back( str.data() );
        argv.push_back( nullptr ); // Unsure if needed, just in case.
        status = pybind11::non_limited_api::PyConfig_SetBytesArgv( config, argv.size() - 1, argv.data() );
    }

    if ( pybind11::non_limited_api::PyStatus_Exception( status ) )
    {
        spdlog::error( pybind11::non_limited_api::PyStatus_get_err_msg( status ) );
        UnifiedPythonStream::get() << pybind11::non_limited_api::PyStatus_get_err_msg( status ); // add to unified python stream
        return false;
    }

    status = pybind11::non_limited_api::Py_InitializeFromConfig( config );
    if ( pybind11::non_limited_api::PyStatus_Exception( status ) )
    {
        spdlog::error( pybind11::non_limited_api::PyStatus_get_err_msg( status ) );
        UnifiedPythonStream::get() << pybind11::non_limited_api::PyStatus_get_err_msg( status ); // add to unified python stream
        return false;
    }

    return true;
}

void EmbeddedPython::ensureInterpreterThreadIsRunning_()
{
    [[maybe_unused]] static const auto once = [&]{
        // Start the interpreter thread.
        interpreterThread_ = std::thread( [this]
        {
            std::unique_lock guard( cvMutex_ );

            while ( true )
            {
                { // Wait for source code.
                    bool stop = false;
                    cv_.wait( guard, [&]
                    {
                        if ( stopInterpreterThread_ )
                        {
                            stop = true;
                            return true;
                        }

                        return state_ == State::running;
                    } );
                    if ( stop )
                        break;
                }

                lastRunSuccessful_ = false;

                static bool initOk = init_();
                if ( !initOk )
                {
                    spdlog::error( "Failied to initialize Python." );
                }
                else
                {
                    try
                    {
                        auto libDir = SystemPath::getPythonModulesDirectory();
                        auto libDirStr = utf8string( libDir );
                        MR::replaceInplace( libDirStr, "\\", "\\\\" ); // double protect first for C++ second for Python
                        // Create an empty dictionary that will function as a namespace.
                        std::string redirectScript =
                            "import sys\n"
                            "import redirector\n"
                            "sys.stdout = redirector.stdout()\n"
                            "sys.stderr = redirector.stderr()\n"
                            "sys.path.insert(1,\"" + libDirStr + "\")\n";
                        pybind11::exec( redirectScript.c_str() );

                        // Execute code
                        pybind11::exec( queuedSource_.c_str() );

                        pybind11::exec( "sys.stdout.flush()\nsys.stderr.flush()" );

                        lastRunSuccessful_ = true;
                    }
                    catch ( std::exception& e )
                    {
                        spdlog::error( e.what() );
                        UnifiedPythonStream::get() << e.what(); // add to unified python stream
                    }
                    catch ( ... )
                    {
                        spdlog::error( "Unknown exception while executing a Python script." );
                    }

                    try
                    {
                        // We used to reset globals using `pybind11::finalize_interpreter()`, but that beaks some modules (including ours at one point,
                        //   and apparently numpy too (https://stackoverflow.com/q/7676314/2752075), so it's not necessarily our bug).
                        pybind11::globals().clear();
                    }
                    catch ( ... )
                    {
                        spdlog::error( "Unable to reset the global variables after the script because of an exception." );
                    }
                }

                { // Signal that we're done.
                    if ( onDoneAsync_ )
                    {
                        onDoneAsync_( lastRunSuccessful_ );
                        onDoneAsync_ = nullptr;
                        state_ = State::idle;
                    }
                    else
                    {
                        state_ = State::finishing;
                    }
                    cv_.notify_all();
                }
            }

            pybind11::finalize_interpreter();
        } );

        return nullptr;
    }();
}

EmbeddedPython& EmbeddedPython::instance_()
{
    static EmbeddedPython instance;
    return instance;
}

} //namespace MR
