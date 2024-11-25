#pragma once

/**
 * NOTE: Include this header AFTER any OpenVDB header; otherwise it will break MSVC debug builds.
 *
 * External links:
 * - https://github.com/pybind/pybind11/blob/v2.9.2/include/pybind11/detail/common.h#L161
 * - https://github.com/oneapi-src/oneTBB/blob/v2021.9.0/include/oneapi/tbb/detail/_config.h#L105
 */

#include "exports.h"
#include "MRPybind11.h"

#include "MRMesh/MRExpected.h"
#include <functional>
#include <filesystem>
#include <unordered_map>

#define MR_INIT_PYTHON_MODULE( moduleName ) MR_INIT_PYTHON_MODULE_PRECALL( moduleName, [](){} )

#define MR_INIT_PYTHON_MODULE_PRECALL( moduleName, precall )\
PYBIND11_MODULE( moduleName, m )\
{\
    precall();\
    auto& adders = MR::PythonExport::instance().functions( #moduleName );\
    for ( auto& fs : adders )\
        for ( auto& f : fs )\
            f( m );\
}\
static MR::PythonFunctionAdder moduleName##_init_( #moduleName, &PyInit_##moduleName );

#define MR_ADD_PYTHON_FUNCTION( moduleName , name , func , description ) \
    static MR::PythonFunctionAdder name##_adder_( #moduleName, [](pybind11::module_& m){ m.def(#name, func, description);} );

#define MR_ADD_PYTHON_CUSTOM_DEF( moduleName , name , ... ) \
_Pragma("warning(push)") \
_Pragma("warning(disable:4459)") \
    static MR::PythonFunctionAdder name##_adder_( #moduleName, __VA_ARGS__ ); \
_Pragma("warning(pop)")

/// Python class wrapper name; for internal use only
#define _MR_PYTHON_CUSTOM_CLASS_HOLDER_NAME( name ) name##_class_

/// Python class wrapper accessor
/// \sa MR_ADD_PYTHON_CUSTOM_CLASS
#define MR_PYTHON_CUSTOM_CLASS( name ) ( *_MR_PYTHON_CUSTOM_CLASS_HOLDER_NAME( name ) )

/// Explicitly declare the Python class wrapper
/// \sa MR_ADD_PYTHON_CUSTOM_CLASS
#define MR_ADD_PYTHON_CUSTOM_CLASS_DECL( moduleName, name, ... ) \
static std::optional<pybind11::class_<__VA_ARGS__>> _MR_PYTHON_CUSTOM_CLASS_HOLDER_NAME( name );

/// Explicitly instantiate the Python class wrapper
/// \sa MR_ADD_PYTHON_CUSTOM_CLASS
#define MR_ADD_PYTHON_CUSTOM_CLASS_INST( moduleName, name ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name##_inst_, [] ( pybind11::module_& module ) \
{                                                                \
    _MR_PYTHON_CUSTOM_CLASS_HOLDER_NAME( name ).emplace( module, #name );            \
}, MR::PythonExport::Priority::Declaration )

/// Explicitly instantiate the Python class wrapper with a custom function
/// (pybind11::bind_vector, pybind11::bind_map, etc.)
/// \sa MR_ADD_PYTHON_CUSTOM_CLASS
/// \sa MR_ADD_PYTHON_VEC
/// \sa MR_ADD_PYTHON_MAP
#define MR_ADD_PYTHON_CUSTOM_CLASS_INST_FUNC( moduleName, name, ... ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name##_inst_, [] ( pybind11::module_& module ) \
{                                                                          \
    _MR_PYTHON_CUSTOM_CLASS_HOLDER_NAME( name ) = __VA_ARGS__ ( module );  \
}, MR::PythonExport::Priority::Declaration )

/**
 * For the proper stub file generation, a class must be defined prior to its depending methods.
 * To achieve it, declare the class *before* the \ref MR_ADD_PYTHON_CUSTOM_DEF block:
 *  - for simple cases use \ref MR_ADD_PYTHON_CUSTOM_CLASS macro;
 *  - to customize class declaration or instantiation (custom holder, custom container binding, etc.), use
 *    \ref MR_ADD_PYTHON_CUSTOM_CLASS_DECL and \ref MR_ADD_PYTHON_CUSTOM_CLASS_INST_FUNC macros
 * Finally replace the former class declaration within the \ref MR_ADD_PYTHON_CUSTOM_DEF with the
 * MR_PYTHON_CUSTOM_CLASS( class-name ) construction.
 * @code
 * MR_ADD_PYTHON_CUSTOM_CLASS( moduleName, pythonTypeName, cppTypeName )
 * MR_ADD_PYTHON_CUSTOM_DEF( moduleName, scopeName, [] ( pybind11::module_& m )
 * {
 *     MR_PYTHON_CUSTOM_CLASS( pythonTypeName )
 *         .doc() = "Class description";
 *     MR_PYTHON_CUSTOM_CLASS( pythonTypeName )
 *         .def( pybind11::init<>() );
 * }
 * @endcode
 * See also \ref MR_ADD_PYTHON_VEC and \ref MR_ADD_PYTHON_MAP macros for customized class definition examples.
 */
#define MR_ADD_PYTHON_CUSTOM_CLASS( moduleName, name, ... ) \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL( moduleName, name, __VA_ARGS__ ) \
MR_ADD_PYTHON_CUSTOM_CLASS_INST( moduleName, name )

#define MR_ADD_PYTHON_VEC( moduleName, name, type) \
PYBIND11_MAKE_OPAQUE( std::vector<type> )          \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL( moduleName, name, std::vector<type>, std::unique_ptr<std::vector<type>> ) \
MR_ADD_PYTHON_CUSTOM_CLASS_INST_FUNC( moduleName, name, [] ( pybind11::module_& module ) { return pybind11::bind_vector<std::vector<type>>( module, #name, pybind11::module_local(false) ); } ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name, [] ( pybind11::module_& )                                           \
{\
    using vecType = std::vector<type>;\
    MR_PYTHON_CUSTOM_CLASS( name ).\
        def( pybind11::init<>() ).\
        def( pybind11::init<size_t>(), pybind11::arg( "size" ) ).\
        def( "empty", &vecType::empty ).\
        def( "size", &vecType::size ).\
        def( "resize", ( void ( vecType::* )( const vecType::size_type ) )& vecType::resize ).\
        def( "clear", &vecType::clear ); \
} )

#define MR_ADD_PYTHON_MAP( moduleName, name, mapType ) \
PYBIND11_MAKE_OPAQUE( mapType )                        \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL( moduleName, name, mapType, std::unique_ptr<mapType> ) \
MR_ADD_PYTHON_CUSTOM_CLASS_INST_FUNC( moduleName, name, [] ( pybind11::module_& module ) { return pybind11::bind_map<mapType>( module, #name, pybind11::module_local(false) ); } ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name, [] ( pybind11::module_& )                       \
{\
    MR_PYTHON_CUSTOM_CLASS( name ).\
        def( pybind11::init<>() ).\
        def( "size", &mapType::size );\
} )

#define MR_ADD_PYTHON_EXPECTED( moduleName, name, type, errorType ) \
using name##_expected_type_ = MR::Expected<type, errorType>;        \
MR_ADD_PYTHON_CUSTOM_CLASS( moduleName, name, name##_expected_type_ ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name, [] ( pybind11::module_& )      \
{\
    using expectedType = Expected<type,errorType>;\
    MR_PYTHON_CUSTOM_CLASS( name ).\
        def( "has_value", []() \
        { PyErr_WarnEx(PyExc_DeprecationWarning, ".has_value is deprecated. Please use 'try - except ValueError'", 1); \
            return &expectedType::has_value; \
        }).\
        def( "value", []() \
        { \
            PyErr_WarnEx(PyExc_DeprecationWarning, ".value is deprecated. Please use 'try - except ValueError'", 1); \
            return ( type& ( expectedType::* )( )& )& expectedType::value; \
        }, pybind11::return_value_policy::reference_internal ).\
        def( "error", []() \
        { \
            PyErr_WarnEx(PyExc_DeprecationWarning, ".error is deprecated. Please use 'try - except ValueError'", 1); \
            return ( const errorType& ( expectedType::* )( )const& )& expectedType::error; \
        } );\
} )


enum StreamType
{
    Stdout,
    Stderr
};

template<StreamType T>
class PythonStreamRedirector
{
public:
    MRPYTHON_API void write( const std::string& text );
    MRPYTHON_API void flush();
    MRPYTHON_API static int getNumWritten();
};

using StdoutPyRedirector = PythonStreamRedirector<Stdout>;
using StderrPyRedirector = PythonStreamRedirector<Stderr>;

namespace MR
{

class PythonExport
{
public:
    MRPYTHON_API static PythonExport& instance();

    using PythonRegisterFuncton = std::function<void( pybind11::module_& m )>;

    enum class Priority
    {
        Declaration,
        Implementation,
        Count,
    };

    struct ModuleData
    {
        PyObject* ( *initFncPointer )( void );
        std::array<std::vector<PythonRegisterFuncton>, size_t( Priority::Count )> functions;
    };

    void addFunc( const std::string& moduleName, PythonRegisterFuncton func, Priority priority )
    {
        auto& mod = moduleData_[moduleName];
        mod.functions[size_t( priority )].push_back( func );
    }
    void setInitFuncPtr( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) )
    {
        auto& mod = moduleData_[moduleName];
        mod.initFncPointer = initFncPointer;
    }
    const std::array<std::vector<PythonRegisterFuncton>, size_t( Priority::Count )>& functions( const std::string& moduleName ) const
    {
        auto it = moduleData_.find( moduleName );
        const static std::array<std::vector<PythonRegisterFuncton>, size_t( Priority::Count )> empty;
        if ( it == moduleData_.end() )
            return empty;
        return it->second.functions;
    }

    const std::unordered_map<std::string, ModuleData>& modules() const
    {
        return moduleData_;
    }
private:
    PythonExport() = default;
    ~PythonExport() = default;

     std::unordered_map<std::string, ModuleData> moduleData_;
};

struct PythonFunctionAdder
{
    MRPYTHON_API PythonFunctionAdder( const std::string& moduleName, std::function<void( pybind11::module_& m )> func, PythonExport::Priority priority = PythonExport::Priority::Implementation );
    MRPYTHON_API PythonFunctionAdder( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) );
};

// overload `toString` functoion to throw exception from custom `Expected::error` type
template<typename E>
[[noreturn]] void throwExceptionFromExpected(const E& err)
{
    if constexpr (std::is_nothrow_convertible<E, std::string>::value)
        throw std::runtime_error(err);
    else
        throw std::runtime_error(toString(err));
}

// Like `e.value()`, but throws using `throwExceptionFromExpected` (which is better, because it allows Python to see the proper error message), and also supports `T == void`.
template <typename T>
[[nodiscard]] decltype(auto) expectedValueOrThrow( T&& e )
{
    if ( e )
    {
        if constexpr ( std::is_void_v<typename std::remove_cvref_t<T>::value_type> )
            return;
        else
            return *std::forward<T>( e );
    }
    else
    {
        throwExceptionFromExpected( e.error() );
    }
}

template<typename R, typename E, typename... Args>
auto decorateExpected( std::function<Expected<R, E>( Args... )>&& f ) -> std::function<R( Args... )>
{
    return[fLocal = std::move( f )]( Args&&... args ) mutable -> R
    {
        auto res = fLocal(std::forward<Args>( args )...);
        if (!res.has_value())
            throwExceptionFromExpected(res.error());

        if constexpr (std::is_void<R>::value)
            return;
        else
            return res.value();
    };
}

template<typename F>
auto decorateExpected( F&& f )
{
    return decorateExpected( std::function( std::forward<F>( f ) ) );
}

template<typename R, typename T, typename... Args>
auto decorateExpected( R( T::* memFunction )( Args... ) )
{
    return decorateExpected( std::function<R( T*, Args... )>( std::mem_fn( memFunction ) ) );
}

}
