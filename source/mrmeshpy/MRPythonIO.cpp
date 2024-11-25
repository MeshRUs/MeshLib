#include "MRPython/MRPython.h"
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include "MRMesh/MRObjectsAccess.h"
#include "MRMesh/MRSceneRoot.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMeshSave.h"
#include "MRMesh/MRLinesSave.h"
#include "MRMesh/MRLinesLoad.h"
#include "MRMesh/MRPointsSave.h"
#include "MRMesh/MRPointsLoad.h"
#include "MRMesh/MRObjectLoad.h"
#include "MRMesh/MRMeshLoad.h"
#include "MRMesh/MRSerializer.h"
#include "MRMesh/MRLog.h"
#include "MRMesh/MRExpected.h"
#include "MRMesh/MRSceneLoad.h"
#include "MRMesh/MRObjectSave.h"
#include "MRMesh/MROnInit.h"
#include "MRIOExtras/MRIOExtras.h"
#pragma warning(push)
#pragma warning(disable: 4464) // relative include path contains '..'
#include <pybind11/stl/filesystem.h>
#pragma warning(pop)

#ifndef MESHLIB_NO_VOXELS
#include "MRVoxels/MRVoxelsSave.h"
#include "MRVoxels/MRVoxelsLoad.h"
#ifndef MRVOXELS_NO_DICOM
#include "MRVoxels/MRDicom.h"
#endif
#endif

using namespace MR;

namespace MR
{

class PythonIstreamBuf : public std::streambuf
{
public:
    PythonIstreamBuf( pybind11::object inFileHandle ) :
        pyseek_( inFileHandle.attr( "seek" ) ),
        pytell_( inFileHandle.attr( "tell" ) ),
        pyread_( inFileHandle.attr( "read" ) )
    {
        size_ = pyseek_( 0, 2 ).cast<std::streamsize>();
        pyseek_( 0 );
    };

    virtual std::streamsize showmanyc() override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        return size_ - currentPos;
    }

    virtual std::streamsize xsgetn( char* elem, std::streamsize count ) override
    {
        count = std::min( showmanyc(), count );
        if ( count == 0 )
            return 0;
        std::string readBytes = pyread_( count ).cast<std::string>();
        std::copy( readBytes.c_str(), readBytes.c_str() + count, elem );
        return count;
    }

    virtual int_type underflow() override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        auto res = uflow();
        if ( res == std::streambuf::traits_type::eof() )
            return std::streambuf::traits_type::eof();
        pyseek_( currentPos );
        return res;
    }

    virtual int_type uflow() override
    {
        char c;
        auto numRead = xsgetn( &c, 1 );
        if ( numRead == 0 )
            return std::streambuf::traits_type::eof();
        return std::streambuf::traits_type::to_int_type( c );
    }

    virtual pos_type seekoff( off_type off, std::ios_base::seekdir way, std::ios_base::openmode ) override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        std::streamsize reqPos = currentPos + off;
        if ( way == std::ios_base::beg )
            reqPos = off;
        else if ( way == std::ios_base::end )
            reqPos = size_ + off;
        pyseek_( reqPos );
        return reqPos;
    }

    virtual pos_type seekpos( pos_type pos, std::ios_base::ios_base::openmode ) override
    {
        pyseek_( std::streamsize( pos ) );
        return pytell_().cast<std::streamsize>();
    }

    virtual int_type pbackfail( int_type c ) override
    {
        std::streamsize currentPos = pytell_().cast<std::streamsize>();
        if ( currentPos == 0 )
            return std::streambuf::traits_type::eof();
        pyseek_( currentPos - 1 );
        return c;
    }

private:
    pybind11::object pyseek_;
    pybind11::object pytell_;
    pybind11::object pyread_;
    std::streamsize size_;
};

// Buffer that writes in Python instead of C++
class PythonOstreamBuf : public std::stringbuf
{
public:
    PythonOstreamBuf( pybind11::object outFileHandle ) :
        pywrite_( outFileHandle.attr( "write" ) ),
        pyflush_( outFileHandle.attr( "flush" ) )
    {
    }
    ~PythonOstreamBuf()
    {
        sync_();
    }
    int sync() override
    {
        sync_();
        return 0;
    }

private:
    pybind11::object pywrite_;
    pybind11::object pyflush_;
    void sync_()
    {
        pybind11::bytes bytes = pybind11::bytes( this->str() );
        pywrite_( bytes );
        pyflush_();
    }
};

}

Expected<MR::Mesh> pythonLoadMeshFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    return MR::MeshLoad::fromAnySupportedFormat( ifs, extension );
}

Expected<void> pythonSaveMeshToAnyFormat( const Mesh& mesh, const std::string& extension, pybind11::object fileHandle )
{
    if ( !( pybind11::hasattr( fileHandle, "write" ) && pybind11::hasattr( fileHandle, "flush" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonOstreamBuf pybuf( fileHandle );
    std::ostream outfs( &pybuf );
    return MR::MeshSave::toAnySupportedFormat( mesh, extension, outfs );
}

Expected<void> pythonSaveLinesToAnyFormat( const MR::Polyline3& lines, const std::string& extension, pybind11::object fileHandle )
{
    if ( !( pybind11::hasattr( fileHandle, "write" ) && pybind11::hasattr( fileHandle, "flush" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonOstreamBuf pybuf( fileHandle );
    std::ostream outfs( &pybuf );
    return MR::LinesSave::toAnySupportedFormat( lines, extension, outfs );
}

Expected<Polyline3> pythonLoadLinesFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    return MR::LinesLoad::fromAnySupportedFormat( ifs, extension );
}

Expected<void> pythonSavePointCloudToAnyFormat( const PointCloud& points, const std::string& extension, pybind11::object fileHandle )
{
    if ( !( pybind11::hasattr( fileHandle, "write" ) && pybind11::hasattr( fileHandle, "flush" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonOstreamBuf pybuf( fileHandle );
    std::ostream outfs( &pybuf );
    return MR::PointsSave::toAnySupportedFormat( points, extension, outfs );
}

Expected<PointCloud> pythonLoadPointCloudFromAnyFormat( pybind11::object fileHandle, const std::string& extension )
{
    if ( !( pybind11::hasattr( fileHandle, "read" ) && pybind11::hasattr( fileHandle, "seek" ) && pybind11::hasattr( fileHandle, "tell" ) ) )
        return unexpected( "Argument is not file handle" );
    PythonIstreamBuf streambuf( fileHandle );
    std::istream ifs( &streambuf );
    return MR::PointsLoad::fromAnySupportedFormat( ifs, extension );
}

Expected<std::shared_ptr<Object>> pythonLoadSceneObjectFromAnyFormat( const std::filesystem::path& path, ProgressCallback callback )
{
    auto result = SceneLoad::fromAnySupportedFormat( { path }, std::move( callback ) );
    if ( !result.scene || !result.errorSummary.empty() )
        return unexpected( std::move( result.errorSummary ) );

    if ( !result.isSceneConstructed || result.scene->children().size() != 1 )
        return result.scene;
    else
        return result.scene->children().front();
}

Expected<void> pythonSaveSceneObjectToAnySupportedFormat( const std::shared_ptr<Object>& object, const std::filesystem::path& path, ProgressCallback callback )
{
    return ObjectSave::toAnySupportedFormat( *object, path, std::move( callback ) );
}

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveMesh, [] ( pybind11::module_& m )
{
    m.def( "saveMesh",
        MR::decorateExpected( []( const MR::Mesh& m, const std::filesystem::path& p, const VertColors* cs, ProgressCallback cb )
            { return MR::MeshSave::toAnySupportedFormat( m, p, { .colors = cs, .progress = cb } ); } ),
        pybind11::arg( "mesh" ), pybind11::arg( "path" ), pybind11::arg( "colors" ) = nullptr, pybind11::arg( "callback" ) = ProgressCallback{},
        "detects the format from file extension and save mesh to it" );
    m.def( "saveMesh",
        MR::decorateExpected( ( Expected<void>( * )( const MR::Mesh&, const std::string&, pybind11::object ) )& pythonSaveMeshToAnyFormat ),
        pybind11::arg( "mesh" ), pybind11::arg( "extension" ), pybind11::arg( "fileHandle" ), "saves mesh in python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadMesh, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::MeshLoadSettings>( m, "MeshLoadSettings", "mesh load settings" ).
        def( pybind11::init<>() ).
        def_readwrite( "colors", &MR::MeshLoadSettings::colors ).
        def_readwrite( "uvCoords", &MR::MeshLoadSettings::uvCoords ).
        def_readwrite( "normals", &MR::MeshLoadSettings::normals ).
        //def_readwrite( "texture", &MR::MeshLoadSettings::texture ). // MeshTexture is not exposed yet
        def_readwrite( "skippedFaceCount", &MR::MeshLoadSettings::skippedFaceCount ).
        def_readwrite( "duplicatedVertexCount", &MR::MeshLoadSettings::duplicatedVertexCount ).
        def_readwrite( "xf", &MR::MeshLoadSettings::xf ).
        def_readwrite( "callback", &MR::MeshLoadSettings::callback );

    m.def( "loadMesh",
        MR::decorateExpected( ( Expected<MR::Mesh>( * )( const std::filesystem::path&, const MeshLoadSettings& ) )& MR::MeshLoad::fromAnySupportedFormat),
        pybind11::arg( "path" ), pybind11::arg_v( "settings", MeshLoadSettings(), "MeshLoadSettings()" ),
        "detects the format from file extension and loads mesh from it" );
    m.def( "loadMesh",
        MR::decorateExpected( ( Expected<MR::Mesh>( * )( pybind11::object, const std::string& ) )& pythonLoadMeshFromAnyFormat ),
        pybind11::arg( "fileHandle" ), pybind11::arg( "extension" ), "load mesh from python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveLines, [] ( pybind11::module_& m )
{
    m.def( "saveLines",
        MR::decorateExpected( []( const MR::Polyline3& pl, const std::filesystem::path& p, ProgressCallback cb )
            { return MR::LinesSave::toAnySupportedFormat( pl, p, { .progress = cb } ); } ),
        pybind11::arg( "polyline" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "detects the format from file extension and saves polyline in it" );
    m.def( "saveLines",
        MR::decorateExpected( ( Expected<void>( * )( const MR::Polyline3&, const std::string&, pybind11::object ) )& pythonSaveLinesToAnyFormat ),
        pybind11::arg( "polyline" ), pybind11::arg( "extension" ), pybind11::arg( "fileHandle" ), "saves lines in python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadLines, [] ( pybind11::module_& m )
{
    m.def( "loadLines",
        MR::decorateExpected( ( Expected<Polyline3>( * )( const std::filesystem::path&, ProgressCallback ) )& MR::LinesLoad::fromAnySupportedFormat ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "detects the format from file extension and loads polyline from it" );
    m.def( "loadLines",
        MR::decorateExpected( ( Expected<Polyline3>( * )( pybind11::object, const std::string& ) )& pythonLoadLinesFromAnyFormat) ,
        pybind11::arg( "fileHandle" ), pybind11::arg( "extension" ), "load lines from python file handler, second arg: extension (`*.ext` format)" );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SavePoints, [] ( pybind11::module_& m )
{
    m.def( "savePoints",
        MR::decorateExpected( []( const PointCloud& cloud, const std::filesystem::path& file, const VertColors* colors, ProgressCallback callback )
            { return MR::PointsSave::toAnySupportedFormat( cloud, file, { .colors = colors, .progress = callback } ); } ),
        pybind11::arg( "pointCloud" ), pybind11::arg( "path" ), pybind11::arg( "colors" ) = nullptr, pybind11::arg( "callback" ) = ProgressCallback{},
        "detects the format from file extension and save points to it" );
    m.def( "savePoints",
        MR::decorateExpected( ( Expected<void>( * )( const MR::PointCloud&, const std::string&, pybind11::object ) )& pythonSavePointCloudToAnyFormat ),
        pybind11::arg( "pointCloud" ), pybind11::arg( "extension" ), pybind11::arg( "fileHandle" ), "saves point cloud in python file handler, second arg: extension (`*.ext` format)" );
} )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadPoints, [] ( pybind11::module_& m )
{
    pybind11::class_<MR::PointsLoadSettings>( m, "PointsLoadSettings", "points load settings" ).
        def( pybind11::init<>() ).
        def_readwrite( "colors", &MR::PointsLoadSettings::colors ).
        def_readwrite( "outXf", &MR::PointsLoadSettings::outXf ).
        def_readwrite( "callback", &MR::PointsLoadSettings::callback );

    m.def( "loadPoints",
        MR::decorateExpected( ( Expected<PointCloud>( * )( const std::filesystem::path&, const PointsLoadSettings& ) )& MR::PointsLoad::fromAnySupportedFormat ),
        pybind11::arg( "path" ), pybind11::arg_v( "settings", PointsLoadSettings(), "PointsLoadSettings()" ),
        "detects the format from file extension and loads points from it" );
    m.def( "loadPoints",
        MR::decorateExpected( ( Expected<PointCloud>( * )( pybind11::object, const std::string& ) )& pythonLoadPointCloudFromAnyFormat ),
        pybind11::arg( "fileHandle" ), pybind11::arg( "extension" ), "load point cloud from python file handler, second arg: extension (`*.ext` format)" );
} )

#ifndef MESHLIB_NO_VOXELS
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveVoxels, [] ( pybind11::module_& m )
{
    m.def( "saveVoxels",
        MR::decorateExpected( &MR::VoxelsSave::toAnySupportedFormat ),
        pybind11::arg( "vdbVoxels" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Saves voxels in a file, detecting the format from file extension." );
    m.def( "saveVoxelsRaw",
        MR::decorateExpected( static_cast<Expected<void> ( * )( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback )>( &MR::VoxelsSave::toRawAutoname ) ),
        pybind11::arg( "vdbVoxels" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Save raw voxels file, writing parameters in name." );
    m.def( "saveVoxelsGav",
        MR::decorateExpected( static_cast<Expected<void> ( * )( const VdbVolume& vdbVolume, const std::filesystem::path& file, ProgressCallback callback )>( &MR::VoxelsSave::toGav ) ),
        pybind11::arg( "vdbVoxels" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Save Gav voxels file." );
    m.def( "saveVoxelsVdb",
        MR::decorateExpected( &MR::VoxelsSave::toVdb ),
        pybind11::arg( "vdbVoxels" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Save voxels file in OpenVDB format." );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadVoxels, [] ( pybind11::module_& m )
{
    m.def( "loadVoxels",
        MR::decorateExpected( &MR::VoxelsLoad::fromAnySupportedFormat ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Detects the format from file extension and loads voxels from it. The older version of this function is avaiable now as loadVoxelsRaw." );
    m.def( "loadVoxelsRaw",
        MR::decorateExpected( static_cast<Expected<VdbVolume>( * )( const std::filesystem::path&, const ProgressCallback& )>( &MR::VoxelsLoad::fromRaw ) ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Load raw voxels file, parsing parameters from name." );
    m.def( "loadVoxelsGav",
        MR::decorateExpected( static_cast<Expected<VdbVolume>( * )( const std::filesystem::path&, const ProgressCallback& )>( &MR::VoxelsLoad::fromGav ) ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Load Gav voxels file, parsing parameters from name." );
    m.def( "loadVoxelsVdb",
        MR::decorateExpected( &MR::VoxelsLoad::fromVdb ),
        pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback{},
        "Load all voxel volumes from OpenVDB file." );

} )

#ifndef MRVOXELS_NO_DICOM
MR_ADD_PYTHON_CUSTOM_CLASS( mrmeshpy, DicomVolumeAsVdb, MR::VoxelsLoad::DicomVolumeAsVdb )
MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, DicomVolumeAsVdb, [] ( pybind11::module_& )
{
    MR_PYTHON_CUSTOM_CLASS( DicomVolumeAsVdb ).
        def_readwrite( "vdbVolume", &MR::VoxelsLoad::DicomVolumeAsVdb::vdbVolume ).
        def_readwrite( "name", &MR::VoxelsLoad::DicomVolumeAsVdb::name ).
        def_readwrite( "xf", &MR::VoxelsLoad::DicomVolumeAsVdb::xf );
} )

MR_ADD_PYTHON_VEC( mrmeshpy, LoadDCMResults, MR::VoxelsLoad::DicomVolumeAsVdb )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadVoxelsDicom, [] ( pybind11::module_& m )
{
    m.def( "loadDicomFolderAsVdb", MR::decorateExpected( &MR::VoxelsLoad::loadDicomFolderAsVdb ),
        pybind11::arg( "path" ), pybind11::arg( "maxNumThreads" ) = 4, pybind11::arg( "callback" ) = ProgressCallback{},
        "Loads first volumetric data from DICOM file(s)" );

    m.def( "loadDicomsFolderAsVdb",
        [] ( const std::filesystem::path& p, unsigned maxNumThreads, const ProgressCallback& cb)
    {
        auto res = MR::VoxelsLoad::loadDicomsFolderAsVdb( p, maxNumThreads, cb );
        std::vector<MR::VoxelsLoad::DicomVolumeAsVdb> resVec;
        std::string accumError;
        for ( auto& r : res )
        {
            if ( r.has_value() )
                resVec.push_back( std::move( *r ) );
            else
                accumError += ( r.error() + "\n" );
        }
        if ( resVec.empty() )
            throwExceptionFromExpected( accumError );
        return resVec;
    },
        pybind11::arg( "path" ), pybind11::arg( "maxNumThreads" ) = 4, pybind11::arg( "callback" ) = ProgressCallback{},
        "Loads all volumetric data from DICOM file(s)" );
} )
#endif
#endif

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, LoadSceneObject, [] ( pybind11::module_& m )
{
    m.def( "loadSceneObject",
           MR::decorateExpected( pythonLoadSceneObjectFromAnyFormat ),
           pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback(),
           "Detects the format from file extension and loads scene object from it." );
} )

MR_ADD_PYTHON_CUSTOM_DEF( mrmeshpy, SaveSceneObject, [] ( pybind11::module_& m )
{
    m.def( "saveSceneObject",
           MR::decorateExpected( pythonSaveSceneObjectToAnySupportedFormat ),
           pybind11::arg( "object" ), pybind11::arg( "path" ), pybind11::arg( "callback" ) = ProgressCallback(),
           "Detects the format from file extension and saves scene object to it. "
           "If the object doesn't contain any entities of the corresponding type, an empty file will be created." );
} )

// force load MRIOExtras library to load extra file formats
MR_ON_INIT { MR::loadIOExtras(); };
