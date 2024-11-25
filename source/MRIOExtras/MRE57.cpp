#include "MRE57.h"
#ifndef MRIOEXTRAS_NO_E57
#include <MRMesh/MRBox.h>
#include <MRMesh/MRIOFormatsRegistry.h>
#include <MRMesh/MRObjectPoints.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRProgressCallback.h>
#include <MRMesh/MRStringConvert.h>
#include <MRMesh/MRQuaternion.h>
#include <MRMesh/MRTimer.h>
#include <MRPch/MRFmt.h>

#include <climits>

#pragma warning(push)
#pragma warning(disable: 4251) // class needs to have dll-interface to be used by clients of another class
#pragma warning(disable: 4275) // vcpkg `2022.11.14`: non dll-interface class 'std::exception' used as base for dll-interface class 'e57::E57Exception'
#include <E57Format/E57SimpleReader.h>
#if !__has_include(<E57Format/E57Version.h>)
#define  MR_OLD_E57
#endif

#pragma warning(pop)

namespace MR::PointsLoad
{

Expected<std::vector<NamedCloud>> fromSceneE57File( const std::filesystem::path& file, const E57LoadSettings & settings )
{
    MR_TIMER
    std::vector<NamedCloud> res;
    std::optional<AffineXf3d> xf0; // returned transformation of the first not-empty cloud
    if ( settings.identityXf )
        xf0.emplace();

    try
    {
#ifdef MR_OLD_E57
        e57::Reader eReader( utf8string( file ) );
#else
        e57::Reader eReader( utf8string( file ), {} );
#endif
        const auto numScans = eReader.GetData3DCount();
        res.resize( numScans );
        for ( int scanIndex = 0; scanIndex < numScans; ++scanIndex )
        {
            auto sp = subprogress( settings.progress, float( scanIndex ) / numScans, float( scanIndex + 1 ) / numScans );
            auto & nc = res[scanIndex];
            e57::Data3D scanHeader;
            eReader.ReadData3D( scanIndex, scanHeader );
            nc.name = scanHeader.name;
            const AffineXf3d e57Xf(
                Quaterniond( scanHeader.pose.rotation.w, scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z ),
                Vector3d( scanHeader.pose.translation.x, scanHeader.pose.translation.y, scanHeader.pose.translation.z )
            );
            const bool sphericalCoords = scanHeader.pointFields.sphericalRangeField
                && scanHeader.pointFields.sphericalAzimuthField
                && scanHeader.pointFields.sphericalElevationField;
            assert( sphericalCoords || ( scanHeader.pointFields.cartesianXField
                && scanHeader.pointFields.cartesianYField
                && scanHeader.pointFields.cartesianZField ) );

            std::optional<AffineXf3d> aXf; // will be applied to all points
            if ( settings.identityXf )
                aXf = e57Xf;
            else if ( settings.combineAllObjects && xf0 )
                aXf = xf0->inverse() * e57Xf;

            if ( !aXf )
            {
                if ( sphericalCoords )
                    aXf = AffineXf3d();
                else
                {
                    const auto& bounds = scanHeader.cartesianBounds;
                    const Box3d box {
                        { bounds.xMinimum, bounds.yMinimum, bounds.zMinimum },
                        { bounds.xMaximum, bounds.yMaximum, bounds.zMaximum },
                    };
                    if ( box.valid() )
                    {
                        if ( box.contains( Vector3d() ) ) // if zero of space is within bounding box (e.g. the position of camera capturing 360 degrees around),
                            aXf = AffineXf3d();           // then keep point coordinates as is
                        else
                            aXf = AffineXf3d::translation( -box.center() ); // otherwise shift all points for the center of bounding box to receive zero coordinates
                    }
                }
            }

            int64_t nColumn = 0;
            int64_t nRow = 0;
            int64_t nPointsSize = 0;
            int64_t nGroupsSize = 0;
            int64_t nCountSize = 0;
            bool bColumnIndex = false;

            if ( !eReader.GetData3DSizes( scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex) )
                return MR::unexpected( std::string( "GetData3DSizes failed during reading of " + utf8string( file ) ) );

            if ( nPointsSize > INT_MAX )
                return MR::unexpected( fmt::format( "Too many points {} in {}.\nMaximum supported is {}.", nPointsSize, utf8string( file ), INT_MAX ) );

            // how many points to read in a time
            const int64_t nSize = std::min( nPointsSize, int64_t( 1024 ) * 128 );

    #ifdef MR_OLD_E57
            e57::Data3DPointsData_d buffers;
    #else
            e57::Data3DPointsDouble buffers;
    #endif

            std::vector<double> xs, ys, zs;
            std::vector<double> rgs, azs, els;
            if ( sphericalCoords )
            {
                rgs.resize( nSize );
                azs.resize( nSize );
                els.resize( nSize );
                buffers.sphericalRange = rgs.data();
                buffers.sphericalAzimuth = azs.data();
                buffers.sphericalElevation = els.data();
            }
            else
            {
                xs.resize( nSize );
                ys.resize( nSize );
                zs.resize( nSize );
                buffers.cartesianX = xs.data();
                buffers.cartesianY = ys.data();
                buffers.cartesianZ = zs.data();
            }

    #ifdef MR_OLD_E57
            std::vector<uint8_t> rs, gs, bs;
    #else
            std::vector<uint16_t> rs, gs, bs;
    #endif
            std::vector<int8_t> invalidColors;
            rs.resize( nSize );
            gs.resize( nSize );
            bs.resize( nSize );
            buffers.colorRed = rs.data();
            buffers.colorGreen = gs.data();
            buffers.colorBlue = bs.data();
            invalidColors.resize( nSize );
            buffers.isColorInvalid = invalidColors.data();

            e57::CompressedVectorReader dataReader = eReader.SetUpData3DPointsData( scanIndex, nSize, buffers );

            auto & cloud = nc.cloud;
            auto & colors = nc.colors;
            cloud.points.reserve( nPointsSize );
            unsigned long size = 0;
            bool hasInputColors = false;
            while ( ( size = dataReader.read() ) > 0 )
            {
                reportProgress( sp, float( cloud.points.size() ) / nPointsSize );
                if ( cloud.points.empty() )
                {
                    hasInputColors = invalidColors.front() == 0;
                    if ( hasInputColors )
                        colors.reserve( nPointsSize );
                }
                if ( !aXf )
                {
                    aXf = AffineXf3d::translation(
                    {
                        -buffers.cartesianX[0],
                        -buffers.cartesianY[0],
                        -buffers.cartesianZ[0],
                    } );
                }
                for ( unsigned long i = 0; i < size; ++i )
                {
                    Vector3d p;
                    if ( sphericalCoords )
                    {
                        const auto r = buffers.sphericalRange[i];
                        const auto a = buffers.sphericalAzimuth[i];
                        const auto e = buffers.sphericalElevation[i];
                        p.x = r * std::cos( e ) * std::cos( a );
                        p.y = r * std::cos( e ) * std::sin( a );
                        p.z = r * std::sin( e );
                    }
                    else
                        p = Vector3d( buffers.cartesianX[i], buffers.cartesianY[i], buffers.cartesianZ[i] );
                    cloud.points.emplace_back( Vector3f( (*aXf)( p ) ) );
                    if ( hasInputColors )
                    {
                        colors.emplace_back(
                            buffers.colorRed[i],
                            buffers.colorGreen[i],
                            buffers.colorBlue[i]
                        );
                    }
                }
            }

            assert( cloud.points.size() == (size_t)nPointsSize );
            cloud.validPoints.resize( cloud.points.size(), true );

            dataReader.close();
            nc.xf = ( settings.identityXf || !aXf ) ? AffineXf3f() :
                AffineXf3f( e57Xf * aXf->inverse() );
            if ( !xf0 && aXf )
                xf0 = e57Xf * aXf->inverse();
        }
    }
    catch( const e57::E57Exception & e )
    {
        return MR::unexpected( fmt::format( "Error '{}' during reading of {}",
            e57::Utilities::errorCodeToString( e.errorCode() ), utf8string( file ) ) );
    }

    if ( !settings.combineAllObjects || res.size() <= 1 )
        return res;

    size_t totalPoints = 0;
    bool keepColors = true;

    for ( int i = 0; i < res.size(); ++i )
    {
        totalPoints += res[i].cloud.points.size();
        keepColors = keepColors && res[i].cloud.points.size() == res[i].colors.size();
    }

    res[0].cloud.points.reserve( totalPoints );
    if ( keepColors )
        res[0].colors.reserve( totalPoints );
    else
        res[0].colors = {};

    for ( int i = 1; i < res.size(); ++i )
    {
        res[0].cloud.points.vec_.insert( end( res[0].cloud.points ), begin( res[i].cloud.points ), end( res[i].cloud.points ) );
        if ( keepColors )
            res[0].colors.vec_.insert( end( res[0].colors ), begin( res[i].colors ), end( res[i].colors ) );
    }

    res.resize( 1 );
    assert( res[0].cloud.points.size() == totalPoints );
    assert( !keepColors || res[0].colors.size() == totalPoints );
    res[0].cloud.validPoints.resize( res[0].cloud.points.size(), true );
    if ( xf0 )
        res[0].xf = AffineXf3f( *xf0 );

    return res;
}

Expected<PointCloud> fromE57( const std::filesystem::path& file, const PointsLoadSettings& settings )
{
    auto x = fromSceneE57File( file, { .combineAllObjects = true, .identityXf = !settings.outXf, .progress = settings.callback } );
    if ( !x )
        return unexpected( std::move( x.error() ) );
    if ( x->empty() )
        return PointCloud();
    assert( x->size() == 1 );
    if ( settings.colors )
        *settings.colors = std::move( (*x)[0].colors );
    if ( settings.outXf )
        *settings.outXf = (*x)[0].xf;
    return std::move( (*x)[0].cloud );
}

Expected<PointCloud> fromE57( std::istream&, const PointsLoadSettings& )
{
    return unexpected( "no support for reading e57 from arbitrary stream yet" );
}

MR_ADD_POINTS_LOADER( IOFilter( "E57 (.e57)", "*.e57" ), fromE57 )

Expected<LoadedObjects> loadObjectFromE57( const std::filesystem::path& path, const ProgressCallback& cb )
{
    return fromSceneE57File( path, { .progress = std::move( cb ) } )
    .transform( [&path] ( std::vector<NamedCloud>&& nclouds )
    {
        LoadedObjects res;
        res.objs.resize( nclouds.size() );
        for ( int i = 0; i < res.objs.size(); ++i )
        {
            auto objectPoints = std::make_shared<ObjectPoints>();
            if ( nclouds[i].name.empty() || nclouds.size() <= 1 ) // if only one cloud in file, then take name from file
                objectPoints->setName( utf8string( path.stem() ) );
            else
                objectPoints->setName( std::move( nclouds[i].name ) );
            objectPoints->select( true );
            objectPoints->setPointCloud( std::make_shared<PointCloud>( std::move( nclouds[i].cloud ) ) );
            objectPoints->setXf( nclouds[i].xf );
            if ( !nclouds[i].colors.empty() )
            {
                objectPoints->setVertsColorMap( std::move( nclouds[i].colors ) );
                objectPoints->setColoringType( ColoringType::VertsColorMap );
            }
            res.objs[i] = std::dynamic_pointer_cast< Object >( std::move( objectPoints ) );
        }
        return res;
    } );
}

MR_ADD_OBJECT_LOADER( IOFilter( "E57 (.e57)", "*.e57" ), loadObjectFromE57 )

} //namespace MR::PointsLoad
#endif
