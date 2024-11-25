#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRMeshFwd.h"
#include "MRVector3.h"

namespace MR
{

/// \defgroup ContourGroup Contour
/// \ingroup MathGroup
/// \{

/// >0 for clockwise loop, < 0 for CCW loop
/// \tparam R is the type for the accumulation and for result
template<typename T, typename R = T>
R calcOrientedArea( const Contour2<T> & contour )
{
    if ( contour.size() < 3 )
        return 0;

    R area = 0;
    Vector2<R> p0{ contour[0] };

    for ( int i = 2; i < contour.size(); ++i )
    {
        Vector2<R> p1{ contour[i - 1] };
        Vector2<R> p2{ contour[i] };
        area += cross( p2 - p0, p1 - p0 );
    }

    return R(0.5) * area;
}

/// returns the vector with the magnitude equal to contour area, and directed to see the contour
/// in ccw order from the vector tip
/// \tparam R is the type for the accumulation and for result
template<typename T, typename R = T>
Vector3<R> calcOrientedArea( const Contour3<T> & contour )
{
    if ( contour.size() < 3 )
        return {};

    Vector3<R> area;
    Vector3<R> p0{ contour[0] };

    for ( int i = 2; i < contour.size(); ++i )
    {
        Vector3<R> p1{ contour[i - 1] };
        Vector3<R> p2{ contour[i] };
        area += cross( p1 - p0, p2 - p0 );
    }

    return R(0.5) * area;
}

/// returns sum length of the given contour
/// \tparam R is the type for the accumulation and for result
template<typename V, typename R = typename V::ValueType>
R calcLength( const Contour<V>& contour )
{
    R l = R( 0 );
    for ( int i = 1; i < contour.size(); ++i )
        l += R( ( contour[i] - contour[i - 1] ).length() );
    return l;
}

/// copy double-contour to float-contour, or vice versa
template<typename To, typename From>
To copyContour( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & p : from )
        res.emplace_back( p );
    return res;
}

/// copy double-contours to float-contours, or vice versa
template<typename To, typename From>
To copyContours( const From & from )
{
    To res;
    res.reserve( from.size() );
    for ( const auto & c : from )
        res.push_back( copyContour<typename To::value_type>( c ) );
    return res;
}

/// \}

// Instantiate the templates when generating bindings.
MR_BIND_TEMPLATE( float calcOrientedArea( const Contour2<float> & contour ) )
MR_BIND_TEMPLATE( double calcOrientedArea( const Contour2<double> & contour ) )
MR_BIND_TEMPLATE( float calcLength( const Contour2<float>& contour ) )
MR_BIND_TEMPLATE( double calcLength( const Contour2<double>& contour ) )
MR_BIND_TEMPLATE( float calcLength( const Contour3<float>& contour ) )
MR_BIND_TEMPLATE( double calcLength( const Contour3<double>& contour ) )
MR_BIND_TEMPLATE( Vector3<float> calcOrientedArea( const Contour3<float> & contour ) )
MR_BIND_TEMPLATE( Vector3<double> calcOrientedArea( const Contour3<double> & contour ) )
MR_BIND_TEMPLATE( Contour2<float> copyContour( const Contour2<double> & from ) )
MR_BIND_TEMPLATE( Contour3<float> copyContour( const Contour3<double> & from ) )
MR_BIND_TEMPLATE( Contour2<double> copyContour( const Contour2<float> & from ) )
MR_BIND_TEMPLATE( Contour3<double> copyContour( const Contour3<float> & from ) )

} // namespace MR
