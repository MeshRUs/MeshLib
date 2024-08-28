#pragma once

#include "MRMacros.h"
#include "MRMeshFwd.h"
#include <type_traits>

namespace MR
{

/// affine transformation: y = A*x + b, where A in VxV, and b in V
/// \ingroup MathGroup
template <typename V>
struct AffineXf
{
    using T = typename V::ValueType;
    using M = typename V::MatrixType;

    M A;
    V b;

    constexpr AffineXf() noexcept = default;
    constexpr AffineXf( const M & A, const V & b ) noexcept : A( A ), b( b ) { }
    template <typename U>
    constexpr explicit AffineXf( const AffineXf<U> & xf ) noexcept : A( xf.A ), b( xf.b ) { }
    /// creates translation-only transformation (with identity linear component)
    [[nodiscard]] static constexpr AffineXf translation( const V & b ) noexcept { return AffineXf{ M{}, b }; }
    /// creates linear-only transformation (without translation)
    [[nodiscard]] static constexpr AffineXf linear( const M & A ) noexcept { return AffineXf{ A, V{} }; }
    /// creates transformation with given linear part with given stable point
    [[nodiscard]] static constexpr AffineXf xfAround( const M & A, const V & stable ) noexcept { return AffineXf{ A, stable - A * stable }; }

    /// application of the transformation to a point
    [[nodiscard]] constexpr V operator() ( const V & x ) const noexcept { return A * x + b; }
    /// applies only linear part of the transformation to given vector (e.g. to normal) skipping adding shift (b)
    /// for example if this is a rigid transformation, then only rotates input vector
    [[nodiscard]] constexpr V linearOnly( const V & x ) const noexcept { return A * x; }
    /// computes inverse transformation
    [[nodiscard]] constexpr AffineXf inverse() const noexcept MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> );

    /// composition of two transformations:
    /// \f( y = (u * v) ( x ) = u( v( x ) ) = ( u.A * ( v.A * x + v.b ) + u.b ) = ( u.A * v.A ) * x + ( u.A * v.b + u.b ) \f)
    friend AffineXf<V> operator * ( const AffineXf<V> & u, const AffineXf<V> & v )
        { return { u.A * v.A, u.A * v.b + u.b }; }

    ///
    friend bool operator == ( const AffineXf<V> & a, const AffineXf<V> & b )
    {
        return a.A == b.A && a.b == b.b;
    }

    ///
    friend bool operator != ( const AffineXf<V> & a, const AffineXf<V> & b )
    {
        return !( a == b );
    }
};

/// \related AffineXf
/// \{

template <typename V>
inline constexpr AffineXf<V> AffineXf<V>::inverse() const noexcept MR_REQUIRES_IF_SUPPORTED( !std::is_integral_v<T> )
{
    AffineXf<V> res;
    res.A = A.inverse();
    res.b = -( res.A * b );
    return res;
}

/// \}

} // namespace MR
