#pragma once

#include "MRParabola.h"
#include "MRSymMatrix3.h"

namespace MR
{

/// accumulates a number of (x,y) points to find the best-least-squares parabola approximating them
template <typename T>
class BestFitParabola
{
public:
    /// accumulates one more point for parabola fitting
    void addPoint( T x, T y );

    /// accumulates one more point with given weight for parabola fitting
    void addPoint( T x, T y, T weight );

    /// computes the best approximating parabola from the accumulated points;
    Parabola<T> getBestParabola( T tol = std::numeric_limits<T>::epsilon() ) const;

private:
    SymMatrix3<T> m_;
    Vector3<T> b_;
};

template <typename T>
void BestFitParabola<T>::addPoint( T x, T y )
{
    const Vector3<T> v{ x*x, x, T(1) };
    m_ += outerSquare( v );
    b_ += y * v;
}

template <typename T>
void BestFitParabola<T>::addPoint( T x, T y, T weight )
{
    const Vector3<T> v{ x*x, x, T(1) };
    m_ += outerSquare( weight, v );
    b_ += weight * y * v;
}

template <typename T>
Parabola<T> BestFitParabola<T>::getBestParabola( T tol ) const
{
    auto res = m_.pseudoinverse( tol ) * b_;
    return { res[0], res[1], res[2] };
}

} //namespace MR
