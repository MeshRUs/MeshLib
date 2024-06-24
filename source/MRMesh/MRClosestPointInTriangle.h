// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //
#pragma once

#include "MRVector3.h"
#include "MRTriPoint.h"

namespace MR
{

template <typename T>
static std::pair<Vector3<T>, TriPoint<T>> closestPointInTriangle( const Vector3<T>& p, const Vector3<T>& a, const Vector3<T>& b, const Vector3<T>& c )
{
    // https://stackoverflow.com/a/74395029/7325599
    const Vector3<T> ab = b - a;
    const Vector3<T> ac = c - a;
    const Vector3<T> ap = p - a;

    const T d1 = dot( ab, ap );
    const T d2 = dot( ac, ap );
    if ( d1 <= 0 && d2 <= 0 ) 
        return { a, { 0, 0 } }; //#1

    const Vector3<T> bp = p - b;
    const T d3 = dot( ab, bp );
    const T d4 = dot( ac, bp );
    if ( d3 >= 0 && d4 <= d3 ) 
        return { b, { 1, 0 } }; //#2

    const Vector3<T> cp = p - c;
    const T d5 = dot( ab, cp );
    const T d6 = dot( ac, cp );
    if ( d6 >= 0 && d5 <= d6 ) 
        return { c, { 0, 1 } }; //#3

    const T vc = d1 * d4 - d3 * d2;
    if ( vc <= 0 && d1 >= 0 && d3 <= 0 )
    {
        const T v = d1 / ( d1 - d3 );
        return { a + v * ab, { v, 0 } }; //#4
    }

    const T vb = d5 * d2 - d1 * d6;
    if ( vb <= 0 && d6 <= 0 )
    {
        assert( d2 >= 0 );
        const T v = d2 / ( d2 - d6 );
        return { a + v * ac, { 0, v } }; //#5
    }

    const T va = d3 * d6 - d5 * d4;
    if ( va <= 0 )
    {
        // d4-d3 = dot(bc,bp) >= 0 in #6
        if ( d4 < d3 ) // floating-point rounding errors
            return { b, { 1, 0 } }; //#2

        // d5-d6 = dot(cb,cp) >= 0 in #6
        if ( d5 < d6 ) // floating-point rounding errors
            return { c, { 0, 1 } }; //#3

        // ( d4 - d3 ) + ( d5 - d6 ) = bc^2 >= 0
        const T v = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
        return { b + v * ( c - b ), { 1 - v, v } }; //#6
    }

    assert( va > 0 && vb > 0 && vc > 0 );
    const T denom = 1 / ( va + vb + vc );
    const T v = vb * denom;
    const T w = vc * denom;
    return { a + v * ab + w * ac, { v, w } }; //#0
}

} //namespace MR
