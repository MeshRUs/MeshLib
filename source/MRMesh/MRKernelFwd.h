#pragma once

#include "exports.h"

#include "config.h"

// Not-zero _ITERATOR_DEBUG_LEVEL in Microsoft STL greatly reduces the performance of STL containers.
//
// Pre-build binaries from MeshLib distribution are prepared with _ITERATOR_DEBUG_LEVEL=0,
// and if you build MeshLib by yourself then _ITERATOR_DEBUG_LEVEL=0 is also selected see
// 1) vcpkg/triplets/x64-windows-meshlib.cmake and
// 2) MeshLib/source/common.props
// Please note that all other modules (.exe, .dll, .lib) with MS STL calls in your application also need
// to define exactly the same value of _ITERATOR_DEBUG_LEVEL to be operational after linking.
//
// If you deliberately would like to work with not zero _ITERATOR_DEBUG_LEVEL, then please define
// additionally MR_ITERATOR_DEBUG_LEVEL with the same value to indicate that it is done intentionally
// (and you are ok with up to 100x slowdown).
//
#if defined _MSC_VER
	#if !defined _ITERATOR_DEBUG_LEVEL
		#define _ITERATOR_DEBUG_LEVEL 0
	#endif
	#if !defined MR_ITERATOR_DEBUG_LEVEL
		#define MR_ITERATOR_DEBUG_LEVEL 0
	#endif
	#if _ITERATOR_DEBUG_LEVEL != MR_ITERATOR_DEBUG_LEVEL
		#error _ITERATOR_DEBUG_LEVEL is inconsistent with MeshLib
	#endif
#endif

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace MR
{

struct NoInit {};
inline constexpr NoInit noInit;
template <typename T> struct MRMESH_CLASS NoDefInit;

class MRMESH_CLASS VertTag;
class MRMESH_CLASS EdgeTag;
class MRMESH_CLASS UndirectedEdgeTag;
class MRMESH_CLASS FaceTag;

template <typename T> class MRMESH_CLASS Id;
using VertId = Id<VertTag>;
using EdgeId = Id<EdgeTag>;
using UndirectedEdgeId = Id<UndirectedEdgeTag>;
using FaceId = Id<FaceTag>;

template <typename T, typename I> class MRMESH_CLASS Vector;
template <typename T, typename I = size_t> class MRMESH_CLASS Buffer;
struct PackMapping;

struct Color;

template <typename T> struct MRMESH_CLASS Vector2;
using Vector2b = Vector2<bool>;
using Vector2i = Vector2<int>;
using Vector2ll = Vector2<long long>;
using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

template <typename T> struct MRMESH_CLASS Vector3;
using Vector3b = Vector3<bool>;
using Vector3i = Vector3<int>;
using Vector3ll = Vector3<long long>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <typename T> struct Vector4;
using Vector4b = Vector4<bool>;
using Vector4i = Vector4<int>;
using Vector4ll = Vector4<long long>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

template <typename T> struct Matrix2;
using Matrix2b = Matrix2<bool>;
using Matrix2i = Matrix2<int>;
using Matrix2ll = Matrix2<long long>;
using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;

template <typename T> struct Matrix3;
using Matrix3b = Matrix3<bool>;
using Matrix3i = Matrix3<int>;
using Matrix3ll = Matrix3<long long>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <typename T> struct Matrix4;
using Matrix4b = Matrix4<bool>;
using Matrix4i = Matrix4<int>;
using Matrix4ll = Matrix4<long long>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

template <typename T> struct SymMatrix2;
using SymMatrix2b = SymMatrix2<bool>;
using SymMatrix2i = SymMatrix2<int>;
using SymMatrix2ll = SymMatrix2<long long>;
using SymMatrix2f = SymMatrix2<float>;
using SymMatrix2d = SymMatrix2<double>;

template <typename T> struct SymMatrix3;
using SymMatrix3b = SymMatrix3<bool>;
using SymMatrix3i = SymMatrix3<int>;
using SymMatrix3ll = SymMatrix3<long long>;
using SymMatrix3f = SymMatrix3<float>;
using SymMatrix3d = SymMatrix3<double>;

template <typename T> struct SymMatrix4;
using SymMatrix4b = SymMatrix4<bool>;
using SymMatrix4i = SymMatrix4<int>;
using SymMatrix4ll = SymMatrix4<long long>;
using SymMatrix4f = SymMatrix4<float>;
using SymMatrix4d = SymMatrix4<double>;

template <typename V> struct AffineXf;
template <typename T> using AffineXf2 = AffineXf<Vector2<T>>;
using AffineXf2f = AffineXf2<float>;
using AffineXf2d = AffineXf2<double>;

template <typename T> using AffineXf3 = AffineXf<Vector3<T>>;
using AffineXf3f = AffineXf3<float>;
using AffineXf3d = AffineXf3<double>;

template <typename T> struct RigidXf3;
using RigidXf3f = RigidXf3<float>;
using RigidXf3d = RigidXf3<double>;

template <typename T> struct RigidScaleXf3;
using RigidScaleXf3f = RigidScaleXf3<float>;
using RigidScaleXf3d = RigidScaleXf3<double>;

template <typename T> struct Sphere;
template <typename T> using Sphere2 = Sphere<Vector2<T>>;
using Sphere2f = Sphere2<float>;
using Sphere2d = Sphere2<double>;

template <typename T> using Sphere3 = Sphere<Vector3<T>>;
using Sphere3f = Sphere3<float>;
using Sphere3d = Sphere3<double>;

template <typename V> struct Line;
template <typename T> using Line2 = Line<Vector2<T>>;
using Line2f = Line2<float>;
using Line2d = Line2<double>;

template <typename T> using Line3 = Line<Vector3<T>>;
using Line3f = Line3<float>;
using Line3d = Line3<double>;

template <typename V> struct LineSegm;
template <typename T> using LineSegm2 = LineSegm<Vector2<T>>;
using LineSegm2f = LineSegm2<float>;
using LineSegm2d = LineSegm2<double>;

template <typename T> using LineSegm3 = LineSegm<Vector3<T>>;
using LineSegm3f = LineSegm3<float>;
using LineSegm3d = LineSegm3<double>;

template <typename T> struct Parabola;
using Parabolaf = Parabola<float>;
using Parabolad = Parabola<double>;

template <typename T> class Cylinder3;
using Cylinder3f = Cylinder3<float>;
using Cylinder3d = Cylinder3<double>;

template <typename T> class Cone3;
using Cone3f = Cone3<float>;
using Cone3d = Cone3<double>;

template <typename V> using Contour = std::vector<V>;
template <typename T> using Contour2 = Contour<Vector2<T>>;
template <typename T> using Contour3 = Contour<Vector3<T>>;
using Contour2d = Contour2<double>;
using Contour2f = Contour2<float>;
using Contour3d = Contour3<double>;
using Contour3f = Contour3<float>;

template <typename V> using Contours = std::vector<Contour<V>>;
template <typename T> using Contours2 = Contours<Vector2<T>>;
template <typename T> using Contours3 = Contours<Vector3<T>>;
using Contours2d = Contours2<double>;
using Contours2f = Contours2<float>;
using Contours3d = Contours3<double>;
using Contours3f = Contours3<float>;

template <typename T> using Contour3 = Contour<Vector3<T>>;
using Contour3d = Contour3<double>;
using Contours3d = std::vector<Contour3d>;
using Contour3f = Contour3<float>;
using Contours3f = std::vector<Contour3f>;

template <typename T> struct Plane3;
using Plane3f = Plane3<float>;
using Plane3d = Plane3<double>;

template <typename V> struct Box;
template <typename T> using Box2 = Box<Vector2<T>>;
using Box2i = Box2<int>;
using Box2ll = Box2<long long>;
using Box2f = Box2<float>;
using Box2d = Box2<double>;

template <typename T> using Box3 = Box<Vector3<T>>;
using Box3i = Box3<int>;
using Box3ll = Box3<long long>;
using Box3f = Box3<float>;
using Box3d = Box3<double>;

template <typename V> struct QuadraticForm;

template <typename T> using QuadraticForm2 = QuadraticForm<Vector2<T>>;
using QuadraticForm2f = QuadraticForm2<float>;
using QuadraticForm2d = QuadraticForm2<double>;

template <typename T> using QuadraticForm3 = QuadraticForm<Vector3<T>>;
using QuadraticForm3f = QuadraticForm3<float>;
using QuadraticForm3d = QuadraticForm3<double>;

template <typename T> struct Quaternion;
using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

template <typename T> using Triangle3 = std::array<Vector3<T>, 3>;
using Triangle3i = Triangle3<int>;
using Triangle3f = Triangle3<float>;
using Triangle3d = Triangle3<double>;

template <typename T> struct SegmPoint;
using SegmPointf = SegmPoint<float>;
using SegmPointd = SegmPoint<double>;

template <typename T> struct TriPoint;
using TriPointf = TriPoint<float>;
using TriPointd = TriPoint<double>;

using VertMap = Vector<VertId, VertId>;
using EdgeMap = Vector<EdgeId, EdgeId>;
using UndirectedEdgeMap = Vector<UndirectedEdgeId, UndirectedEdgeId>;
using FaceMap = Vector<FaceId, FaceId>;

using VertColors = Vector<Color, VertId>;
using EdgeColors = Vector<Color, EdgeId>;
using UndirectedEdgeColors = Vector<Color, UndirectedEdgeId>;
using FaceColors = Vector<Color, FaceId>;

using VertScalars = Vector<float, VertId>;
using EdgeScalars = Vector<float, EdgeId>;
using UndirectedEdgeScalars = Vector<float, UndirectedEdgeId>;
using FaceScalars = Vector<float, FaceId>;

template <typename T> struct IntersectionPrecomputes;
template <typename T> struct IntersectionPrecomputes2;

template <typename I> struct IteratorRange;

template <typename T, typename I> struct MRMESH_CLASS BMap;
using VertBMap = BMap<VertId, VertId>;
using EdgeBMap = BMap<EdgeId, EdgeId>;
using UndirectedEdgeBMap = BMap<UndirectedEdgeId, UndirectedEdgeId>;
using WholeEdgeBMap = BMap<EdgeId, UndirectedEdgeId>;
using FaceBMap = BMap<FaceId, FaceId>;

template <typename T>
[[nodiscard]] inline bool contains( const std::function<bool( Id<T> )>& pred, Id<T> id )
{
	return id.valid() && ( !pred || pred( id ) );
}

template <typename I> class UnionFind;

template<typename T>
class FewSmallest;

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

template <typename T>
constexpr inline T sqr( T x ) noexcept
{
	return x * x;
}

template <typename T>
constexpr inline int sgn( T x ) noexcept
{
	return x > 0 ? 1 : ( x < 0 ? -1 : 0 );
}

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

// explicit deduction guide (not needed as of C++20, but still needed in Clang)
template<class... Ts>
overloaded( Ts... ) -> overloaded<Ts...>;

} // namespace MR

#ifdef __cpp_lib_unreachable
#   define MR_UNREACHABLE std::unreachable();
#   define MR_UNREACHABLE_NO_RETURN std::unreachable();
#else
#   ifdef __GNUC__
#       define MR_UNREACHABLE __builtin_unreachable();
#       define MR_UNREACHABLE_NO_RETURN __builtin_unreachable();
#   else
#       define MR_UNREACHABLE { assert( false ); return {}; }
#       define MR_UNREACHABLE_NO_RETURN assert( false );
#   endif
#endif
