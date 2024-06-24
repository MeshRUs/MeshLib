#pragma once

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
#include <vector>
#include <string>
#include <parallel_hashmap/phmap_fwd_decl.h>
#include <functional>

#ifdef _WIN32
#   ifdef MRMESH_EXPORT
#       define MRMESH_API __declspec(dllexport)
#   else
#       define MRMESH_API __declspec(dllimport)
#   endif
#   define MRMESH_CLASS
#else
#   define MRMESH_API   __attribute__((visibility("default")))
// to fix undefined reference to `typeinfo/vtable`
// Also it's important to use this on any type for which `typeid` is used in multiple shared libraries, and then passed across library boundaries.
//   Otherwise on Mac the resulting typeids will incorrectly compare not equal.
#   define MRMESH_CLASS __attribute__((visibility("default")))
#endif

namespace MR
{

struct NoInit {};
inline constexpr NoInit noInit;
template <typename T> struct MRMESH_CLASS NoDefInit;

class MRMESH_CLASS EdgeTag;
class MRMESH_CLASS UndirectedEdgeTag;
class MRMESH_CLASS FaceTag;
class MRMESH_CLASS VertTag;
class MRMESH_CLASS PixelTag;
class MRMESH_CLASS VoxelTag;
class MRMESH_CLASS RegionTag;
class MRMESH_CLASS NodeTag;
class MRMESH_CLASS ObjTag;

template <typename T> class MRMESH_CLASS Id;
template <typename T, typename I> class MRMESH_CLASS Vector;
template <typename T, typename I = size_t> class MRMESH_CLASS Buffer;
struct PackMapping;

using EdgeId = Id<EdgeTag>;
using UndirectedEdgeId = Id<UndirectedEdgeTag>;
using FaceId = Id<FaceTag>;
using VertId = Id<VertTag>;
using PixelId = Id<PixelTag>;
using VoxelId = Id<VoxelTag>;
using RegionId = Id<RegionTag>;
using NodeId = Id<NodeTag>;
using ObjId = Id<ObjTag>;

class ViewportId;
class ViewportMask;

/// three vertex ids describing a triangle topology
using ThreeVertIds = std::array<VertId, 3>;
/// mapping from FaceId to a triple of vertex indices
using Triangulation = Vector<ThreeVertIds, FaceId>;

struct UnorientedTriangle;
struct SomeLocalTriangulations;
struct AllLocalTriangulations;

using EdgePath = std::vector<EdgeId>;
using EdgeLoop = std::vector<EdgeId>;

class MRMESH_CLASS BitSet;
template <typename T> class MRMESH_CLASS TaggedBitSet;
template <typename T> class MRMESH_CLASS SetBitIteratorT;

struct Color;

struct MRMESH_CLASS Dipole;
using Dipoles = Vector<Dipole, NodeId>;

using FaceBitSet = TaggedBitSet<FaceTag>;
using VertBitSet = TaggedBitSet<VertTag>;
using EdgeBitSet = TaggedBitSet<EdgeTag>;
using UndirectedEdgeBitSet = TaggedBitSet<UndirectedEdgeTag>;
using PixelBitSet = TaggedBitSet<PixelTag>;
using VoxelBitSet = TaggedBitSet<VoxelTag>;
using NodeBitSet = TaggedBitSet<NodeTag>;
using ObjBitSet = TaggedBitSet<ObjTag>;

template <typename T> class SetBitIteratorT;

using SetBitIterator     = SetBitIteratorT<BitSet>;
using FaceSetBitIterator = SetBitIteratorT<FaceBitSet>;
using VertSetBitIterator = SetBitIteratorT<VertBitSet>;
using EdgeSetBitIterator = SetBitIteratorT<EdgeBitSet>;
using UndirectedEdgeSetBitIterator = SetBitIteratorT<UndirectedEdgeBitSet>;

template <typename T> struct MRMESH_CLASS Vector2;
using Vector2b = Vector2<bool>;
using Vector2i = Vector2<int>;
using Vector2ll= Vector2<long long>;
using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

template <typename T> struct MRMESH_CLASS Vector3;
using Vector3b = Vector3<bool>;
using Vector3i = Vector3<int>;
using Vector3ll= Vector3<long long>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <typename T> struct Vector4;
using Vector4b = Vector4<bool>;
using Vector4i = Vector4<int>;
using Vector4ll= Vector4<long long>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

template <typename T> struct Matrix2;
using Matrix2b = Matrix2<bool>;
using Matrix2i = Matrix2<int>;
using Matrix2ll= Matrix2<long long>;
using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;

template <typename T> struct Matrix3;
using Matrix3b = Matrix3<bool>;
using Matrix3i = Matrix3<int>;
using Matrix3ll= Matrix3<long long>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <typename T> struct Matrix4;
using Matrix4b = Matrix4<bool>;
using Matrix4i = Matrix4<int>;
using Matrix4ll= Matrix4<long long>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

template <typename T> struct SymMatrix2;
using SymMatrix2b = SymMatrix2<bool>;
using SymMatrix2i = SymMatrix2<int>;
using SymMatrix2ll= SymMatrix2<long long>;
using SymMatrix2f = SymMatrix2<float>;
using SymMatrix2d = SymMatrix2<double>;

template <typename T> struct SymMatrix3;
using SymMatrix3b = SymMatrix3<bool>;
using SymMatrix3i = SymMatrix3<int>;
using SymMatrix3ll= SymMatrix3<long long>;
using SymMatrix3f = SymMatrix3<float>;
using SymMatrix3d = SymMatrix3<double>;

template <typename T> struct SymMatrix4;
using SymMatrix4b = SymMatrix4<bool>;
using SymMatrix4i = SymMatrix4<int>;
using SymMatrix4ll= SymMatrix4<long long>;
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

class PointToPointAligningTransform;
class PointToPlaneAligningTransform;

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

template <typename T> class BestFitParabola;
using BestFitParabolaf = BestFitParabola<float>;
using BestFitParabolad = BestFitParabola<double>;

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

class PointAccumulator;

template <typename T> struct SegmPoint;
using SegmPointf = SegmPoint<float>;
using SegmPointd = SegmPoint<double>;
struct EdgePoint;
struct EdgeSegment;
using MeshEdgePoint = EdgePoint;
using SurfacePath = std::vector<MeshEdgePoint>;
using SurfacePaths = std::vector<SurfacePath>;
using IsoLine = SurfacePath;
using IsoLines = SurfacePaths;
using PlaneSection = SurfacePath;
using PlaneSections = SurfacePaths;
struct EdgePointPair;
class Laplacian;

using VertPair = std::pair<VertId, VertId>;
using FacePair = std::pair<FaceId, FaceId>;
using EdgePair = std::pair<EdgeId, EdgeId>;
using UndirectedEdgePair = std::pair<UndirectedEdgeId, UndirectedEdgeId>;

template <typename T> struct TriPoint;
using TriPointf = TriPoint<float>;
using TriPointd = TriPoint<double>;
struct PointOnFace;
struct PointOnObject;
struct MeshTriPoint;
struct MeshProjectionResult;
struct MeshIntersectionResult;
template <typename T> struct IntersectionPrecomputes;

template <typename I> struct IteratorRange;

/// Coordinates on texture
/// \param x,y should be in range [0..1], otherwise result depends on wrap type of texture (no need to clamp it, it is done on GPU if wrap type is "Clamp" )
using UVCoord = Vector2f;

using FaceMap = Vector<FaceId, FaceId>;
using VertMap = Vector<VertId, VertId>;
using EdgeMap = Vector<EdgeId, EdgeId>;
using UndirectedEdgeMap = Vector<UndirectedEdgeId, UndirectedEdgeId>;
using ObjMap = Vector<ObjId, ObjId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeMap = Vector<EdgeId, UndirectedEdgeId>;
using UndirectedEdge2RegionMap = Vector<RegionId, UndirectedEdgeId>;
using Face2RegionMap = Vector<RegionId, FaceId>;
using Vert2RegionMap = Vector<RegionId, VertId>;

using VertCoords = Vector<Vector3f, VertId>;
using VertNormals = Vector<Vector3f, VertId>;
using VertUVCoords = Vector<UVCoord, VertId>;
using FaceNormals = Vector<Vector3f, FaceId>;

using VertColors = Vector<Color, VertId>;
using FaceColors = Vector<Color, FaceId>;
using EdgeColors = Vector<Color, EdgeId>;
using UndirectedEdgeColors = Vector<Color, UndirectedEdgeId>;

using VertScalars = Vector<float, VertId>;
using FaceScalars = Vector<float, FaceId>;
using EdgeScalars = Vector<float, EdgeId>;
using UndirectedEdgeScalars = Vector<float, UndirectedEdgeId>;

using VertPredicate = std::function<bool( VertId )>;
using FacePredicate = std::function<bool( FaceId )>;
using EdgePredicate = std::function<bool( EdgeId )>;
using UndirectedEdgePredicate = std::function<bool( UndirectedEdgeId )>;

template <typename T>
[[nodiscard]] inline bool contains( const std::function<bool( Id<T> )> & pred, Id<T> id )
{
    return id.valid() && ( !pred || pred( id ) );
}

using VertMetric = std::function<float( VertId )>;
using FaceMetric = std::function<float( FaceId )>;
using EdgeMetric = std::function<float( EdgeId )>;
using UndirectedEdgeMetric = std::function<float( UndirectedEdgeId )>;

template <typename T, typename I> struct MRMESH_CLASS BMap;
using FaceBMap = BMap<FaceId, FaceId>;
using VertBMap = BMap<VertId, VertId>;
using EdgeBMap = BMap<EdgeId, EdgeId>;
using UndirectedEdgeBMap = BMap<UndirectedEdgeId, UndirectedEdgeId>;
using WholeEdgeBMap = BMap<EdgeId, UndirectedEdgeId>;

template <typename T, typename Hash = phmap::priv::hash_default_hash<T>, typename Eq = phmap::priv::hash_default_eq<T>>
using HashSet = phmap::flat_hash_set<T, Hash, Eq>;
template <typename T, typename Hash = phmap::priv::hash_default_hash<T>, typename Eq = phmap::priv::hash_default_eq<T>>
using ParallelHashSet = phmap::parallel_flat_hash_set<T, Hash, Eq>;

using FaceHashSet = HashSet<FaceId>;
using VertHashSet = HashSet<VertId>;
using EdgeHashSet = HashSet<EdgeId>;

template <typename K, typename V, typename Hash = phmap::priv::hash_default_hash<K>, typename Eq = phmap::priv::hash_default_eq<K>>
using HashMap = phmap::flat_hash_map<K, V, Hash, Eq>;
template <typename K, typename V, typename Hash = phmap::priv::hash_default_hash<K>, typename Eq = phmap::priv::hash_default_eq<K>>
using ParallelHashMap = phmap::parallel_flat_hash_map<K, V, Hash, Eq>;

using FaceHashMap = HashMap<FaceId, FaceId>;
using VertHashMap = HashMap<VertId, VertId>;
using EdgeHashMap = HashMap<EdgeId, EdgeId>;
using UndirectedEdgeHashMap = HashMap<UndirectedEdgeId, UndirectedEdgeId>;
///  mapping of whole edges: map[e]->f, map[e.sym()]->f.sym(), where only map[e] for even edges is stored
using WholeEdgeHashMap = HashMap<UndirectedEdgeId, EdgeId>;

template <typename I> class UnionFind;
template <typename T, typename I, typename P> class Heap;

class MRMESH_CLASS MeshTopology;
struct MRMESH_CLASS Mesh;
struct MRMESH_CLASS MeshPart;
class MRMESH_CLASS MeshOrPoints;
struct MRMESH_CLASS PointCloud;
class MRMESH_CLASS AABBTree;
class MRMESH_CLASS AABBTreePoints;
class MRMESH_CLASS AABBTreeObjects;
struct MRMESH_CLASS CloudPartMapping;
struct MRMESH_CLASS PartMapping;
struct MeshOrPointsXf;
struct MeshTexture;
struct GridSettings;
struct TriMesh;

template<typename T> class UniqueThreadSafeOwner;

class PolylineTopology;
template<typename V>
struct Polyline;
using Polyline2 = Polyline<Vector2f>;
using Polyline3 = Polyline<Vector3f>;
template<typename V>
class MRMESH_CLASS AABBTreePolyline;
using AABBTreePolyline2 = AABBTreePolyline<Vector2f>;
using AABBTreePolyline3 = AABBTreePolyline<Vector3f>;

template<typename T> struct IntersectionPrecomputes;
template<typename T> struct IntersectionPrecomputes2;

template<typename V> struct [[nodiscard]] PolylineProjectionResult;
using PolylineProjectionResult2 = PolylineProjectionResult<Vector2f>;
using PolylineProjectionResult3 = PolylineProjectionResult<Vector3f>;

template<typename V> struct [[nodiscard]] PolylineProjectionWithOffsetResult;
using Polyline2ProjectionWithOffsetResult = PolylineProjectionWithOffsetResult<Vector2f>;
using PolylineProjectionWithOffsetResult3 = PolylineProjectionWithOffsetResult<Vector3f>;

class DistanceMap;

using GcodeSource = std::vector<std::string>;

class Object;
class SceneRootObject;
class VisualObject;
class ObjectMeshHolder;
class ObjectMesh;
class ObjectPointsHolder;
class ObjectPoints;
class ObjectLinesHolder;
class ObjectLines;
class ObjectDistanceMap;
class ObjectLabel;
class ObjectGcode;
class PointObject;
class LineObject;
class CircleObject;
class PlaneObject;
class SphereObject;
class CylinderObject;
class ConeObject;

struct Image;
class AnyVisualizeMaskEnum;

template <typename T>
struct VoxelsVolume;
using SimpleVolume = VoxelsVolume<std::vector<float>>;
using SimpleVolumeU16 = VoxelsVolume<std::vector<uint16_t>>;

template <typename T>
using VoxelValueGetter = std::function<T ( const Vector3i& )>;
using FunctionVolume = VoxelsVolume<VoxelValueGetter<float>>;
using FunctionVolumeU8 = VoxelsVolume<VoxelValueGetter<uint8_t>>;

#ifndef MRMESH_NO_OPENVDB
class ObjectVoxels;

struct OpenVdbFloatGrid;
using FloatGrid = std::shared_ptr<OpenVdbFloatGrid>;
using VdbVolume = VoxelsVolume<FloatGrid>;
#endif

class HistoryAction;
class ChangeObjectAction;
class MRMESH_CLASS ChangeSceneAction;
class ChangeMeshFaceSelectionAction;
class ChangeMeshEdgeSelectionAction;
class ChangeMeshCreasesAction;
class ChangePointPointSelectionAction;
class ChangeMeshAction;
class ChangeMeshPointsAction;
class ChangeMeshTopologyAction;
class ChangeXfAction;
class CombinedHistoryAction;
class SwapRootAction;

template <typename Tag>
class MRMESH_CLASS ColorMapAggregator;
using VertColorMapAggregator = ColorMapAggregator<VertTag>;
using UndirEdgeColorMapAggregator = ColorMapAggregator<UndirectedEdgeTag>;
using FaceColorMapAggregator = ColorMapAggregator<FaceTag>;

template<typename T>
class FewSmallest;

class Graph;
class GraphVertTag;
class GraphEdgeTag;
using GraphVertId = Id<GraphVertTag>;
using GraphEdgeId = Id<GraphEdgeTag>;
using GraphVertBitSet = TaggedBitSet<GraphVertTag>;
using GraphEdgeBitSet = TaggedBitSet<GraphEdgeTag>;

class WatershedGraph;

/// Argument value - progress in [0,1];
/// returns true to continue the operation and returns false to stop the operation
/// \ingroup BasicStructuresGroup
typedef std::function<bool( float )> ProgressCallback;

enum class FilterType : char
{
    Linear,
    Discrete
};

enum class WrapType : char
{
    Repeat,
    Mirror,
    Clamp
};

/// determines how points to be ordered
enum class Reorder : char
{
    None,              ///< the order is not changed
    Lexicographically, ///< the order is determined by lexicographical sorting by coordinates (optimal for uniform sampling)
    AABBTree           ///< the order is determined so to put close in space points in close indices (optimal for compression)
};

template <typename T>
constexpr inline T sqr( T x ) noexcept { return x * x; }

template <typename T>
constexpr inline int sgn( T x ) noexcept { return x > 0 ? 1 : ( x < 0 ? -1 : 0 ); }

template<typename...>
inline constexpr bool dependent_false = false;

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

// explicit deduction guide (not needed as of C++20, but still needed in Clang)
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class IFastWindingNumber;
class IPointsToMeshProjector;

namespace MeshBuilder
{

struct BuildSettings;
struct Triangle;
struct VertDuplication;

} //namespace MeshBuilder

} //namespace MR

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
