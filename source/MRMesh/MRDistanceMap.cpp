#include "MRDistanceMap.h"
#include "MRMeshIntersect.h"
#include "MRBox.h"
#include "MRImageSave.h"
#include "MRImageLoad.h"
#include "MRImage.h"
#include "MRTriangleIntersection.h"
#include "MRLine3.h"
#include "MRTimer.h"
#include "MRBox.h"
#include "MRRegularGridMesh.h"
#include "MRPolyline2Project.h"
#include "MRBitSetParallelFor.h"
#include "MRPolyline2Intersect.h"
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <vector>

namespace MR
{

static constexpr float NOT_VALID_VALUE = std::numeric_limits<float>::lowest();

DistanceMap::DistanceMap( const MR::Matrix<float>& m ) 
    : RectIndexer( m ) 
    , data_( m.data() )
{
}

DistanceMap::DistanceMap( size_t resX, size_t resY )
    : RectIndexer( { (int)resX, (int)resY } )
    , data_( size(), NOT_VALID_VALUE )
{
    invalidateAll();
}

bool DistanceMap::isValid( size_t x, size_t y ) const
{
    return data_[ toIndex( { int( x ), int( y ) } ) ] != NOT_VALID_VALUE;
}

bool DistanceMap::isValid( size_t i ) const
{
    return data_[i] != NOT_VALID_VALUE;
}

std::optional<float> DistanceMap::get( size_t x, size_t y ) const
{
    if ( isValid( x, y ) )
        return data_[ toIndex( { int( x ), int( y ) } ) ];
    else
        return std::nullopt;
}

std::optional<float> DistanceMap::get( size_t i ) const
{
    if ( isValid( i ) )
        return data_[i];
    else
        return std::nullopt;
}

std::optional<float> DistanceMap::getInterpolated( float x, float y ) const
{
    if ( x < 0.f )
    {
        return std::nullopt;
    }
    else if ( x < 0.5f )
    {
        x = 0.f;
    }
    else if ( x > float( resX() ) )
    {
        return std::nullopt;
    }
    else if( x > float( resX() ) - 0.5f )
    {
        x = float( resX() ) - 1.f;
    }
    else
    {
        x -= 0.5f;
    }

    if ( y < 0.f )
    {
        return std::nullopt;
    }
    else if ( y < 0.5f )
    {
        y = 0.f;
    }
    else if ( y > float( resY() ) )
    {
        return std::nullopt;
    }
    else if ( y > float( resY() ) - 0.5f )
    {
        y = float( resY() ) - 1.f;
    }
    else
    {
        y -= 0.5f;
    }

    float xlowf = float( std::floor( x ) );
    float ylowf = float( std::floor( y ) );
    float xhighf = xlowf + 1.f;
    float yhighf = ylowf + 1.f;
    int xlow = int( xlowf );
    int ylow = int( ylowf );
    int xhigh = xlow + 1;
    int yhigh = ylow + 1;

    auto lowlow = get( xlow, ylow );
    auto lowhigh = get( xlow, yhigh );
    auto highlow = get( xhigh, ylow );
    auto highhigh = get( xhigh, yhigh );
    if ( lowlow && lowhigh && highlow && highhigh )
    {
        // bilinear interpolation
        // https://en.wikipedia.org/wiki/Bilinear_interpolation
        return ( ( *lowlow ) * ( yhighf - y ) + ( *lowhigh ) * ( y - ylowf ) ) * ( xhighf - x ) +
            ( ( *highlow ) * ( yhighf - y ) + ( *highhigh ) * ( y - ylowf ) ) * ( x - xlowf );
    }
    else
    {
        return std::nullopt;
    }
}

void DistanceMap::set( size_t x, size_t y, float val )
{
    data_[ toIndex( { int( x ), int( y ) } ) ] = val;
}

void DistanceMap::set( size_t i, float val )
{
    data_[i] = val;
}

void DistanceMap::set( std::vector<float> data )
{
    assert( data.size() == data_.size() );
    data_ = std::move( data );
}

void DistanceMap::unset( size_t x, size_t y )
{
    data_[ toIndex( { int( x ), int( y ) } ) ] = NOT_VALID_VALUE;
}

void DistanceMap::unset( size_t i )
{
    data_[i] = NOT_VALID_VALUE;
}

void DistanceMap::invalidateAll()
{
    for( auto& elem : data_ )
        elem = NOT_VALID_VALUE;
}

void DistanceMap::clear()
{
    RectIndexer::resize( {0, 0} );
    data_.clear();
}

std::optional<Vector3f> DistanceMap::unproject( size_t x, size_t y, const DistanceMapToWorld& toWorldStruct ) const
{
    auto val = get( x, y );
    if ( !val )
        return {};
    return toWorldStruct.toWorld( x + 0.5f, y + 0.5f, *val );
}

std::optional<Vector3f> DistanceMap::unprojectInterpolated( float x, float y, const DistanceMapToWorld& toWorldStruct ) const
{
    auto val = getInterpolated( x, y );
    if ( !val )
        return {};
    return toWorldStruct.toWorld( x, y, *val );
}

Mesh distanceMapToMesh( const DistanceMap& distMap, const AffineXf3f& xf )
{
    DistanceMapToWorld toWorldParams;
    toWorldParams.direction = xf.A.z;
    toWorldParams.orgPoint = xf.b;
    toWorldParams.pixelXVec = xf.A.x;
    toWorldParams.pixelYVec = xf.A.y;

    return distanceMapToMesh( distMap, toWorldParams );
}

Mesh distanceMapToMesh( const DistanceMap& distMap, const DistanceMapToWorld& toWorldStruct )
{
    auto resX = distMap.resX();
    auto resY = distMap.resY();

    if (resX < 2 || resY < 2)
    {
        return Mesh();
    }

    return makeRegularGridMesh( resX, resY, [&]( size_t x, size_t y )
    {
        return distMap.isValid( x, y );
    },
                                            [&]( size_t x, size_t y )
    {
        return distMap.unproject( x, y, toWorldStruct ).value_or( Vector3f{} );
    } );
}

VoidOrErrStr saveDistanceMapToImage( const DistanceMap& dm, const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    threshold = std::clamp( threshold, 0.f, 1.f );
    auto size = dm.numPoints();
    std::vector<Color> pixels( size );
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    // find min-max
    for ( int i = 0; i < size; ++i )
    {
        const auto val = dm.get( i );
        if ( val )
        {
            if ( *val < min )
                min = *val;
            if ( val > max )
                max = *val;
        }
    }

    for ( int i = 0; i < size; ++i )
    {
        const auto val = dm.get( i );
        pixels[i] = val ?
            Color( Vector3f::diagonal( ( max - *val ) / ( max - min ) * ( 1 - threshold ) + threshold ) ) :
            Color::black();
    }

    return ImageSave::toAnySupportedFormat( { pixels, { int( dm.resX() ), int( dm.resY() ) } }, filename );
}


tl::expected<MR::DistanceMap, std::string> convertImageToDistanceMap( const Image& image, float threshold /*= 1.f / 255*/ )
{
    threshold = std::clamp( threshold * 255, 0.f, 255.f );
    DistanceMap dm( image.resolution.x, image.resolution.y );
    const auto& pixels = image.pixels;
    for ( int i = 0; i < image.pixels.size(); ++i )
    {
        const bool monochrome = pixels[i].r == pixels[i].g && pixels[i].g == pixels[i].b;
        assert( monochrome );
        if ( !monochrome )
            return tl::make_unexpected( "Error convert Image to DistanceMap: image isn't monochrome" );
        if ( pixels[i].r < threshold )
            continue;
        dm.set( i, 255.0f - pixels[i].r );
    }
    return dm;
}

tl::expected<MR::DistanceMap, std::string> loadDistanceMapFromImage( const std::filesystem::path& filename, float threshold /*= 1.f / 255*/ )
{
    auto resLoad = ImageLoad::fromAnySupportedFormat( filename );
    if ( !resLoad.has_value() )
        return tl::make_unexpected( resLoad.error() );
    return convertImageToDistanceMap( *resLoad, threshold );
}

template <typename T = float>
DistanceMap computeDistanceMap_( const MeshPart& mp, const MeshToDistanceMapParams& params, ProgressCallback cb = {} )
{
    DistanceMap distMap( params.resolution.x, params.resolution.y );

    // precomputed some values
    IntersectionPrecomputes<T> prec( Vector3<T>( params.direction ) );

    auto ori = params.orgPoint;
    float shift = 0.f;
    if ( params.allowNegativeValues )
    {
        AffineXf3f xf( Matrix3f( params.xRange.normalized(), params.yRange.normalized(), params.direction.normalized() ), Vector3f() );
        Box box = mp.mesh.computeBoundingBox( mp.region,&xf );
        shift = dot( params.direction, ori - box.min );
        if ( shift > 0.f )
        {
            ori -= params.direction * shift;
        }
        else
        {
            shift = T( 0 );
        }
    }

    T xStep_1 = T( 1 ) / T( params.resolution.x );
    T yStep_1 = T( 1 ) / T( params.resolution.y );

    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };

    if ( params.useDistanceLimits )
    {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, params.resolution.x ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            const auto xMin = range.begin();
            const auto xMax = xMin + range.size();

            for ( size_t x = range.begin(); x < range.end(); x++ )
            {
                if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                    break;

                for ( size_t y = 0; y < params.resolution.y; y++ )
                {
                    Vector3<T> rayOri = Vector3<T>( ori ) +
                        Vector3<T>( params.xRange ) * ( ( T( x ) + T( 0.5 ) ) * xStep_1 ) +
                        Vector3<T>( params.yRange ) * ( ( T( y ) + T( 0.5 ) ) * yStep_1 );
                    if ( auto meshIntersectionRes = rayMeshIntersect( mp, Line3<T>( Vector3<T>( rayOri ), Vector3<T>( params.direction ) ),
                        -std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), &prec ) )
                    {
                        if ( ( meshIntersectionRes->distanceAlongLine < params.minValue ) || ( meshIntersectionRes->distanceAlongLine > params.maxValue ) )
                            distMap.set( x, y, meshIntersectionRes->distanceAlongLine );
                    }
                }

                if ( cb && std::this_thread::get_id() == mainThreadId )
                {
                    if ( cb && !cb( float( x - xMin ) / float( xMax - xMin ) ) )
                        keepGoing.store( false, std::memory_order_relaxed );
                }
            }
        }, tbb::static_partitioner() );
    }
    else
    {
        tbb::parallel_for( tbb::blocked_range<size_t>( 0, params.resolution.x ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            const auto xMin = range.begin();
            const auto xMax = xMin + range.size();
            for ( size_t x = range.begin(); x < range.end(); x++ )
            //debug line
            //for ( size_t x = 0; x < params.resX; x++ )
            {
                
                if ( cb && !keepGoing.load( std::memory_order_relaxed ) )
                    break;

                for ( size_t y = 0; y < params.resolution.y; y++ )
                {
                    Vector3<T> rayOri = Vector3<T>( ori ) +
                        Vector3<T>( params.xRange ) * ( ( T( x ) + T( 0.5 ) ) * xStep_1 ) +
                        Vector3<T>( params.yRange ) * ( ( T( y ) + T( 0.5 ) ) * yStep_1 );
                    if ( auto meshIntersectionRes = rayMeshIntersect( mp, Line3<T>( Vector3<T>( rayOri ), Vector3<T>( params.direction ) ),
                        -std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), &prec ) )
                    {
                        distMap.set( x, y, meshIntersectionRes->distanceAlongLine );
                    }
                }

                if ( cb && std::this_thread::get_id() == mainThreadId )
                {
                    if ( cb && !cb( float( x - xMin ) / float ( xMax - xMin ) ) )
                        keepGoing.store( false, std::memory_order_relaxed );
                }
            }
        }, tbb::static_partitioner() );
    }

    if ( !keepGoing.load( std::memory_order_relaxed ) || ( cb && !cb( 1.0f ) ) )
        return DistanceMap{};

    if ( params.allowNegativeValues )
    {
        for ( int i = 0; i < distMap.numPoints(); i++ )
        {
            const auto val = distMap.get( i );
            if ( val )
                distMap.set( i, *val - shift );
        }
    }

    return distMap;
}

DistanceMap computeDistanceMap( const MeshPart& mp, const MeshToDistanceMapParams& params, ProgressCallback cb )
{
    return computeDistanceMap_<float>( mp, params, cb );
}

DistanceMap computeDistanceMapD( const MeshPart& mp, const MeshToDistanceMapParams& params, ProgressCallback cb )
{
    return computeDistanceMap_<double>( mp, params, cb );
}

DistanceMap distanceMapFromContours( const Polyline2& polyline, const ContourToDistanceMapParams& params,
    const ContoursDistanceMapOptions& options )
{
    MR_TIMER
    assert( polyline.topology.isConsistentlyOriented() );

    if ( options.offsetParameters )
    {
        bool goodSize = options.offsetParameters->perEdgeOffset.size() >= polyline.topology.undirectedEdgeSize();
        if ( !goodSize )
        {
            assert( false );
            spdlog::error( "Offset per edges should contain offset for all edges" );
            return {};
        }
    }

    const Vector3f originPoint = Vector3f{ params.orgPoint.x, params.orgPoint.y, 0.F } +
        Vector3f{ params.pixelSize.x / 2.F, params.pixelSize.y / 2.F, 0.F };

    size_t size = size_t( params.resolution.x ) * params.resolution.y;
    if ( options.outClosestEdges )
        options.outClosestEdges->resize( size );

    DistanceMap distMap( params.resolution.x, params.resolution.y );
    if ( !polyline.topology.lastNotLoneEdge().valid())
        return distMap;

    const auto maxDistSq = sqr( options.maxDist );
    const auto minDistSq = sqr( options.minDist );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, size ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( options.region && !options.region->test( PixelId( int( i ) ) ) )
                continue;
            size_t x = i % params.resolution.x;
            size_t y = i / params.resolution.x;
            Vector2f p;
            p.x = params.pixelSize.x * x + originPoint.x;
            p.y = params.pixelSize.y * y + originPoint.y;
            Polyline2ProjectionWithOffsetResult res;
            if ( options.offsetParameters )
            {
                res = findProjectionOnPolyline2WithOffset( p, polyline, options.offsetParameters->perEdgeOffset, options.maxDist, nullptr, options.minDist );
            }
            else
            {
                auto noOffsetRes = findProjectionOnPolyline2( p, polyline, maxDistSq, nullptr, minDistSq );
                res.line = noOffsetRes.line;
                res.point = noOffsetRes.point;
                res.dist = std::sqrt( noOffsetRes.distSq );
            }
            
            if ( options.outClosestEdges )
                ( *options.outClosestEdges )[i] = res.line;

            if ( params.withSign && ( !options.offsetParameters || options.offsetParameters->type != ContoursDistanceMapOffset::OffsetType::Shell ) )
            {
                bool positive = true;
                if ( options.signMethod == ContoursDistanceMapOptions::SignedDetectionMethod::ContourOrientation )
                {
                    const EdgeId e = res.line;
                    const auto& v0 = polyline.points[polyline.topology.org( e )];
                    const auto& v1 = polyline.points[polyline.topology.dest( e )];
                    auto vecA = v1 - v0;
                    auto ray = res.point - p;
                    float ratio = dot( res.point - v0, vecA ) / vecA.lengthSq();
                    if ( ratio <= 0.0f || ratio >= 1.0f )
                    {
                        Vector2f vecB;
                        const EdgeId prevEdge = polyline.topology.next( e ).sym();
                        const EdgeId nextEdge = polyline.topology.next( e.sym() );
                        if ( ratio <= 0.0f && e.sym() != prevEdge )
                        {
                            const auto& v2 = polyline.points[polyline.topology.org( prevEdge )];
                            vecB = v0 - v2;
                        }
                        else if ( ratio >= 1.0f && e.sym() != nextEdge )
                        {
                            const auto& v2 = polyline.points[polyline.topology.dest( nextEdge )];
                            vecB = v2 - v1;
                        }
                        vecA = ( vecA.normalized() + vecB.normalized() ) * 0.5f;
                    }
                    if ( cross( vecA, ray ) > 0.0f )
                        positive = false;
                }
                else if ( options.signMethod == ContoursDistanceMapOptions::SignedDetectionMethod::WindingRule )
                {
                    if ( isPointInsidePolyline( polyline, p ) )
                        positive = false;
                }
                if ( !positive )
                {
                    res.dist *= -1.0f;
                    if ( options.offsetParameters )
                        res.dist -= 2.0f * options.offsetParameters->perEdgeOffset[res.line];
                }
            }
            if ( !params.withSign && options.offsetParameters && options.offsetParameters->type == ContoursDistanceMapOffset::OffsetType::Shell )
                res.dist = std::abs( res.dist );
            distMap.set( x, y, res.dist );
        }
    } );
    return distMap;
}

std::vector<Vector3f> edgePointsFromContours( const Polyline2& polyline, float pixelSize, float threshold )
{
    assert( polyline.topology.isConsistentlyOriented() );
    std::vector<Vector3f> edgePoints;
    auto box = polyline.getBoundingBox();
    assert( box.valid() );
    auto resX = ( int )std::ceil( ( box.max.x - box.min.x ) / pixelSize );
    auto resY = ( int )std::ceil( ( box.max.y - box.min.y ) / pixelSize );

    std::vector<Vector2f> prevLine;
    Vector2f prevPix;
    prevLine.resize( resX );
    for ( auto x = 0; x < resX; x++ )
    {
        Vector2f p = box.min + Vector2f( pixelSize * ( float( x ) + 0.5f ), 0.f );
        auto res = findProjectionOnPolyline2( p, polyline );
        prevLine[x] = res.point;
    }

    float distLimitSq = threshold * threshold;
    for ( auto y = 1; y < resY; y++ )
    {
        {
            Vector2f p = box.min + Vector2f( 0.f, pixelSize * ( float( y ) + 0.5f ) );
            auto res = findProjectionOnPolyline2( p, polyline );
            prevPix = res.point;
        }
        for ( auto x = 1; x < resX; x++ )
        {
            Vector2f p = box.min + Vector2f( pixelSize * ( float( x ) + 0.5f ), pixelSize * ( float( y ) + 0.5f ) );
            auto res = findProjectionOnPolyline2( p, polyline );
            if ( ( res.point - prevPix ).lengthSq() > distLimitSq ||
                ( res.point - prevLine[x] ).lengthSq() > distLimitSq )
            {
                auto z = std::sqrt( res.distSq );
                edgePoints.emplace_back( p.x, p.y, z );
            }
            prevPix = res.point;
            prevLine[x] = res.point;
        }
    }
    return edgePoints;
}

namespace MarchingSquaresHelper
{

enum class NeighborDir
{
    X, Y, Count
};

// point between two neighbor voxels
struct SeparationPoint
{
    Vector2f position; // coordinate
    VertId vid; // any valid VertId is ok
    // each SeparationPointMap element has two SeparationPoint, it is not guaranteed that all three are valid (at least one is)
    // so there are some points present in map that are not valid
    explicit operator bool() const
    {
        return vid.valid();
    }
};

using SeparationPointSet = std::array<SeparationPoint, size_t( NeighborDir::Count )>;
using SeparationPointMap = ParallelHashMap<size_t, SeparationPointSet>;

// lookup table from
// http://paulbourke.net/geometry/polygonise/
using EdgeDirIndex = std::pair<int, NeighborDir>;
constexpr std::array<EdgeDirIndex, 4> cEdgeIndicesMap = {
   EdgeDirIndex{0,NeighborDir::X},
   EdgeDirIndex{0,NeighborDir::Y},
   EdgeDirIndex{2,NeighborDir::X},
   EdgeDirIndex{1,NeighborDir::Y}
};

const std::array<Vector2i, 4> cPixelNeighbors{
    Vector2i{0,0},
    Vector2i{1,0},
    Vector2i{0,1},
    Vector2i{1,1}
};

// contains indices in `cEdgeIndicesMap` (needed to capture vertices indices from SeparationPointMap)
// each to represent edge in result topology
using TopologyPlan = std::vector<int>;
const std::array<TopologyPlan, 16> cTopologyTable = {
    TopologyPlan{},
    TopologyPlan{1, 0},
    TopologyPlan{0, 3},
    TopologyPlan{1, 3},
    TopologyPlan{2, 1},
    TopologyPlan{2, 0},
    TopologyPlan{0, 1, 2, 3}, // undetermined
    TopologyPlan{2, 3},
    TopologyPlan{3, 2},
    TopologyPlan{1, 0, 3, 2}, // undetermined
    TopologyPlan{0, 2},
    TopologyPlan{1, 2},
    TopologyPlan{3, 1},
    TopologyPlan{3, 0},
    TopologyPlan{0, 1},
    TopologyPlan{}
};

SeparationPoint findSeparationPoint( const DistanceMap& dm, const Vector2i& p, NeighborDir dir, float isoValue )
{
    const auto v0 = dm.getValue( p.x, p.y );
    auto p1 = p;
    p1[int( dir )]++;
    if ( p1.x >= dm.resX() || p1.y >= dm.resY() )
        return {};
    const auto v1 = dm.getValue( p1.x, p1.y );
    if ( v0 == NOT_VALID_VALUE || v1 == NOT_VALID_VALUE )
        return {};
    const bool low0 = v0 < isoValue;
    const bool low1 = v1 < isoValue;
    if ( low0 == low1 )
        return {};

    const float ratio = std::abs( isoValue - v0 ) / std::abs( v1 - v0 );
    SeparationPoint res;
    res.position = ( 1.0f - ratio ) * Vector2f( p ) +
        ratio * Vector2f( p1 ) + Vector2f::diagonal( 0.5f );
    res.vid =0_v;// real number now is not important, only that it is valid
    return res;
}

}

Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float isoValue )
{
    using namespace MarchingSquaresHelper;
    MR_NAMED_TIMER( "distanceMapTo2DIsoPolyline" );
    const size_t resX = distMap.resX();
    const size_t resY = distMap.resY();
    if ( resX == 0 || resY == 0 )
        return {};

    const size_t size = resX * resY;
    auto toPos = [resX] ( size_t ind )->Vector2i
    {
        return { int( ind % resX ),int( ind / resX ) };
    };
    auto toId = [resX] ( const Vector2i& pos )->size_t
    {
        return pos.x + pos.y * size_t( resX );
    };

    SeparationPointMap hmap;
    const auto subcnt = hmap.subcnt();
    // find all separate points
    // fill map in parallel
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t i = 0; i < size; ++i )
        {
            auto pos = toPos( i );

            auto hashval = hmap.hash( i );
            if ( hmap.subidx( hashval ) != range.begin() )
                continue;

            SeparationPointSet set;
            bool atLeastOneOk = false;
            for ( int n = int( NeighborDir::X ); n < int( NeighborDir::Count ); ++n )
            {
                SeparationPoint separation = findSeparationPoint( distMap, pos, NeighborDir( n ), isoValue );
                if ( separation )
                {
                    set[n] = std::move( separation );
                    atLeastOneOk = true;
                }
            }
            if ( !atLeastOneOk )
                continue;
            hmap.insert( { i, set } );
        }
    } );

    // numerate verts in parallel (to have packed polyline as result, determined numeration independent of thread number)
    struct VertsNumeration
    {
        // explicit ctor to fix clang build with `vec.emplace_back( ind, 0 )`
        VertsNumeration( size_t ind, size_t num ) :initIndex{ ind }, numVerts{ num }{}
        size_t initIndex{ 0 };
        size_t numVerts{ 0 };
    };
    using PerThreadVertNumeration = std::vector<VertsNumeration>;
    tbb::enumerable_thread_specific<PerThreadVertNumeration> perThreadVertNumeration;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, size ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        auto& localNumeration = perThreadVertNumeration.local();
        localNumeration.emplace_back( range.begin(), 0 );
        auto& thisRangeNumeration = localNumeration.back().numVerts;
        for ( size_t ind = range.begin(); ind < range.end(); ++ind )
        {
            // as far as map is filled, change of each individual value should be safe
            auto mapIter = hmap.find( ind );
            if ( mapIter == hmap.cend() )
                continue;
            for ( auto& dir : mapIter->second )
                if ( dir )
                    dir.vid = VertId( thisRangeNumeration++ );
        }
    }, tbb::static_partitioner() );// static_partitioner to have bigger grains

    // organize vert numeration
    std::vector<VertsNumeration> resultVertNumeration;
    for ( auto& perThreadNum : perThreadVertNumeration )
    {
        // remove empty
        perThreadNum.erase( std::remove_if( perThreadNum.begin(), perThreadNum.end(),
            [] ( const auto& obj )
        {
            return obj.numVerts == 0;
        } ), perThreadNum.end() );
        if ( perThreadNum.empty() )
            continue;
        // accum not empty
        resultVertNumeration.insert( resultVertNumeration.end(),
            std::make_move_iterator( perThreadNum.begin() ), std::make_move_iterator( perThreadNum.end() ) );
    }
    // sort by voxel index
    std::sort( resultVertNumeration.begin(), resultVertNumeration.end(), [] ( const auto& l, const auto& r )
    {
        return l.initIndex < r.initIndex;
    } );

    auto getVertIndexShiftForPixelId = [&] ( size_t ind )
    {
        size_t shift = 0;
        for ( int i = 1; i < resultVertNumeration.size(); ++i )
        {
            if ( ind >= resultVertNumeration[i].initIndex )
                shift += resultVertNumeration[i - 1].numVerts;
        }
        return VertId( shift );
    };

    // update map with determined vert indices
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ),
    [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        hmap.with_submap( range.begin(), [&] ( const SeparationPointMap::EmbeddedSet& subSet )
        {
            // const_cast here is safe, we don't write to map, just change internal data
            for ( auto& [ind, set] : const_cast< SeparationPointMap::EmbeddedSet& >( subSet ) )
            {
                auto vertShift = getVertIndexShiftForPixelId( ind );
                for ( auto& sepPoint : set )
                    if ( sepPoint )
                        sepPoint.vid += vertShift;
            }
        } );
    } );

    // check neighbor iterator valid
    auto checkIter = [&] ( const auto& iter, int mode ) -> bool
    {
        switch ( mode )
        {
        case 0: // base pixel
            return iter != hmap.cend();
        case 1: // x + 1 pixel
        {
            if ( iter == hmap.cend() )
                return false;
            return bool( iter->second[int( NeighborDir::Y )] );
        }
        case 2: // y + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return bool( iter->second[int( NeighborDir::X )] );
        }
        default:
            return false;
        }
    };

    // Topology by table
    struct TopologyData
    {
        size_t initInd{ 0 }; // this is needed to have determined topology independent of threads number
        std::vector<std::array<VertId, 2>> lines;
    };

    using PerThreadTopologyData = std::vector<TopologyData>;
    tbb::enumerable_thread_specific<PerThreadTopologyData> topologyPerThread;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, size ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        // setup local triangulation
        auto& localTopologyData = topologyPerThread.local();
        localTopologyData.emplace_back();
        auto& thisLineData = localTopologyData.back();
        thisLineData.initInd = range.begin();
        auto& lines = thisLineData.lines;

        // cell data
        std::array<SeparationPointMap::const_iterator, 3> iters;
        std::array<bool, 3> iterStatus;
        unsigned char pixelConfiguration;
        for ( size_t ind = range.begin(); ind < range.end(); ++ind )
        {
            Vector2i basePos = toPos( ind );
            if ( basePos.x + 1 >= resX || basePos.y + 1 >= resY )
                continue;

            bool pixelValid = false;
            for ( int i = 0; i < iters.size(); ++i )
            {
                iters[i] = hmap.find( toId( basePos + cPixelNeighbors[i] ) );
                iterStatus[i] = checkIter( iters[i], i );
                if ( !pixelValid && iterStatus[i] )
                    pixelValid = true;
            }
            if ( !pixelValid )
                continue;
            pixelConfiguration = 0;
            [[maybe_unused]] bool atLeastOneNan = false;
            for ( int i = 0; i < cPixelNeighbors.size(); ++i )
            {
                auto pos = basePos + cPixelNeighbors[i];
                float value = distMap.getValue( pos.x, pos.y );
                if ( value == NOT_VALID_VALUE )
                {
                    pixelValid = false;
                    break;
                }
                if ( value >= isoValue )
                    continue;
                pixelConfiguration |= ( 1 << i );
            }

            if ( !pixelValid )
                continue;

            const auto& plan = cTopologyTable[pixelConfiguration];
            for ( int i = 0; i < plan.size(); i += 2 )
            {
                const auto& [interIndex0, dir0] = cEdgeIndicesMap[plan[i]];
                const auto& [interIndex1, dir1] = cEdgeIndicesMap[plan[i + 1]];
                assert( iterStatus[interIndex0] && iters[interIndex0]->second[int( dir0 )].vid );
                assert( iterStatus[interIndex1] && iters[interIndex1]->second[int( dir1 )].vid );

                lines.emplace_back( std::array<VertId, 2>{ 
                    iters[interIndex0]->second[int( dir0 )].vid,
                        iters[interIndex1]->second[int( dir1 )].vid
                } );
            }
        }
    } );

    // organize per thread topology
    std::vector<TopologyData> resTopologyData;
    for ( auto& threadTopologyData : topologyPerThread )
    {
        // remove empty
        threadTopologyData.erase( std::remove_if( threadTopologyData.begin(), threadTopologyData.end(),
            [] ( const auto& obj )
        {
            return obj.lines.empty();
        } ), threadTopologyData.end() );
        if ( threadTopologyData.empty() )
            continue;
        // accum not empty
        resTopologyData.insert( resTopologyData.end(),
            std::make_move_iterator( threadTopologyData.begin() ), std::make_move_iterator( threadTopologyData.end() ) );
    }
    // sort by pixel index
    std::sort( resTopologyData.begin(), resTopologyData.end(), [] ( const auto& l, const auto& r )
    {
        return l.initInd < r.initInd;
    } );

    Polyline2 polyline;
    size_t pointsSize = resultVertNumeration.empty() ? 0 : size_t( getVertIndexShiftForPixelId( size ) ) + resultVertNumeration.back().numVerts;
    polyline.points.resize( pointsSize );
    polyline.topology.vertResize( polyline.points.size() );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        hmap.with_submap( range.begin(), [&] ( const SeparationPointMap::EmbeddedSet& subSet )
        {
            for ( auto& separation : subSet )
            {
                for ( int i = int( NeighborDir::X ); i < int( NeighborDir::Count ); ++i )
                    if ( separation.second[i].vid.valid() )
                        polyline.points[separation.second[i].vid] = separation.second[i].position;
            }
        } );
    } );

    for ( auto& [ind, lines] : resTopologyData )
    {
        for ( const auto& line : lines )
            polyline.topology.makeEdge( line[0], line[1] );
    }
    assert( polyline.topology.isConsistentlyOriented() );

    return polyline;
}

Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, const ContourToDistanceMapParams& params, float isoValue )
{
    Polyline2 res = distanceMapTo2DIsoPolyline( distMap, isoValue );

    BitSetParallelFor( res.topology.getValidVerts(), [&] ( VertId v )
    {
        res.points[v] = params.toWorld( res.points[v] );
    } );

    return res;
}

std::pair<MR::Polyline2, MR::AffineXf3f> distanceMapTo2DIsoPolyline( const DistanceMap& distMap,
    const DistanceMapToWorld& params, float isoValue, bool useDepth /* = false */)
{
    Polyline2 resContours = distanceMapTo2DIsoPolyline( distMap, isoValue );

    const float depth = useDepth ? isoValue : 0.F;
    const Matrix3f m = Matrix3f::fromColumns( params.pixelXVec, params.pixelYVec, params.direction );
    const AffineXf3f resXf{ m, params.toWorld( 0.F, 0.F, depth ) };
    const AffineXf3f resInv = resXf.inverse();

    BitSetParallelFor( resContours.topology.getValidVerts(), [&] ( VertId v )
    {
        const Vector3f p = params.toWorld( resContours.points[v].x, resContours.points[v].y, 0.F );
        const Vector3f pInv = resInv( p );
        resContours.points[v] = Vector2f{ pInv.x, pInv.y };
    } );

    return { resContours, resXf };
}

Polyline2 distanceMapTo2DIsoPolyline( const DistanceMap& distMap, float pixelSize, float isoValue )
{
    ContourToDistanceMapParams params;
    params.pixelSize = Vector2f{ pixelSize, pixelSize };
    params.resolution = Vector2i{ int( distMap.resX() ), int( distMap.resY() ) };
    auto res = distanceMapTo2DIsoPolyline( distMap, params, isoValue );
    return res;
}

void DistanceMap::negate()
{
    for( auto & v : data_ )
        if ( v != NOT_VALID_VALUE )
            v = -v;
}

// boolean operators
DistanceMap DistanceMap::max( const DistanceMap& rhs ) const
{
    DistanceMap res( resX(), resY() );
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( val )
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, std::max( *val, *valrhs ) );
                    }
                    else
                    {
                        res.set( iX, iY, *val );
                    }
                }
                else
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, *valrhs );
                    }
                }
            }
            else
            {
                res.set( iX, iY, *val );
            }
        }
    }
    return res;
}

const DistanceMap& DistanceMap::mergeMax( const DistanceMap& rhs)
{
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for ( auto iY = 0; iY < resY(); iY++ )
        {
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( valrhs )
                {
                    const auto val = get( iX, iY );
                    if ( !val || ( *val < *valrhs ) )
                        set( iX, iY, *valrhs );
                }
            }
        }
    }
    return *this;
}

DistanceMap DistanceMap::min( const DistanceMap& rhs) const
{
    DistanceMap res( resX(), resY() );
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for ( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( val )
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, std::min( *val, *valrhs ) );
                    }
                    else
                    {
                        res.set( iX, iY, *val );
                    }
                }
                else
                {
                    if ( valrhs )
                    {
                        res.set( iX, iY, *valrhs );
                    }
                }
            }
            else
            {
                res.set( iX, iY, *val );
            }
        }
    }
    return res;
}

const DistanceMap& DistanceMap::mergeMin( const DistanceMap& rhs )
{
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            if ( iX < rhs.resX() && iY < rhs.resY() )
            {
                const auto valrhs = rhs.get( iX, iY );
                if ( valrhs )
                {
                    const auto val = get( iX, iY );
                    if ( !val || ( *val > *valrhs ) )
                        set( iX, iY, *valrhs );
                }
            }
        }
    }
    return *this;
}

DistanceMap DistanceMap::operator-( const DistanceMap& rhs) const
{
    DistanceMap res( resX(), resY() );
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( val )
            {
                if ( iX < rhs.resX() && iY < rhs.resY() )
                {
                    const auto valrhs = rhs.get( iX, iY );
                    if ( valrhs )
                    {
                        res.set( iX, iY, *val - *valrhs );
                    }
                }
                else
                {
                    res.set( iX, iY, *val );
                }
            }
        }
    }
    return res;
}

const DistanceMap& DistanceMap::operator-=( const DistanceMap& rhs )
{
    for( auto iX = 0; iX < resX(); iX++ )
    {
        for( auto iY = 0; iY < resY(); iY++ )
        {
            const auto val = get( iX, iY );
            if ( val )
            {
                if ( iX < rhs.resX() && iY < rhs.resY() )
                {
                    const auto valrhs = rhs.get( iX, iY );
                    if ( valrhs )
                    {
                        set( iX, iY, *val - *valrhs );
                    }
                }
            }
        }
    }
    return *this;
}

DistanceMap DistanceMap::getDerivativeMap() const
{
    auto XYmaps = getXYDerivativeMaps();
    return combineXYderivativeMaps( XYmaps );
}

std::pair< DistanceMap, DistanceMap > DistanceMap::getXYDerivativeMaps() const
{
    auto res = std::make_pair( DistanceMap( resX(), resY() ), DistanceMap( resX(), resY() ) );
    DistanceMap& dx = res.first;
    DistanceMap& dy = res.second;
    if (resX() < 3 || resY() < 3)
    {
        return res;
    }
    tbb::parallel_for( tbb::blocked_range<size_t>( 1, resX() - 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t x = range.begin(); x < range.end(); x++ )
        {
            for ( size_t y = 1; y + 1 < resY(); y++ )
            {
                const auto val = get( x, y );
                if ( val )
                {
                    const auto valxpos = get( x + 1, y );
                    const auto valxneg = get( x - 1, y );
                    if ( valxpos )
                        if ( valxneg )
                            dx.set( x, y, ( ( *valxpos ) - ( *valxneg ) ) / 2.f );
                        else
                            dx.set( x, y, ( *valxpos ) - ( *val ) );
                    else
                        if ( valxneg )
                            dx.set( x, y, ( *val ) - ( *valxneg ) );
                        else
                            dx.unset( x, y );

                    const auto valypos = get( x, y + 1 );
                    const auto valyneg = get( x, y - 1 );
                    if ( valypos )
                        if ( valyneg )
                            dy.set( x, y, ( ( *valypos ) - ( *valyneg ) ) / 2.f );
                        else
                            dy.set( x, y, ( *valypos ) - ( *val ) );
                    else
                        if ( valyneg )
                            dy.set( x, y, ( *val ) - ( *valyneg ) );
                        else
                            dy.unset( x, y );
                }
            }
        }
    } );
    return res;
}

DistanceMap combineXYderivativeMaps( std::pair<DistanceMap, DistanceMap> XYderivativeMaps )
{
    assert( XYderivativeMaps.first.resX() == XYderivativeMaps.second.resX() );
    assert( XYderivativeMaps.first.resY() == XYderivativeMaps.second.resY() );

    DistanceMap dMap( XYderivativeMaps.first.resX(), XYderivativeMaps.second.resY() );
    const auto& dx = XYderivativeMaps.first;
    const auto& dy = XYderivativeMaps.second;

    if ( dx.resX() < 3 || dx.resY() < 3 )
    {
        return dMap;
    }
    // fill the central area
    tbb::parallel_for( tbb::blocked_range<size_t>( 1, dx.resX() - 1 ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t x = range.begin(); x < range.end(); x++ )
        {
            for ( size_t y = 1; y + 1 < dx.resY(); y++ )
            {
                const auto valX = dx.get( x, y );
                const auto valY = dy.get( x, y );
                
                if (valX)
                {
                    if (valY)
                    {
                        dMap.set( x, y, std::sqrt( ( *valX ) * ( *valX ) + ( *valY ) * ( *valY ) ) );
                    }
                    else
                    {
                        dMap.set( x, y, *valY );
                    }
                }
                else
                {
                    if (valY)
                    {
                        dMap.set( x, y, *valY );
                    }
                    else
                    {
                        dMap.unset( x, y );
                    }
                }
            }
        }
    } );
    return dMap;
}

std::vector<std::pair<size_t, size_t>>  DistanceMap::getLocalMaximums() const
{
    typedef std::vector<std::pair<size_t, size_t>> LocalMaxAcc;
    LocalMaxAcc acc;
    // ignore border cases (y==0 or y==resY()-1)
    auto min = tbb::parallel_reduce( tbb::blocked_range<size_t>( resX(), numPoints() - resX() ), acc,
    [&] ( const tbb::blocked_range<size_t> range, LocalMaxAcc localAcc )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            // ignore border cases (x==0 or x==resX()-1)
            if ( ( i % resX() == 0 ) || ( ( i + 1 ) % resX() == 0 ) )
            {
                continue;
            }
            const auto& val = data_[i];
            if ( ( data_[i - 1 - resX()] < val ) &&
                 ( data_[i - 1] < val ) &&
                 ( data_[i - 1 + resX()] < val ) &&
                 ( data_[i - resX()] < val ) &&
                 ( data_[i + resX()] < val ) &&
                 ( data_[i + 1 - resX()] < val ) &&
                 ( data_[i + 1] < val ) &&
                 ( data_[i + 1 + resX()] < val ) )
            {
                localAcc.push_back( { i % resX(), i / resX() } );
            }
        }
        return localAcc;
    },
    [&] ( const LocalMaxAcc& a, const LocalMaxAcc& b )
    {
        LocalMaxAcc res = a;
        res.insert( res.end(), b.begin(), b.end() );
        return res;
    } );

    return acc;
}

std::pair<float, float> DistanceMap::getMinMaxValues() const
{
    struct MinMax
    {
        float min;
        float max;
        size_t minI;
        size_t maxI;
    };
    MinMax minElem{ std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 0, 0 };
    auto minmax = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, numPoints() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MinMax curMinMax )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            auto val = get( i );
            if ( val )
            {
                if ( *val < curMinMax.min )
                {
                    curMinMax.min = *val;
                    curMinMax.minI = i;
                }
                if ( *val > curMinMax.max )
                {
                    curMinMax.max = *val;
                    curMinMax.maxI = i;
                }
            }
        }
        return curMinMax;
    },
    [&] ( const MinMax& a, const MinMax& b )
    {
        MinMax res;
        if ( a.min < b.min )
        {
            res.min = a.min;
            res.minI = a.minI;
        }
        else
        {
            res.min = b.min;
            res.minI = b.minI;
        }
        if ( a.max > b.max )
        {
            res.max = a.max;
            res.maxI = a.maxI;
        }
        else
        {
            res.max = b.max;
            res.maxI = b.maxI;
        }
        return res;
    } );

    return { minmax.min, minmax.max };
}

std::pair<size_t, size_t> DistanceMap::getMinIndex() const
{
    typedef std::pair<float, size_t> MinElem;
    MinElem minElem{ std::numeric_limits<float>::max(), 0 };
    auto min = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, numPoints() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MinElem curMin )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            auto val = get( i );
            if ( val && ( *val < curMin.first ) )
            {
                curMin.first = *val;
                curMin.second = i;
            }
        }
        return curMin;
    },
    [&] ( const MinElem& a, const MinElem& b )
    {
        return a.first < b.first ? a : b;
    } );

    return { min.second / resY(), min.second % resY() };
}

std::pair<size_t, size_t> DistanceMap::getMaxIndex() const
{
    typedef std::pair<float, size_t> MaxElem;
    MaxElem minElem{ -std::numeric_limits<float>::max(), 0 };
    auto max = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, numPoints() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MaxElem curMin )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            auto val = get( i );
            if ( val && ( *val > curMin.first ) )
            {
                curMin.first = *val;
                curMin.second = i;
            }
        }
        return curMin;
    },
    [&] ( const MaxElem& a, const MaxElem& b )
    {
        return a.first > b.first ? a : b;
    } );

    return { max.second / resY(), max.second % resY() };
}

Polyline2 contourUnion( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside )
{
    assert( params.withSign );
    auto mapA = distanceMapFromContours( contoursA, params );
    auto mapB = distanceMapFromContours( contoursB, params );
    mapA.mergeMin( mapB );
    return distanceMapTo2DIsoPolyline( mapA, params, offsetInside );
}

Polyline2 contourIntersection( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside )
{
    assert( params.withSign );
    auto mapA = distanceMapFromContours( contoursA, params );
    auto mapB = distanceMapFromContours( contoursB, params );
    mapA.mergeMax( mapB );
    return distanceMapTo2DIsoPolyline( mapA, params, offsetInside );
}

Polyline2 contourSubtract( const Polyline2& contoursA, const Polyline2& contoursB,
    const ContourToDistanceMapParams& params, float offsetInside )
{
    assert( params.withSign );
    auto mapA = distanceMapFromContours( contoursA, params );
    auto mapB = distanceMapFromContours( contoursB, params );
    mapB.negate();
    mapA.mergeMax( mapB );
    return distanceMapTo2DIsoPolyline( mapA, params, offsetInside );
}

Polyline2 polylineOffset( const Polyline2& polyline, float pixelSize, float offset )
{
    MR_TIMER

    assert( offset > 0.f );

    const auto box = polyline.computeBoundingBox();
    const auto size = box.size();

    const auto padding = offset + 2 * pixelSize;

    ContourToDistanceMapParams params;
    params.pixelSize = {
        pixelSize,
        pixelSize,
    };
    params.resolution = {
        int( ( size.x + 2 * padding ) / pixelSize ),
        int( ( size.y + 2 * padding ) / pixelSize ),
    };
    params.orgPoint = {
        box.min.x - padding,
        box.min.y - padding,
    };

    ContoursDistanceMapOptions options;
    //compute precise distances only in the cells crossed by offset-isoline
    options.maxDist = offset + pixelSize;
    options.minDist = std::max( offset - pixelSize, 0.0f );

    const auto distanceMap = distanceMapFromContours( polyline, params, options );

    auto isoline = distanceMapTo2DIsoPolyline( distanceMap, offset );

    DistanceMapToWorld distanceMapToWorld( params );
    for ( auto& p : isoline.points )
        p = Vector2f( distanceMapToWorld.toWorld( p.x, p.y, 0 ) );

    return isoline;
}

} //namespace MR
