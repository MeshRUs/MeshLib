#pragma once

#include "MREdgePaths.h"
#include "MRBitSet.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRphmap.h"
#include "MRMacros.h"
#include <queue>

namespace MR
{

/// \ingroup SurfacePathGroup
/// \{

/// information associated with each vertex by the paths builder
struct VertPathInfo
{
    // edge from this vertex to its predecessor in the forest
    EdgeId back;
    // best summed metric to reach this vertex
    float metric = FLT_MAX;

    bool isStart() const { return !back.valid(); }
};

// ParallelHashMap here was significantly slower (especially in debug version of tunnel detection),
// probably because of more allocations
using VertPathInfoMap = HashMap<VertId, VertPathInfo>;

/// the class is responsible for finding smallest metric edge paths on a mesh
template<class MetricToPenalty>
class EdgePathsBuilderT
{
public:
    EdgePathsBuilderT( const MeshTopology & topology, const EdgeMetric & metric );

    /// compares proposed metric with best value known for startVert;
    /// if proposed metric is smaller then adds it in the queue and returns true
    bool addStart( VertId startVert, float startMetric );

    /// information about just reached vertex (with final metric value)
    struct ReachedVert
    {
        VertId v;

        /// edge from this vertex to its predecessor in the forest (if this vertex is not start)
        EdgeId backward;

        /// not reached vertices are ordered in priority queue by their penalty (with the smallest value on top);
        /// penalty is equal to metric in ordinary Dijkstra, or equal to (metric + target distance lower bound) in A*
        float penalty = FLT_MAX;

        /// summed metric to reach this vertex
        float metric = FLT_MAX;
    };

    /// include one more vertex in the final forest, returning vertex-info for the newly reached vertex;
    /// returns invalid VertId in v-field if no more vertices left
    ReachedVert reachNext();

    /// adds steps for all origin ring edges of the reached vertex;
    /// returns true if at least one step was added
    bool addOrgRingSteps( const ReachedVert & rv );

    /// the same as reachNext() + addOrgRingSteps()
    ReachedVert growOneEdge();

public:
    /// returns true if further edge forest growth is impossible
    bool done() const { return nextSteps_.empty(); }

    /// returns path length till the next candidate vertex or maximum float value if all vertices have been reached
    float doneDistance() const { return nextSteps_.empty() ? FLT_MAX : nextSteps_.top().penalty; }

    /// gives read access to the map from vertex to path to it
    const VertPathInfoMap & vertPathInfoMap() const { return vertPathInfoMap_; }

    /// returns one element from the map (or nullptr if the element is missing)
    const VertPathInfo * getVertInfo( VertId v ) const;

    /// returns the path in the forest from given vertex to one of start vertices
    EdgePath getPathBack( VertId backpathStart ) const;

protected:
    MR_NO_UNIQUE_ADDRESS MetricToPenalty metricToPenalty_;

private:
    const MeshTopology & topology_;
    EdgeMetric metric_;
    VertPathInfoMap vertPathInfoMap_;

    struct CandidateVert
    {
        VertId v;
        /// best penalty to reach this vertex
        float penalty = FLT_MAX;

        /// smaller penalty to be the first
        friend bool operator <( const CandidateVert & a, const CandidateVert & b )
        {
            return a.penalty > b.penalty;
        }
    };
    std::priority_queue<CandidateVert> nextSteps_;

    /// compares proposed step with the value known for org( c.back );
    /// if proposed step is smaller then adds it in the queue and returns true;
    /// otherwise if the known metric to org( c.back ) is already not greater than returns false
    bool addNextStep_( const VertPathInfo & c );
};

/// the vertices in the queue are ordered by their metric from a start location
struct TrivialMetricToPenalty
{
    float operator()( float metric, VertId ) const { return metric; }
};

using EdgePathsBuilder = EdgePathsBuilderT<TrivialMetricToPenalty>;

template<class MetricToPenalty>
EdgePathsBuilderT<MetricToPenalty>::EdgePathsBuilderT( const MeshTopology & topology, const EdgeMetric & metric )
    : topology_( topology )
    , metric_( metric )
{
}

template<class MetricToPenalty>
bool EdgePathsBuilderT<MetricToPenalty>::addStart( VertId startVert, float startMetric )
{
    auto & vi = vertPathInfoMap_[startVert];
    if ( vi.metric > startMetric )
    {
        vi = { EdgeId{}, startMetric };
        nextSteps_.push( CandidateVert{ startVert, metricToPenalty_( startMetric, startVert ) } );
        return true;
    }
    return false;
}

template<class MetricToPenalty>
const VertPathInfo * EdgePathsBuilderT<MetricToPenalty>::getVertInfo( VertId v ) const
{
    auto it = vertPathInfoMap_.find( v );
    return ( it != vertPathInfoMap_.end() ) ? &it->second : nullptr;
}

template<class MetricToPenalty>
EdgePath EdgePathsBuilderT<MetricToPenalty>::getPathBack( VertId v ) const
{
    EdgePath res;
    for (;;)
    {
        auto it = vertPathInfoMap_.find( v );
        if ( it == vertPathInfoMap_.end() )
        {
            assert( false );
            break;
        }
        auto & vi = it->second;
        if ( vi.isStart() )
            break;
        res.push_back( vi.back );
        v = topology_.dest( vi.back );
    }
    return res;
}

template<class MetricToPenalty>
bool EdgePathsBuilderT<MetricToPenalty>::addNextStep_( const VertPathInfo & c )
{
    if ( !( c.metric < FLT_MAX ) )
        return false; // maximal or infinity metric means that this path shall be skipped
    const auto vert = topology_.org( c.back );
    auto & vi = vertPathInfoMap_[vert];
    if ( vi.metric > c.metric )
    {
        vi = c;
        nextSteps_.push( CandidateVert{ vert, metricToPenalty_( c.metric, vert ) } );
        return true;
    }
    return false;
}

template<class MetricToPenalty>
bool EdgePathsBuilderT<MetricToPenalty>::addOrgRingSteps( const ReachedVert & rv )
{
    bool aNextStepAdded = false;
    if ( !rv.v )
    {
        assert( !rv.backward );
        return aNextStepAdded;
    }
    const float orgMetric = rv.metric;
    const EdgeId back = rv.backward ? rv.backward : topology_.edgeWithOrg( rv.v );
    for ( EdgeId e : orgRing( topology_, back ) )
    {
        VertPathInfo c;
        c.back = e.sym();
        c.metric = orgMetric + metric_( e );
        aNextStepAdded = addNextStep_( c ) || aNextStepAdded;
    }
    return aNextStepAdded;
}

template<class MetricToPenalty>
auto EdgePathsBuilderT<MetricToPenalty>::reachNext() -> ReachedVert
{
    while ( !nextSteps_.empty() )
    {
        const auto c = nextSteps_.top();
        nextSteps_.pop();
        auto & vi = vertPathInfoMap_[c.v];
        if ( metricToPenalty_( vi.metric, c.v ) < c.penalty )
        {
            // shorter path to the vertex was found
            continue;
        }
        assert( metricToPenalty_( vi.metric, c.v ) == c.penalty );
        return { .v = c.v, .backward = vi.back, .penalty = c.penalty, .metric = vi.metric };
    }
    return {};
}

template<class MetricToPenalty>
auto EdgePathsBuilderT<MetricToPenalty>::growOneEdge() -> ReachedVert
{
    auto res = reachNext();
    addOrgRingSteps( res );
    return res;
}

/// the vertices in the queue are ordered by the sum of their metric from a start location and the
/// lower bound of a path till target point (A* heuristic)
struct MetricToAStarPenalty
{
    const VertCoords * points = nullptr;
    Vector3f target;
    float operator()( float metric, VertId v ) const
    {
        return metric + ( (*points)[v] - target ).length();
    }
};

/// the class is responsible for finding shortest edge paths on a mesh in Euclidean metric
/// using A* heuristics
class EdgePathsAStarBuilder: public EdgePathsBuilderT<MetricToAStarPenalty>
{
public:
    EdgePathsAStarBuilder( const Mesh & mesh, VertId target, VertId start ) :
        EdgePathsBuilderT( mesh.topology, edgeLengthMetric( mesh ) )
    {
        metricToPenalty_.points = &mesh.points;
        metricToPenalty_.target = mesh.points[target];
        addStart( start, 0 );
    }
    EdgePathsAStarBuilder( const Mesh & mesh, const MeshTriPoint & target, const MeshTriPoint & start ) :
        EdgePathsBuilderT( mesh.topology, edgeLengthMetric( mesh ) )
    {
        metricToPenalty_.points = &mesh.points;
        metricToPenalty_.target = mesh.triPoint( target );
        const auto startPt = mesh.triPoint( start );
        mesh.topology.forEachVertex( start, [&]( VertId v )
        {
            addStart( v, ( mesh.points[v] - startPt ).length() );
        } );
    }
};

/// \}

} // namespace MR
