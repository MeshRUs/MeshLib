#include "MRObjectPoints.h"
#include "MRObjectFactory.h"
#include "MRMeshToPointCloud.h"
#include "MRObjectMesh.h"
#include "MRRegionBoundary.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRJson.h"

namespace MR
{

MR_ADD_CLASS_FACTORY( ObjectPoints )

ObjectPoints::ObjectPoints( const ObjectMesh& objMesh, bool saveNormals/*=true*/ )
{
    if ( !objMesh.mesh() )
        return;

    const auto verts = getInnerVerts( objMesh.mesh()->topology, objMesh.getSelectedFaces() );
    setPointCloud( std::make_shared<PointCloud>( meshToPointCloud( *objMesh.mesh(), saveNormals, verts.count() > 0 ? &verts : nullptr) ) );
    setName( objMesh.name() + " Points" );
    setVertsColorMap( objMesh.getVertsColorMap() );
    setFrontColor( objMesh.getFrontColor( true ), true );
    setFrontColor( objMesh.getFrontColor( false ), false );
    setBackColor( objMesh.getBackColor() );
    setColoringType( objMesh.getColoringType() );
}

std::vector<std::string> ObjectPoints::getInfoLines() const
{
    std::vector<std::string> res = ObjectPointsHolder::getInfoLines();

    if ( points_ )
    {
        if ( points_->normals.empty() )
            res.push_back( "points: " );
        else
            res.push_back( "points with normals: " );
        const auto nValidPoints = numValidPoints();
        res.back() += std::to_string( nValidPoints );

        const auto nSelectedPoints = numSelectedPoints();
        if( nSelectedPoints )
            res.back() += " / " + std::to_string( nSelectedPoints ) + " selected";

        if( nValidPoints < points_->points.size() )
            res.back() += " / " + std::to_string( points_->points.size() ) + " size";

        if( points_->points.size() < points_->points.capacity() )
            res.back() += " / " + std::to_string( points_->points.capacity() ) + " capacity";

        if ( !vertsColorMap_.empty() )
        {
            res.push_back( "colors: " + std::to_string( vertsColorMap_.size() ) );
            if ( vertsColorMap_.size() < vertsColorMap_.capacity() )
                res.back() += " / " + std::to_string( vertsColorMap_.capacity() ) + " capacity";
        }

        res.push_back( "max rendered points: " +
            ( getMaxRenderingPoints() == ObjectPoints::MaxRenderingPointsUnlimited ?
              "unlimited" : std::to_string( getMaxRenderingPoints() ) ) );

        boundingBoxToInfoLines_( res );
    }
    else
        res.push_back( "no points" );

    return res;
}

void ObjectPoints::setDirtyFlags( uint32_t mask, bool invalidateCaches )
{
    ObjectPointsHolder::setDirtyFlags( mask, invalidateCaches );
    if ( points_ )
    {
        if ( mask & DIRTY_POSITION || mask & DIRTY_FACE )
            pointsChangedSignal( mask );
        if ( mask & DIRTY_RENDER_NORMALS )
            normalsChangedSignal( mask );
    }
}

std::shared_ptr<Object> ObjectPoints::clone() const
{
    auto res = std::make_shared<ObjectPoints>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = std::make_shared<PointCloud>( *points_ );
    return res;
}

std::shared_ptr<Object> ObjectPoints::shallowClone() const
{
    auto res = std::make_shared<ObjectPoints>( ProtectedStruct{}, *this );
    if ( points_ )
        res->points_ = points_;
    return res;
}

void ObjectPoints::swapPointCloud( std::shared_ptr< PointCloud >& points )
{
    if ( points == points_ )
        return;
    points_.swap( points );
    setDirtyFlags( DIRTY_ALL );
}

void ObjectPoints::swapBase_( Object& other )
{
    if ( auto otherPoints = other.asType<ObjectPoints>() )
        std::swap( *this, *otherPoints );
    else
        assert( false );
}

void ObjectPoints::swapSignals_( Object& other )
{
    ObjectPointsHolder::swapSignals_( other );
    if ( auto otherPoints = other.asType<ObjectPoints>() )
    {
        std::swap( pointsChangedSignal, otherPoints->pointsChangedSignal );
        std::swap( normalsChangedSignal, otherPoints->normalsChangedSignal );
    }
    else
        assert( false );
}

void ObjectPoints::serializeFields_( Json::Value& root ) const
{
    ObjectPointsHolder::serializeFields_( root );

    root["Type"].append( ObjectPoints::TypeName() );
}

std::shared_ptr<ObjectPoints> merge( const std::vector<std::shared_ptr<ObjectPoints>>& objsPoints )
{
    MR_TIMER
    auto pointCloud = std::make_shared<PointCloud>();
    auto& points = pointCloud->points;

    bool allWithNormals = true;
    bool anyWithColors = false;
    for ( const auto& obj : objsPoints )
    {
        const auto & pc = obj->pointCloud();
        if ( !pc || !pc->validPoints.any() )
            continue;
        if ( !pc->hasNormals() )
            allWithNormals = false;
        if ( ( obj->getColoringType() == ColoringType::VertsColorMap ) &&
             ( obj->getVertsColorMap().size() > int( obj->pointCloud()->validPoints.find_last() ) ) )
            anyWithColors = true;
    }
    const VertNormals emptyNormals;

    VertColors colors;
    for ( const auto& obj : objsPoints )
    {
        if ( !obj->pointCloud() )
            continue;

        VertMap vertMap{};
        pointCloud->addPartByMask( *obj->pointCloud(), obj->pointCloud()->validPoints, { .src2tgtVerts = &vertMap },
            allWithNormals ? nullptr : &emptyNormals );

        const bool withColors = ( obj->getColoringType() == ColoringType::VertsColorMap ) &&
            ( obj->getVertsColorMap().size() > int( obj->pointCloud()->validPoints.find_last() ) ) ;
        const auto& objColors = obj->getVertsColorMap();
        if ( anyWithColors )
            colors.resize( size_t( vertMap.back() ) + 1, obj->getFrontColor( true ) );
        auto worldXf = obj->worldXf();
        auto normalsMatrix = worldXf.A.inverse().transposed();
        ParallelFor( vertMap, [&] ( VertId v )
        {
            auto vInd = vertMap[v];
            if ( !vInd.valid() )
                return;
            points[vInd] = worldXf( points[vInd] );
            if ( allWithNormals )
                pointCloud->normals[vInd] = ( normalsMatrix * pointCloud->normals[vInd] ).normalized();
            if ( withColors )
                colors[vInd] = objColors[v];
        } );
    }

    auto objectPoints = std::make_shared<ObjectPoints>();
    objectPoints->setPointCloud( std::move( pointCloud ) );
    if ( !colors.empty() )
    {
        objectPoints->setVertsColorMap( std::move( colors ) );
        objectPoints->setColoringType( ColoringType::VertsColorMap );
    }
    return objectPoints;
}

} //namespace MR
