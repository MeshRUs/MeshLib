#include "MRMultiwayICP.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMultiwayICP.h"

#include <span>

using namespace MR;

REGISTER_AUTO_CAST( ICPMethod )
REGISTER_AUTO_CAST( ICPMode )
REGISTER_AUTO_CAST( MeshOrPointsXf )
REGISTER_AUTO_CAST( MultiwayICP )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST2( MultiwayICPSamplingParameters::CascadeMode, MRMultiwayICPSamplingParametersCascadeMode )
REGISTER_VECTOR_LIKE( MRVectorAffineXf3f, AffineXf3f )

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

MRMultiwayICPSamplingParameters mrMultiwayIcpSamplingParametersNew( void )
{
    static const MultiwayICPSamplingParameters def;
    return {
        COPY_FROM( def, samplingVoxelSize )
        COPY_FROM( def, maxGroupSize )
        .cascadeMode = auto_cast( def.cascadeMode ),
        .cb = nullptr,
    };
}

MRMultiwayICP* mrMultiwayICPNew( const MRMeshOrPointsXf** objects_, size_t objectsNum, const MRMultiwayICPSamplingParameters* samplingParams_ )
{
    ICPObjects objectsVec;
    objectsVec.reserve( objectsNum );

    for ( auto i = 0; i < objectsNum; ++i )
        objectsVec.push_back( auto_cast( *objects_[i] ) );

    MultiwayICPSamplingParameters samplingParams;
    if ( samplingParams_ )
    {
        const auto& src = *samplingParams_;
        samplingParams = {
            COPY_FROM( src, samplingVoxelSize )
            COPY_FROM( src, maxGroupSize )
            .cascadeMode = static_cast<MultiwayICPSamplingParameters::CascadeMode>( src.cascadeMode ),
            COPY_FROM( src, cb )
        };
    }

    RETURN_NEW( MultiwayICP( objectsVec, samplingParams ) );
}

MRVectorAffineXf3f* mrMultiwayICPCalculateTransformations( MRMultiwayICP* mwicp_, MRProgressCallback cb )
{
    ARG( mwicp );
    RETURN_NEW_VECTOR( mwicp.calculateTransformations( cb ).vec_ );
}

bool mrMultiwayICPResamplePoints( MRMultiwayICP* mwicp_, const MRMultiwayICPSamplingParameters* samplingParams_ )
{
    ARG( mwicp );

    MultiwayICPSamplingParameters samplingParams;
    if ( samplingParams_ )
    {
        const auto& src = *samplingParams_;
        samplingParams = {
            COPY_FROM( src, samplingVoxelSize )
            COPY_FROM( src, maxGroupSize )
            .cascadeMode = auto_cast( src.cascadeMode ),
            COPY_FROM( src, cb )
        };
    }

    return mwicp.resamplePoints( samplingParams );
}

bool mrMultiwayICPUpdateAllPointPairs( MRMultiwayICP* mwicp_, MRProgressCallback cb )
{
    ARG( mwicp );
    return mwicp.updateAllPointPairs( cb );
}

void mrMultiwayICPSetParams( MRMultiwayICP* mwicp_, const MRICPProperties* prop_ )
{
    ARG( mwicp );

    const ICPProperties prop {
        .method = auto_cast( prop_->method ),
        COPY_FROM( *prop_, p2plAngleLimit )
        COPY_FROM( *prop_, p2plScaleLimit )
        COPY_FROM( *prop_, cosThreshold )
        COPY_FROM( *prop_, distThresholdSq )
        COPY_FROM( *prop_, farDistFactor )
        .icpMode = auto_cast( prop_->icpMode ),
        .fixedRotationAxis = auto_cast( prop_->fixedRotationAxis ),
        COPY_FROM( *prop_, iterLimit )
        COPY_FROM( *prop_, badIterStopCount )
        COPY_FROM( *prop_, exitVal )
        COPY_FROM( *prop_, mutualClosest )
    };

    mwicp.setParams( prop );
}

float mrMultiWayICPGetMeanSqDistToPoint( const MRMultiwayICP* mwicp_, const double* value )
{
    ARG( mwicp );
    return mwicp.getMeanSqDistToPoint( value ? std::optional( *value ) : std::nullopt );
}

float mrMultiWayICPGetMeanSqDistToPlane( const MRMultiwayICP* mwicp_, const double* value )
{
    ARG( mwicp );
    return mwicp.getMeanSqDistToPlane( value ? std::optional( *value ) : std::nullopt );
}

size_t mrMultiWayICPGetNumSamples( const MRMultiwayICP* mwicp_ )
{
    ARG( mwicp );
    return mwicp.getNumSamples();
}

size_t mrMultiWayICPGetNumActivePairs( const MRMultiwayICP* mwicp_ )
{
    ARG( mwicp );
    return mwicp.getNumActivePairs();
}

void mrMultiwayICPFree( MRMultiwayICP* mwicp_ )
{
    ARG_PTR( mwicp );
    delete mwicp;
}
