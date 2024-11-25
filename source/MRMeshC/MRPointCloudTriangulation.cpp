#include "MRPointCloudTriangulation.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "detail/TypeCast.h"

#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointCloudTriangulation.h"
#include "MRMesh/MRMesh.h"

#define COPY_FROM( obj, field ) . field = ( obj ). field ,

using namespace MR;

REGISTER_AUTO_CAST( PointCloud )
REGISTER_AUTO_CAST( Mesh )

MRTriangulationParameters mrTriangulationParametersNew( void )
{
    static const TriangulationParameters def;
    return {
        COPY_FROM( def, numNeighbours )
        COPY_FROM( def, radius )
        COPY_FROM( def, critAngle )
        COPY_FROM( def, boundaryAngle )
        COPY_FROM( def, critHoleLength )
        COPY_FROM( def, automaticRadiusIncrease )
        .searchNeighbors = auto_cast( def.searchNeighbors )
    };
}

MRMesh* mrTriangulatePointCloud( const MRPointCloud* pointCloud_, const MRTriangulationParameters* params_ )
{
    ARG( pointCloud );

    TriangulationParameters params;
    if ( params_ )
    {
        const auto& src = *params_;
        params = {
            COPY_FROM( src, numNeighbours )
            COPY_FROM( src, radius )
            COPY_FROM( src, critAngle )
            COPY_FROM( src, boundaryAngle )
            COPY_FROM( src, critHoleLength )
            COPY_FROM( src, automaticRadiusIncrease )
            .searchNeighbors = auto_cast( src.searchNeighbors )
        };
    }

    if ( auto res = triangulatePointCloud( pointCloud, params ) )
        RETURN_NEW( std::move( *res ) );

    return nullptr;
}