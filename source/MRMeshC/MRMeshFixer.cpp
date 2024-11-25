#include "MRMeshFixer.h"
#include "MRBitSet.h"
#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMeshFixer.h"
#include "detail/TypeCast.h"

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( MeshPart )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_AUTO_CAST2( std::string, MRString )

MRFaceBitSet* mrFindHoleComplicatingFaces( MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( findHoleComplicatingFaces( mesh ) );
}

MRFaceBitSet* mrFindDegenerateFaces( const MRMeshPart* mp_, float criticalAspectRatio, MRProgressCallback cb_, MRString** errorString )
{
    ARG( mp );
    auto cb = [cb_]( float progress ) -> bool { return cb_( progress ); };
    auto res = cb_ ? findDegenerateFaces( mp, criticalAspectRatio, cb ) : findDegenerateFaces( mp, criticalAspectRatio );
    if ( res )
        RETURN_NEW( *res );

    if ( errorString && !res )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;    
}

MRUndirectedEdgeBitSet* mrFindShortEdges( const MRMeshPart* mp_, float criticalLength, MRProgressCallback cb_, MRString** errorString )
{
    ARG( mp );
    auto cb = [cb_]( float progress ) -> bool { return cb_( progress ); };
    auto res = cb_ ? findShortEdges( mp, criticalLength, cb ) : findShortEdges( mp, criticalLength );
    if ( res )
        RETURN_NEW( *res );

    if ( errorString && !res )
        *errorString = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}

void fixMultipleEdges( MRMesh* mesh_, const MRMultipleEdge* multipleEdges, size_t multipleEdgesNum )
{
    ARG( mesh );
    std::vector<MultipleEdge> multipleEdgesVec( (MultipleEdge*)multipleEdges, ( MultipleEdge* )multipleEdges + multipleEdgesNum );
    fixMultipleEdges( mesh, multipleEdgesVec );
}

void findAndFixMultipleEdges( MRMesh* mesh_ )
{
    ARG( mesh );
    fixMultipleEdges( mesh );
}