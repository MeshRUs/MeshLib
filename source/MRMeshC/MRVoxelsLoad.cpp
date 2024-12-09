#include "MRVoxelsLoad.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRVoxels/MRVoxelsLoad.h"

using namespace MR;
REGISTER_AUTO_CAST2( std::string, MRString )
REGISTER_AUTO_CAST( FloatGrid )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST( Vector3i )
MR_VECTOR_LIKE_IMPL( VdbVolumes, VdbVolume )

MRVdbVolumes* mrVoxelsLoadFromAnySupportedFormat( const char* file, MRProgressCallback cb_, MRString** errorStr )
{
    auto cb = [cb_] ( float progress ) -> bool { return cb_( progress ); };
    auto res = cb_ ? VoxelsLoad::fromAnySupportedFormat( file, cb ) : VoxelsLoad::fromAnySupportedFormat( file );

    if ( res )
    {
        std::vector<MRVdbVolume> volumes( res->size() );
        
        for ( size_t i = 0; i < res->size(); ++i )
        {
            volumes[i].data = auto_cast( new_from( std::move( ( *res )[i].data ) ) );
            volumes[i].dims = auto_cast( ( *res )[i].dims );
            volumes[i].voxelSize = auto_cast( ( *res )[i].voxelSize );
            volumes[i].min = ( *res )[i].min;
            volumes[i].max = ( *res )[i].max;
        }
        
        return (MRVdbVolumes*)( NEW_VECTOR( std::move( volumes ) ) );
    }

    if ( errorStr && !res )
        *errorStr = auto_cast( new_from( std::move( res.error() ) ) );

    return nullptr;
}