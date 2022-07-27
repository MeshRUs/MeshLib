#include "MRProgressReadWrite.h"

namespace MR
{

bool writeWithProgress( std::ostream& out, const char* data, size_t dataSize, ProgressCallback callback /*= {}*/, size_t blockSize /*= ( size_t( 1 ) << 16 )*/ )
{
    if ( !callback )
    {
        out.write( data, dataSize );
        return true;
    }

    int blockIndex = 0;
    for ( size_t max = dataSize / blockSize; blockIndex < max; ++blockIndex )
    {
        out.write( data + blockIndex * blockSize, blockSize );
        if ( callback && !callback( float( blockIndex * blockSize ) / dataSize ) )
            return false;
    }
    const size_t remnant = dataSize - blockIndex * blockSize;
    if ( remnant )
        out.write( data + blockIndex * blockSize, remnant );
    if ( callback && !callback( float( blockIndex * blockSize + remnant ) / dataSize ) )
        return false;

    return true;
}

}
