#pragma once
#include "MRMesh/MRMeshFwd.h"
#ifndef MRMESH_NO_CPR
#include "MRViewerFwd.h"
#include "MRMesh/MRExpected.h"
#include "MRPch/MRJson.h"
#include <unordered_map>
#include <string>
#include <functional>

namespace MR
{
// returns json value of text or error if response failed
MRVIEWER_API Expected<Json::Value, std::string> parseResponse( const Json::Value& response );

// this class is needed to unify cpp and wasm requests
class MRVIEWER_CLASS WebRequest
{
public:
    WebRequest() = default;
    explicit WebRequest( std::string url );

    enum class Method 
    {
        Get,
        Post,
        Patch,
        Put,
        Delete,
    };

    // clears all request data
    MRVIEWER_API void clear();

    MRVIEWER_API void setMethod( Method method );

    // sets timeout in milliseconds
    MRVIEWER_API void setTimeout( int timeoutMs );

    // sets parameters
    MRVIEWER_API void setParameters( std::unordered_map<std::string, std::string> parameters );

    MRVIEWER_API void setHeaders( std::unordered_map<std::string, std::string> headers );

    MRVIEWER_API void setInputPath( std::string inputPath );

    // sets payload in multipart format
    struct FormData
    {
        std::string path;
        std::string contentType;
        std::string name;
        std::string fileName;
    };
    MRVIEWER_API void setFormData( std::vector<FormData> formData );

    MRVIEWER_API void setBody( std::string body );

    // prefer to save the response to file
    MRVIEWER_API void setOutputPath( std::string outputPath );

    MRVIEWER_API void setAsync( bool async );

    MRVIEWER_API void setLogName( std::string logName );

    using ResponseCallback = std::function<void( const Json::Value& response )>;

    /// sends request, calling callback on answer, 
    /// if async then callback is called in next frame after getting response
    /// NOTE: downloading a binary file in synchronous mode is forbidden by JavaScript
    /// \param logName name for logging
    MRVIEWER_API void send( std::string url, const std::string & logName, ResponseCallback callback, bool async = true );

    MRVIEWER_API void send( ResponseCallback callback );

private:
    Method method_{ Method::Get };
    std::string url_;
    std::string logName_;
    bool async_{ true };
    int timeout_{ 10000 };
    std::unordered_map<std::string, std::string> params_;
    std::unordered_map<std::string, std::string> headers_;
    std::string inputPath_;
    std::vector<FormData> formData_;
    std::string body_;
    std::string outputPath_;
};

}
#endif
