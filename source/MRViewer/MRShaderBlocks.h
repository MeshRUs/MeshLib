#pragma once
#include "exports.h"
#include <string>

namespace MR
{

MRVIEWER_API std::string getPickerFragmentShader( bool points, bool cornerMode = true );

MRVIEWER_API std::string getFragmentShaderClippingBlock();

MRVIEWER_API std::string getFragmentShaderPointSizeBlock();

MRVIEWER_API std::string getFragmentShaderOnlyOddBlock( bool sampleMask );

MRVIEWER_API std::string getFragmentShaderHeaderBlock( bool gl4, bool alphaSort );

MRVIEWER_API std::string getFragmentShaderEndBlock( bool alphaSort );

MRVIEWER_API std::string getShaderMainBeginBlock();

}