#pragma once

#include "MRMeshFwd.h"
#include "MRId.h"
#include "MRVector3.h"

MR_EXTERN_C_BEGIN

/// ...
typedef struct MRPointOnFace
{
    MRFaceId face;
    MRVector3f point;
} MRPointOnFace;

MR_EXTERN_C_END
