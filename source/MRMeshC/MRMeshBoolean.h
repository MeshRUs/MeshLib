#pragma once

#include "MRMeshFwd.h"
#include "MRAffineXf.h"
#include "MRBooleanOperation.h"

MR_EXTERN_C_BEGIN

typedef struct MRMESHC_CLASS MRBooleanParameters
{
    MRAffineXf3f* rigidB2A;
    bool mergeAllNonIntersectingComponents;
    MRProgressCallback cb;
} MRBooleanParameters;

typedef struct MRMESHC_CLASS MRBooleanResult
{
    MRMesh* mesh;
    MRString* errorString;
} MRBooleanResult;

MRMESHC_API MRBooleanResult mrBoolean( const MRMesh* meshA, const MRMesh* meshB, MRBooleanOperation operation, const MRBooleanParameters* params );

MR_EXTERN_C_END
