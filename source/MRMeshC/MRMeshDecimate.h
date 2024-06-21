#pragma once

#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

enum MRDecimateStrategy
{
    MRDecimateStrategyMinimizeError = 0,
    MRDecimateStrategyShortestEdgeFirst
};

typedef struct MRMESHC_CLASS MRDecimateSettings
{
    MRDecimateStrategy strategy;
    float maxError;
    float maxEdgeLen;
    float maxBdShift;
    float maxTriangleAspectRatio;
    float criticalTriAspectRatio;
    float tinyEdgeLength;
    float stabilizer;
    bool optimizeVertexPos;
    int maxDeletedVertices;
    int maxDeletedFaces;
    MRFaceBitSet* region;
    // TODO: notFlippable
    // TODO: edgesToCollapse
    // TODO: touchBdVertices
    // TODO: bdVerts
    float maxAngleChange;
    // TODO: preCollapse
    // TODO: adjustCollapse
    // TODO: onEdgeDel
    // TODO: vertForms
    bool packMesh;
    MRProgressCallback progressCallback;
    int subdivideParts;
    bool decimateBetweenParts;
    // TODO: partFaces
    int minFacesInPart;
} MRDecimateSettings;

MRMESHC_API MRDecimateSettings mrDecimateSettingsDefault();

typedef struct MRMESHC_CLASS MRDecimateResult
{
    int vertsDeleted;
    int facesDeleted;
    float errorIntroduced;
    bool cancelled;
} MRDecimateResult;

MRMESHC_API MRDecimateResult mrDecimateMesh( MRMesh* mesh, const MRDecimateSettings* settings );

typedef struct MRMESHC_CLASS MRResolveMeshDegenSettings
{
    float maxDeviation;
    float tinyEdgeLength;
    float maxAngleChange;
    float criticalAspectRatio;
    float stabilizer;
    MRFaceBitSet* region;
} MRResolveMeshDegenSettings;

MRMESHC_API MRResolveMeshDegenSettings mrResolveMeshDegenSettingsDefault();

MRMESHC_API bool mrResolveMeshDegenerations( MRMesh* mesh, const MRResolveMeshDegenSettings* settings );

typedef struct MRMESHC_CLASS MRRemeshSettings
{
    float targetEdgeLen;
    int maxEdgeSplits;
    float maxAngleChangeAfterFlip;
    float maxBdShift;
    bool useCurvature;
    int finalRelaxIters;
    bool finalRelaxNoShrinkage;
    MRFaceBitSet* region;
    // TODO: notFlippable
    bool packMesh;
    bool projectOnOriginalMesh;
    // TODO: onEdgeSplit
    // TODO: onEdgeDel
    // TODO: preCollapse
    MRProgressCallback progressCallback;
} MRRemeshSettings;

MRMESHC_API MRRemeshSettings mrRemeshSettingsDefault();

MRMESHC_API bool mrRemesh( MRMesh* mesh, const MRRemeshSettings* settings );

MR_EXTERN_C_END
