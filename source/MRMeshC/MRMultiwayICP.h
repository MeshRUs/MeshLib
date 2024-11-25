#pragma once

#include "MRMeshFwd.h"
#include "MRGridSampling.h"
#include "MRICP.h"
#include "MRVector.h"

MR_EXTERN_C_BEGIN

typedef enum MRMultiwayICPSamplingParametersCascadeMode
{
    /// separates objects on groups based on their index in ICPObjects (good if all objects about the size of all objects together)
    MRMultiwayICPSamplingParametersCascadeModeSequential = 0,
    /// builds AABB tree based on each object bounding box and separates subtrees (good if each object much smaller then all objects together)
    MRMultiwayICPSamplingParametersCascadeModeABBTreeBased
} MRMultiwayICPSamplingParametersCascadeMode;

/// Parameters that are used for sampling of the MultiwayICP objects
typedef struct MRMultiwayICPSamplingParameters
{
    /// sampling size of each object
    float samplingVoxelSize;
    /// size of maximum icp group to work with
    /// if number of objects exceeds this value, icp is applied in cascade mode
    int maxGroupSize;
    MRMultiwayICPSamplingParametersCascadeMode cascadeMode;
    /// callback for progress reports
    MRProgressCallback cb;
} MRMultiwayICPSamplingParameters;

/// initializes a default instance
MRMESHC_API MRMultiwayICPSamplingParameters mrMultiwayIcpSamplingParametersNew( void );

/// This class allows you to register many objects having similar parts
/// and known initial approximations of orientations/locations using
/// Iterative Closest Points (ICP) point-to-point or point-to-plane algorithms
typedef struct MRMultiwayICP MRMultiwayICP;

MRMESHC_API MRMultiwayICP* mrMultiwayICPNew( const MRMeshOrPointsXf** objects, size_t objectsNum, const MRMultiwayICPSamplingParameters* samplingParams );

/// runs ICP algorithm given input objects, transformations, and parameters;
/// \return adjusted transformations of all objects to reach registered state
MRMESHC_API MRVectorAffineXf3f* mrMultiwayICPCalculateTransformations( MRMultiwayICP* mwicp, MRProgressCallback cb );

/// select pairs with origin samples on all objects
MRMESHC_API bool mrMultiwayICPResamplePoints( MRMultiwayICP* mwicp, const MRMultiwayICPSamplingParameters* samplingParams );

/// in each pair updates the target data and performs basic filtering (activation)
/// in cascade mode only useful for stats update
MRMESHC_API bool mrMultiwayICPUpdateAllPointPairs( MRMultiwayICP* mwicp, MRProgressCallback cb );

/// tune algorithm params before run calculateTransformations()
MRMESHC_API void mrMultiwayICPSetParams( MRMultiwayICP* mwicp, const MRICPProperties* prop );

/// computes root-mean-square deviation between points
/// or the standard deviation from given value if present
MRMESHC_API float mrMultiWayICPGetMeanSqDistToPoint( const MRMultiwayICP* mwicp, const double* value );

/// computes root-mean-square deviation from points to target planes
/// or the standard deviation from given value if present
MRMESHC_API float mrMultiWayICPGetMeanSqDistToPlane( const MRMultiwayICP* mwicp, const double* value );

/// computes the number of samples able to form pairs
MRMESHC_API size_t mrMultiWayICPGetNumSamples( const MRMultiwayICP* mwicp );

/// computes the number of active point pairs
MRMESHC_API size_t mrMultiWayICPGetNumActivePairs( const MRMultiwayICP* mwicp );

/// deallocates a MultiwayICP object
MRMESHC_API void mrMultiwayICPFree( MRMultiwayICP* mwicp );

MR_EXTERN_C_END
