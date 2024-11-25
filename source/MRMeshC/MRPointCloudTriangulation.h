#pragma once
#include "MRMeshFwd.h"

MR_EXTERN_C_BEGIN

/**
 * \brief Parameters of point cloud triangulation
 */
typedef struct MRTriangulationParameters
{
    /**
     * \brief The number of nearest neighbor points to use for building of local triangulation
     * \note Too small value can make not optimal triangulation and additional holes
     * Too big value increases difficulty of optimization and decreases performance    
     */
    int numNeighbours;
    /**
     * Radius of neighborhood around each point to consider for building local triangulation.
     * This is an alternative to numNeighbours parameter.
     * Please set to positive value only one of them.
     */
    float radius;
    /**
     * \brief Critical angle of triangles in local triangulation (angle between triangles in fan should be less then this value)    
     */
    float critAngle;

    /// the vertex is considered as boundary if its neighbor ring has angle more than this value
    float boundaryAngle;
    /**
     * \brief Critical length of hole (all holes with length less then this value will be filled)
     * \details If value is subzero it is set automaticly to 0.7*bbox.diagonal()
     */
    float critHoleLength;

    /// automatic increase of the radius if points outside can make triangles from original radius not-Delone
    bool automaticRadiusIncrease;

    /// optional: if provided this cloud will be used for searching of neighbors (so it must have same validPoints)
    const MRPointCloud* searchNeighbors;
} MRTriangulationParameters;

MRMESHC_API MRTriangulationParameters mrTriangulationParametersNew( void );

/**
 * \brief Creates mesh from given point cloud according params
 * Returns empty optional if was interrupted by progress bar
 */
MRMESHC_API MRMesh* mrTriangulatePointCloud( const MRPointCloud* pointCloud, const MRTriangulationParameters* params );

MR_EXTERN_C_END