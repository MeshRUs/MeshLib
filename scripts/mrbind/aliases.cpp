#include MRBIND_HEADER

static const char MRBIND_UNIQUE_VAR = []
{
    #define MR_ALIAS(alias, target) MRBind::pb11::RegisterCustomAlias(#alias, #target)

    MR_ALIAS( BooleanResMapObj,                     BooleanResultMapper.MapObject                       );
    MR_ALIAS( buildUnitedLocalTriangulations,       TriangulationHelpers.buildUnitedLocalTriangulations );
    MR_ALIAS( copyMesh,                             Mesh                                                );
    MR_ALIAS( FaceMap.vec,                          FaceMap.vec_                                        );
    MR_ALIAS( FaceNormals.vec,                      FaceNormals.vec_                                    );
    MR_ALIAS( findUndercuts,                        FixUndercuts.findUndercuts                          );
    MR_ALIAS( FixSelfIntersectionMethod,            SelfIntersections.Settings.Method                   );
    MR_ALIAS( FixSelfIntersectionSettings,          SelfIntersections.Settings                          );
    MR_ALIAS( fixUndercuts,                         FixUndercuts.fixUndercuts                           );
    MR_ALIAS( GeneralOffsetParametersMode,          GeneralOffsetParameters.Mode                        );
    MR_ALIAS( getAllComponents,                     MeshComponents.getAllComponents                     );
    MR_ALIAS( getAllComponentsVerts,                MeshComponents.getAllComponentsVerts                );
    MR_ALIAS( ICP.getLastICPInfo,                   ICP.getStatusInfo                                   );
    MR_ALIAS( LaplacianEdgeWeightsParam,            EdgeWeights                                         );
    MR_ALIAS( loadLines,                            LinesLoad.fromAnySupportedFormat                    );
    MR_ALIAS( loadMesh,                             MeshLoad.fromAnySupportedFormat                     );
    MR_ALIAS( loadPoints,                           PointsLoad.fromAnySupportedFormat                   );
    MR_ALIAS( loadVoxels,                           VoxelsLoad.fromAnySupportedFormat                   );
    MR_ALIAS( loadVoxelsGav,                        VoxelsLoad.fromGav                                  );
    MR_ALIAS( loadVoxelsRaw,                        VoxelsLoad.fromRaw                                  );
    MR_ALIAS( localFindSelfIntersections,           SelfIntersections.getFaces                          );
    MR_ALIAS( localFixSelfIntersections,            SelfIntersections.fix                               );
    MR_ALIAS( MeshBuilderSettings,                  MeshBuilder.BuildSettings                           );
    MR_ALIAS( MeshToVolumeParamsType,               MeshToVolumeParams.Type                             );
    MR_ALIAS( ObjectDistanceMap.extractDistanceMap, ObjectDistanceMap.getDistanceMap                    );
    MR_ALIAS( ObjectLines.extractLines,             ObjectLines.polyline                                );
    MR_ALIAS( ObjectMesh.extractMesh,               ObjectMesh.mesh                                     );
    MR_ALIAS( ObjectPoints.extractPoints,           ObjectPoints.pointCloud                             );
    MR_ALIAS( objectSave,                           ObjectSave.toAnySupportedFormat                     );
    MR_ALIAS( ObjectVoxels.extractVoxels,           ObjectVoxels.vdbVolume                              );
    MR_ALIAS( saveAllSlicesToImage,                 VoxelsSave.saveAllSlicesToImage                     );
    MR_ALIAS( saveLines,                            LinesSave.toAnySupportedFormat                      );
    MR_ALIAS( saveMesh,                             MeshSave.toAnySupportedFormat                       );
    MR_ALIAS( savePoints,                           PointsSave.toAnySupportedFormat                     );
    MR_ALIAS( saveSliceToImage,                     VoxelsSave.saveSliceToImage                         );
    MR_ALIAS( saveVoxels,                           VoxelsSave.toAnySupportedFormat                     );
    MR_ALIAS( saveVoxelsGav,                        VoxelsSave.toGav                                    );
    MR_ALIAS( saveVoxelsRaw,                        VoxelsSave.toRawAutoname                            );
    MR_ALIAS( TextAlignParams,                      TextMeshAlignParams                                 );
    MR_ALIAS( topologyFromTriangles,                MeshBuilder.fromTriangles                           );
    MR_ALIAS( triangulateContours,                  PlanarTriangulation.triangulateContours             );
    MR_ALIAS( Triangulation.vec,                    Triangulation.vec_                                  );
    MR_ALIAS( TriangulationHelpersSettings,         TriangulationHelpers.Settings                       );
    MR_ALIAS( uniteCloseVertices,                   MeshBuilder.uniteCloseVertices                      );
    MR_ALIAS( vectorConstMeshPtr,                   std_vector_const_Mesh                               );
    MR_ALIAS( vectorEdges,                          EdgeLoop                                            );
    MR_ALIAS( VertCoords.vec,                       VertCoords.vec_                                     );
    MR_ALIAS( VertScalars.vec,                      VertScalars.vec_                                    );
    MR_ALIAS( VoxelsSaveSavingSettings,             VoxelsSave.SavingSettings                           );

    return char{};
}();
