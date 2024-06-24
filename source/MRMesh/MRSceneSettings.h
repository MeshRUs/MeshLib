#pragma once
#include "MRMeshFwd.h"
#include "MRCNCMachineSettings.h"
#include <array>

namespace MR
{

/// This singleton struct contains default settings for scene objects
/// \ingroup BasicStructuresGroup
class SceneSettings
{
public:
    // Reset all scene settings to default values
    MRMESH_API static void reset();

    enum class BoolType
    {
        /// on deserialization replace object properties with default values from SceneSettings and SceneColors
        UseDefaultScenePropertiesOnDeserialization,
        /// total count
        Count
    };

    enum class FloatType
    {
        FeaturePointsAlpha,
        FeatureLinesAlpha,
        FeatureMeshAlpha,
        FeatureSubPointsAlpha,
        FeatureSubLinesAlpha,
        FeatureSubMeshAlpha,
        // Line width of line features (line, circle, ...).
        FeatureLineWidth,
        // Line width of line subfeatures (axes, base circles, ...).
        FeatureSubLineWidth,
        // Size of the point feature.
        FeaturePointSize,
        // Size of point subfeatures (various centers).
        FeatureSubPointSize,
        // Ambient multiplication coefficient for ambientStrength for selected objects
        AmbientCoefSelectedObj,

        Count,
    };

    MRMESH_API static bool get( BoolType type );
    MRMESH_API static float get( FloatType type );
    MRMESH_API static void set( BoolType type, bool value );
    MRMESH_API static void set( FloatType type, float value );

    /// Mesh faces shading mode
    enum class ShadingMode
    {
        AutoDetect,
        Smooth,
        Flat
    };

    /// Default shading mode for new mesh objects, or imported form files
    /// Tools may consider this setting when creating new meshes
    /// `AutoDetect`: choose depending of file format and mesh shape, fallback to smooth
    MRMESH_API static ShadingMode getDefaultShadingMode();
    MRMESH_API static void setDefaultShadingMode( ShadingMode mode );

    MRMESH_API static const CNCMachineSettings& getCNCMachineSettings();
    MRMESH_API static void setCNCMachineSettings( const CNCMachineSettings& settings );

    // Alpha for mesh features (plane, cylinder, cone...).
    MRMESH_API static std::uint8_t getFeatureMeshAlpha();
    MRMESH_API static void setFeatureMeshAlpha( std::uint8_t alpha );


private:
    MRMESH_API SceneSettings();
    ~SceneSettings() = default;
    SceneSettings& operator=( const SceneSettings& other ) = default;

    static SceneSettings& instance_();

    std::array<bool, size_t( BoolType::Count ) > boolSettings_;
    std::array<float, size_t( FloatType::Count ) > floatSettings_;

    ShadingMode defaultShadingMode_ = ShadingMode::AutoDetect;
    CNCMachineSettings cncMachineSettings_;
};

}
