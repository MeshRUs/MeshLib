#pragma once

#include "MRMesh/MRFeatureObject.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRSceneColors.h"
#include "MRMesh/MRSceneSettings.h"
#include "MRViewer/MRRenderDefaultUiObject.h"
#include "MRViewer/MRRenderDimensions.h"
#include "MRViewer/MRRenderLinesObject.h"
#include "MRViewer/MRRenderLinesObject.h"
#include "MRViewer/MRRenderMeshObject.h"
#include "MRViewer/MRRenderPointsObject.h"
#include "MRViewer/MRRenderWrapObject.h"

namespace MR::RenderFeatures
{

namespace detail
{
    // See `WrappedModelSubobject` below. This class holds optional components for it that depend on the template parameter.
    template <bool IsPrimary, typename BaseObjectType>
    class WrappedModelSubobjectPart : public BaseObjectType {};

    template <bool IsPrimary>
    class WrappedModelSubobjectPart<IsPrimary, ObjectPoints> : public ObjectPoints
    {
        float getPointSize() const override
        {
            const_cast<WrappedModelSubobjectPart &>( *this ).setPointSize( SceneSettings::get( IsPrimary ? SceneSettings::FloatType::FeaturePointSize : SceneSettings::FloatType::FeatureSubPointSize ) );
            return ObjectPoints::getPointSize();
        }
    };

    template <bool IsPrimary>
    class WrappedModelSubobjectPart<IsPrimary, ObjectLines> : public ObjectLines
    {
        float getLineWidth() const override
        {
            const_cast<WrappedModelSubobjectPart &>( *this ).setLineWidth( SceneSettings::get( IsPrimary ? SceneSettings::FloatType::FeatureLineWidth : SceneSettings::FloatType::FeatureSubLineWidth ) );
            return ObjectLines::getLineWidth();
        }
    };

    template <bool IsPrimary>
    class WrappedModelSubobjectPart<IsPrimary, ObjectMesh> : public ObjectMesh, public virtual RenderWrapObject::BasicWrapperTarget
    {
        const ViewportProperty<uint8_t>& getGlobalAlphaForAllViewports() const
        {
            const_cast<WrappedModelSubobjectPart &>( *this ).setGlobalAlpha( (std::uint8_t)std::clamp( int( target_->getGlobalAlpha() * SceneSettings::get( SceneSettings::FloatType::FeatureMeshAlpha ) ), 0, 255 ) );
            return ObjectMesh::getGlobalAlphaForAllViewports();
        }
    };
}

// Wraps a datamodel object to override some of its visual properties.
// This is used for stub datamodel objects that we store inside of renderobjects to provide them with models (aka visualization data: meshes, etc).
// The base template handles `IsPrimary == true`. We have a specialization below for `false`.
template <bool IsPrimary, typename BaseObjectType>
class WrappedModelSubobject : public detail::WrappedModelSubobjectPart<IsPrimary, BaseObjectType>, public virtual RenderWrapObject::BasicWrapperTarget
{
public:
    bool isSelected() const override
    {
        return target_->isSelected();
    }

    const ViewportProperty<Color>& getFrontColorsForAllViewports( bool selected = true ) const override
    {
        return target_->getFrontColorsForAllViewports( selected );
    }

    const ViewportProperty<Color>& getBackColorsForAllViewports() const override
    {
        return target_->getBackColorsForAllViewports();
    }
};

template <typename BaseObjectType>
class WrappedModelSubobject<false, BaseObjectType> : public detail::WrappedModelSubobjectPart<false, BaseObjectType>, public virtual RenderWrapObject::BasicWrapperTarget
{
public:
    bool isSelected() const override
    {
        return target_->isSelected();
    }

    ViewportMask visibilityMask() const override
    {
        if ( auto p = this->parent() )
        {
            if ( auto f = dynamic_cast<const FeatureObject*>( p ) )
                const_cast<WrappedModelSubobject &>( *this ).setVisibilityMask( f->getVisualizePropertyMask( FeatureVisualizePropertyType::Subfeatures ) );
        }

        return this->visibilityMask_;
    }

    const ViewportProperty<Color>& getFrontColorsForAllViewports( bool selected = true ) const override
    {
        return dynamic_cast<const FeatureObject&>( *target_ ).getDecorationsColorForAllViewports( selected );
    }
};

// A common base class for sub-renderobjects that are combined into the proper features.
// `ObjectType` is the underlying datamodel object that stores the mesh, e.g. `ObjectMesh`.
// `RenderObjectType` is the underlying render object, e.g. `RenderMeshObject`.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary, typename ObjectType, typename RenderObjectType>
class RenderFeatureComponent : public RenderWrapObject::Wrapper<WrappedModelSubobject<IsPrimary, ObjectType>, RenderObjectType>
{
    using Base = RenderWrapObject::Wrapper<WrappedModelSubobject<IsPrimary, ObjectType>, RenderObjectType>;
public:
    using Base::Base;

    bool render( const ModelRenderParams& params )
    {
        // Skip rendering the secondary components (aka subfeatures) if they are disabled.
        if constexpr ( !IsPrimary )
        {
            if ( !this->subobject.target_->getVisualizeProperty( FeatureVisualizePropertyType::Subfeatures, params.viewportId ) )
                return false;
        }
        return Base::render( params );
    }
};

// This renderobject draws custom points.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary>
class RenderFeaturePointsComponent : public RenderFeatureComponent<IsPrimary, ObjectPoints, RenderPointsObject>
{
    using Base = RenderFeatureComponent<IsPrimary, ObjectPoints, RenderPointsObject>;
public:
    using Base::Base;
    auto& getPoints() { return Base::subobject; }
};

// This renderobject draws custom lines.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary>
class RenderFeatureLinesComponent : public RenderFeatureComponent<IsPrimary, ObjectLines, RenderLinesObject>
{
    using Base = RenderFeatureComponent<IsPrimary, ObjectLines, RenderLinesObject>;
public:
    using Base::Base;
    auto& getLines() { return Base::subobject; }
};

// This renderobject draws a custom mesh.
// If `IsPrimary` is true, the visual properties are copied from the target datamodel object.
template <bool IsPrimary>
class RenderFeatureMeshComponent : public RenderFeatureComponent<IsPrimary, ObjectMesh, RenderMeshObject>
{
    using Base = RenderFeatureComponent<IsPrimary, ObjectMesh, RenderMeshObject>;
public:
    using Base::Base;
    auto& getMesh() { return Base::subobject; }
};


// This renderobject draws a plane normal for the target object.
class RenderPlaneNormalComponent : public RenderFeatureMeshComponent<false>
{
public:
    MRVIEWER_API RenderPlaneNormalComponent( const VisualObject& object );

    MRVIEWER_API bool render( const ModelRenderParams& params ) override;
    MRVIEWER_API void renderPicker( const ModelBaseRenderParams& params, unsigned geomId ) override;
};


class RenderPointFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeaturePointsComponent<true>>
{
public:
    MRVIEWER_API RenderPointFeatureObject( const VisualObject& object );

    MRVIEWER_API std::string getObjectNameString( const VisualObject& object, ViewportId viewportId ) const override;
};

class RenderLineFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureLinesComponent<true>>
{
public:
    MRVIEWER_API RenderLineFeatureObject( const VisualObject& object );

    MRVIEWER_API std::string getObjectNameString( const VisualObject& object, ViewportId viewportId ) const override;
};

class RenderCircleFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureLinesComponent<true>, RenderFeaturePointsComponent<false>>
{
    const VisualObject* object_ = nullptr;
    RenderDimensions::RadiusTask radiusTask_;
public:
    MRVIEWER_API RenderCircleFeatureObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

class RenderPlaneFeatureObject
    : public RenderObjectCombinator<
        RenderDefaultUiObject,
        // Main mesh.
        RenderFeatureMeshComponent<true>,
        // Subfeatures.
        RenderFeatureLinesComponent<false>, RenderFeaturePointsComponent<false>,
        // Normal mesh.
        RenderPlaneNormalComponent
    >
{
public:
    MRVIEWER_API RenderPlaneFeatureObject( const VisualObject& object );

    MRVIEWER_API std::string getObjectNameString( const VisualObject& object, ViewportId viewportId ) const override;
};

class RenderSphereFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeaturePointsComponent<false>>
{
    const VisualObject* object_ = nullptr;
    RenderDimensions::RadiusTask radiusTask_;
public:
    MRVIEWER_API RenderSphereFeatureObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

class RenderCylinderFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeatureLinesComponent<false>, RenderFeaturePointsComponent<false>>
{
    const VisualObject* object_ = nullptr;
    RenderDimensions::RadiusTask radiusTask_;
    RenderDimensions::LengthTask lengthTask_;
public:
    MRVIEWER_API RenderCylinderFeatureObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

class RenderConeFeatureObject : public RenderObjectCombinator<RenderDefaultUiObject, RenderFeatureMeshComponent<true>, RenderFeatureLinesComponent<false>, RenderFeaturePointsComponent<false>>
{
    const VisualObject* object_ = nullptr;
    RenderDimensions::RadiusTask radiusTask_;
    RenderDimensions::AngleTask angleTask_;
    RenderDimensions::LengthTask lengthTask_;
public:
    MRVIEWER_API RenderConeFeatureObject( const VisualObject& object );
    MRVIEWER_API void renderUi( const UiRenderParams& params ) override;
};

}
