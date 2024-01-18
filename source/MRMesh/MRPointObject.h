#pragma once
#include "MRMeshFwd.h"
#include "MRObjectPointsHolder.h"
#include "MRFeatureObjectSharedProperties.h"

namespace MR
{

/// Object to show point feature
/// \ingroup FeaturesGroup
class MRMESH_CLASS PointObject : public ObjectPointsHolder, public FeatureObject
{
public:
    /// Creates simple point object with zero position
    MRMESH_API PointObject();
    /// Finds best point to approx given points
    MRMESH_API PointObject( const std::vector<Vector3f>& pointsToApprox );

    PointObject( PointObject&& ) noexcept = default;
    PointObject& operator = ( PointObject&& ) noexcept = default;

    constexpr static const char* TypeName() noexcept { return "PointObject"; }
    virtual const char* typeName() const override { return TypeName(); }

    /// \note this ctor is public only for std::make_shared used inside clone()
    PointObject( ProtectedStruct, const PointObject& obj ) : PointObject( obj )
    {}

    virtual std::string getClassName() const override { return "Point"; }

    MRMESH_API virtual std::shared_ptr<Object> clone() const override;
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const override;

    /// calculates point from xf
    MRMESH_API Vector3f getPoint() const;
    /// updates xf to fit given point
    MRMESH_API void setPoint( const Vector3f& point );

    MRMESH_API virtual  std::vector<FeatureObjectSharedProperty> getAllSharedProperties( void ) override;
protected:
    PointObject( const PointObject& other ) = default;

    /// swaps this object with other
    MRMESH_API virtual void swapBase_( Object& other ) override;

    MRMESH_API virtual void serializeFields_( Json::Value& root ) const override;

    virtual Expected<std::future<VoidOrErrStr>> serializeModel_( const std::filesystem::path& ) const override
        { return {}; }

    virtual VoidOrErrStr deserializeModel_( const std::filesystem::path&, ProgressCallback ) override
        { return {}; }

private:
    void constructPointCloud_();
};

}
