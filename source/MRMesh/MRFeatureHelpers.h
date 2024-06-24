#pragma once

#include "MRMeshFwd.h"
#include "MRFeatureObject.h"

#include <unordered_map>
#include <unordered_set>

namespace MR
{

//! Which object type we're adding.
//! Update `ObjKindTraits` if you change this enum.
enum class FeaturesObjectKind
{
    Point,
    Line,
    Plane,
    Circle,
    Sphere,
    Cylinder,
    Cone,
    _count,
};

//! Various information about different types of objects.
template <FeaturesObjectKind X> struct ObjKindTraits
{
};
template <> struct ObjKindTraits<FeaturesObjectKind::Point>
{
    using type = PointObject;
    static constexpr std::string_view name = "Point";
};
template <> struct ObjKindTraits<FeaturesObjectKind::Line>
{
    using type = LineObject;
    static constexpr std::string_view name = "Line";
};
template <> struct ObjKindTraits<FeaturesObjectKind::Plane>
{
    using type = PlaneObject;
    static constexpr std::string_view name = "Plane";
};
template <> struct ObjKindTraits<FeaturesObjectKind::Circle>
{
    using type = CircleObject;
    static constexpr std::string_view name = "Circle";
};
template <> struct ObjKindTraits<FeaturesObjectKind::Sphere>
{
    using type = SphereObject;
    static constexpr std::string_view name = "Sphere";
};
template <> struct ObjKindTraits<FeaturesObjectKind::Cylinder>
{
    using type = CylinderObject;
    static constexpr std::string_view name = "Cylinder";
};
template <> struct ObjKindTraits<FeaturesObjectKind::Cone>
{
    using type = ConeObject;
    static constexpr std::string_view name = "Cone";
};

//! Calls `func`, which is `( auto kind ) -> bool`, for each known object kind. If it returns true, stops immediately and also returns true.
template <typename F>
bool forEachObjectKind( F&& func )
{
    return[&]<int ...I>( std::integer_sequence<int, I...> )
    {
        return ( func( std::integral_constant<FeaturesObjectKind, FeaturesObjectKind( I )>{} ) || ... );
    }( std::make_integer_sequence<int, int( FeaturesObjectKind::_count )>{} );
}

//! Allocates an object of type `kind`, passing `params...` to its constructor.
template <typename ...P>
[[nodiscard]] std::shared_ptr<VisualObject> makeObjectFromEnum( FeaturesObjectKind kind, P&&... params )
{
    std::shared_ptr<VisualObject> ret;
    [[maybe_unused]] bool ok = forEachObjectKind( [&] ( auto thisKind )
    {
        if ( thisKind.value == kind )
        {
            ret = std::make_shared<typename ObjKindTraits<thisKind.value>::type>( std::forward<P>( params )... );
            return true;
        }
        return false;
    } );
    assert( ok && "This object type isn't added to `ObjKindTraits`." );
    return ret;
}

//! Allocates an object of type `kind`, passing `params...` to its constructor.
template <typename ...P>
[[nodiscard]] std::shared_ptr<VisualObject> makeObjectFromClassName( std::string className, P&&... params )
{
    std::shared_ptr<VisualObject> ret;
    [[maybe_unused]] bool ok = forEachObjectKind( [&] ( auto thisKind )
    {
        if ( ObjKindTraits<thisKind.value>::name == className )
        {
            ret = std::make_shared<typename ObjKindTraits<thisKind.value>::type>( std::forward<P>( params )... );
            return true;
        }
        return false;
    } );
    assert( ok && "This object type isn't added to `ObjKindTraits`." );
    return ret;
}

template <typename T>
concept HasGetNormalMethod = requires( T t )
{
    {
        t.getNormal()
    } -> std::convertible_to<Vector3f>;
};

template <typename T>
concept HasGetDirectionMethod = requires( T t )
{
    {
        t.getDirection()
    } -> std::convertible_to<Vector3f>;
};

// Using forEachObjectKind the template collects a list of features for which the method ...->getNormal() is available
MRMESH_API std::optional<Vector3f> getFeatureNormal( FeatureObject* feature );

// Using forEachObjectKind the template collects a list of features for which the method ...->getDirection() is available
MRMESH_API std::optional<Vector3f> getFeatureDirection( FeatureObject* feature );

// Try to getNormal from specific feature using forEachObjectKind template. Returns nullopt is ...->getNormal() is not available for given feature type.
MRMESH_API std::unordered_set<std::string> getFeaturesTypeWithNormals();

// Try to getDirection from specific feature using forEachObjectKind template. Returns nullopt is ...->getDirection() is not available for given feature type.
MRMESH_API std::unordered_set<std::string> getFeaturesTypeWithDirections();

} // namespace MR
