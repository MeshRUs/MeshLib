#include "MRFeatureHelpers.h"
#include "MRFeatureObjectImpls.h"

namespace MR
{

std::unordered_set<std::string> getFeaturesTypeWithNormals()
{
    std::unordered_set<std::string> validTypes;
    [[maybe_unused]] bool ok = forEachObjectKind( [&] ( auto thisKind )
    {
        auto ret = std::make_shared<typename ObjKindTraits<thisKind.value>::type>();
        if constexpr ( HasGetNormalMethod<typename ObjKindTraits<thisKind.value>::type> )
        {
            validTypes.insert( ret->TypeName() );
        }
        return false;
    } );
    return validTypes;
}

std::unordered_set<std::string> getFeaturesTypeWithDirections()
{
    std::unordered_set<std::string> validTypes;
    [[maybe_unused]] bool ok = forEachObjectKind( [&] ( auto thisKind )
    {
        auto ret = std::make_shared<typename ObjKindTraits<thisKind.value>::type>();
        if constexpr ( HasGetDirectionMethod<typename ObjKindTraits<thisKind.value>::type> )
        {
            validTypes.insert( ret->TypeName() );
        }
        return false;
    } );
    return validTypes;
}

std::optional<Vector3f>  getFeatureNormal( FeatureObject* feature )
{
    std::optional <Vector3f> normal;
    [[maybe_unused]] bool ok = forEachObjectKind( [&] ( auto thisKind )
    {
        if constexpr ( HasGetNormalMethod<typename ObjKindTraits<thisKind.value>::type> )
        {
            auto obj = dynamic_cast< typename ObjKindTraits<thisKind.value>::type* >( feature );
            if ( obj )
            {
                normal = obj->getNormal();
                return true;
            }
        }
        return false;
    } );
    return normal;
}

std::optional <Vector3f>  getFeatureDirection( FeatureObject* feature )
{
    std::optional <Vector3f> direction;
    [[maybe_unused]] bool ok = forEachObjectKind( [&] ( auto thisKind )
    {
        if constexpr ( HasGetDirectionMethod<typename ObjKindTraits<thisKind.value>::type> )
        {
            auto obj = dynamic_cast< typename ObjKindTraits<thisKind.value>::type* >( feature );
            if ( obj )
            {
                direction = obj->getDirection();
                return true;
            }
        }
        return false;
    } );
    return direction;
}

} // namespace MR
