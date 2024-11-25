#pragma once

#include "MRAffineXf3.h"
#include "MRBox.h"
#include "MRBitSet.h"
#include "MRViewportProperty.h"
#include "MRProgressCallback.h"
#include "MRExpected.h"
#include "MRSignal.h"
#include <memory>
#include <vector>
#include <array>
#include <future>
#include <filesystem>

namespace Json
{
class Value;
}

namespace MR
{

/**
 * \defgroup DataModelGroup Data Model
 * \brief This chapter represents documentation about data models
 * \{
 */

/// the main purpose of this class is to avoid copy and move constructor and assignment operator
/// implementation in Object class, which has too many fields for that;
/// since every object stores a pointer on its parent,
/// copying of this object does not copy the children and moving is taken with care
class ObjectChildrenHolder
{
public:
    ObjectChildrenHolder() = default;
    ObjectChildrenHolder( const ObjectChildrenHolder & ) noexcept {}
    ObjectChildrenHolder & operator = ( const ObjectChildrenHolder & ) noexcept { return *this; }
    MRMESH_API ObjectChildrenHolder( ObjectChildrenHolder && ) noexcept;
    MRMESH_API ObjectChildrenHolder & operator = ( ObjectChildrenHolder && ) noexcept;
    MRMESH_API ~ObjectChildrenHolder();

    // returns this Object as shared_ptr
    // finds it among its parent's recognized children
    [[nodiscard]] MRMESH_API std::shared_ptr<Object> getSharedPtr() const;

    /// returns the amount of memory this object occupies on heap,
    /// including the memory of all recognized children
    [[nodiscard]] MRMESH_API size_t heapBytes() const;

protected:
    ObjectChildrenHolder * parent_ = nullptr;
    std::vector< std::shared_ptr< Object > > children_; /// recognized ones
    std::vector< std::weak_ptr< Object > > bastards_; /// unrecognized children to hide from the pubic
};

/// named object in the data model
class MRMESH_CLASS Object : public ObjectChildrenHolder
{
public:
    Object() = default;
    Object( Object && ) noexcept = default;
    Object & operator = ( Object && ) noexcept = default;
    virtual ~Object() = default;

    // return name of subtype for serialization purposes
    constexpr static const char* TypeName() noexcept { return "Object"; }
    virtual const char* typeName() const { return TypeName(); }

    template <typename T>
    T * asType() { return dynamic_cast<T*>( this ); }
    template <typename T>
    const T * asType() const { return dynamic_cast<const T*>( this ); }

    const std::string & name() const { return name_; }
    virtual void setName( std::string name ) { name_ = std::move( name ); }

    /// finds a direct child by name
    MRMESH_API std::shared_ptr<const Object> find( const std::string_view & name ) const;
    std::shared_ptr<Object> find( const std::string_view & name ) { return std::const_pointer_cast<Object>( const_cast<const Object*>( this )->find( name ) ); }

    /// finds a direct child by type
    template <typename T>
    std::shared_ptr<const T> find() const;
    template <typename T>
    std::shared_ptr<T> find() { return std::const_pointer_cast<T>( const_cast<const Object*>( this )->find<T>() ); }

    /// finds a direct child by name and type
    template <typename T>
    std::shared_ptr<const T> find( const std::string_view & name ) const;
    template <typename T>
    std::shared_ptr<T> find( const std::string_view & name ) { return std::const_pointer_cast<T>( const_cast<const Object*>( this )->find<T>( name ) ); }

    /// this space to parent space transformation (to world space if no parent) for default or given viewport
    /// \param isDef receives true if the object has default transformation in this viewport (same as xf() returns)
    const AffineXf3f & xf( ViewportId id = {}, bool * isDef = nullptr ) const { return xf_.get( id, isDef ); }
    MRMESH_API virtual void setXf( const AffineXf3f& xf, ViewportId id = {} );
    /// forgets specific transform in given viewport (or forgets all specific transforms for {} input)
    MRMESH_API virtual void resetXf( ViewportId id = {} );

    /// returns xfs for all viewports, combined into a single object
    const ViewportProperty<AffineXf3f> & xfsForAllViewports() const { return xf_; }
    /// modifies xfs for all viewports at once
    virtual void setXfsForAllViewports( ViewportProperty<AffineXf3f> xf ) { xf_ = std::move( xf ); }

    /// this space to world space transformation for default or specific viewport
    /// \param isDef receives true if the object has default transformation in this viewport (same as worldXf() returns)
    MRMESH_API AffineXf3f worldXf( ViewportId id = {}, bool * isDef = nullptr ) const;
    MRMESH_API void setWorldXf( const AffineXf3f& xf, ViewportId id = {} );

    /// scale object size (all point positions)
    MRMESH_API virtual void applyScale( float scaleFactor );

    /// returns all viewports where this object is visible together with all its parents
    MRMESH_API ViewportMask globalVisibilityMask() const;
    /// returns true if this object is visible together with all its parents in any of given viewports
    bool globalVisibility( ViewportMask viewportMask = ViewportMask::any() ) const { return !( globalVisibilityMask() & viewportMask ).empty(); }
    /// if true sets all predecessors visible, otherwise sets this object invisible
    MRMESH_API void setGlobalVisibility( bool on, ViewportMask viewportMask = ViewportMask::any() );

    /// object properties lock for UI
    bool isLocked() const { return locked_; }
    virtual void setLocked( bool on ) { locked_ = on; }

    /// If true, the scene tree GUI doesn't allow you to drag'n'drop this object into a different parent.
    /// Defaults to false.
    [[nodiscard]] bool isParentLocked() const { return parentLocked_; }
    virtual void setParentLocked( bool lock ) { parentLocked_ = lock; }

    /// returns parent object in the tree
    const Object * parent() const { return static_cast<const Object *>( parent_ ); }
    Object * parent() { return static_cast<Object *>( parent_ ); }

    /// return true if given object is ancestor of this one, false otherwise
    MRMESH_API bool isAncestor( const Object* ancestor ) const;

    /// Find a common ancestor between this object and the other one.
    /// Returns null on failure (which is impossible if both are children of the scene root).
    /// Will return `this` if `other` matches `this`.
    [[nodiscard]] MRMESH_API Object* findCommonAncestor( Object& other );
    [[nodiscard]] const Object* findCommonAncestor( const Object& other ) const
    {
        return const_cast<Object &>( *this ).findCommonAncestor( const_cast<Object &>( other ) );
    }

    /// removes this from its parent children list
    /// returns false if it was already orphan
    MRMESH_API virtual bool detachFromParent();
    /// an object can hold other sub-objects
    const std::vector<std::shared_ptr<Object>>& children() { return children_; }
    const std::vector<std::shared_ptr<const Object>>& children() const { return reinterpret_cast<const std::vector< std::shared_ptr< const Object > > &>( children_ ); }
    /// adds given object at the end of children (recognized or not);
    /// returns false if it was already child of this, of if given pointer is empty;
    /// child object will always report this as parent after the call;
    /// \param recognizedChild if set to false then child object will be excluded from children() and it will be stored by weak_ptr
    MRMESH_API virtual bool addChild( std::shared_ptr<Object> child, bool recognizedChild = true );
    /// adds given object in the recognized children before existingChild;
    /// if newChild was already among this children then moves it just before existingChild keeping the order of other children intact;
    /// returns false if newChild is nullptr, or existingChild is not a child of this
    MRMESH_API virtual bool addChildBefore( std::shared_ptr<Object> newChild, const std::shared_ptr<Object> & existingChild );
    /// returns false if it was not child of this
    bool removeChild( const std::shared_ptr<Object>& child ) { return removeChild( child.get() ); }
    MRMESH_API virtual bool removeChild( Object* child );
    /// detaches all recognized children from this, keeping all unrecognized ones
    MRMESH_API virtual void removeAllChildren();
    /// sort recognized children by name
    MRMESH_API void sortChildren();

    /// selects the object, returns true if value changed, otherwise returns false
    MRMESH_API virtual bool select( bool on );
    virtual bool isSelected() const { return selected_; }

    /// ancillary object is an object hidden (in scene menu) from a regular user
    /// such objects cannot be selected, and if it has been selected, it is unselected when turn ancillary
    MRMESH_API virtual void setAncillary( bool ancillary );
    bool isAncillary() const { return ancillary_; }

    /// sets the object visible in the viewports specified by the mask (by default in all viewports)
    MRMESH_API void setVisible( bool on, ViewportMask viewportMask = ViewportMask::all() );
    /// checks whether the object is visible in any of the viewports specified by the mask (by default in any viewport)
    bool isVisible( ViewportMask viewportMask = ViewportMask::any() ) const { return !( visibilityMask() & viewportMask ).empty(); }
    /// specifies object visibility as bitmask of viewports
    MRMESH_API virtual void setVisibilityMask( ViewportMask viewportMask );
    /// gets object visibility as bitmask of viewports
    virtual ViewportMask visibilityMask() const { return visibilityMask_; }

    /// this method virtual because others data model types could have dirty flags or something
    virtual bool getRedrawFlag( ViewportMask ) const { return needRedraw_; }
    void resetRedrawFlag() const { needRedraw_ = false; }

    /// clones all tree of this object (except ancillary and unrecognized children)
    MRMESH_API std::shared_ptr<Object> cloneTree() const;
    /// clones current object only, without parent and/or children
    MRMESH_API virtual std::shared_ptr<Object> clone() const;
    /// clones all tree of this object (except ancillary and unrecognied children)
    /// clones only pointers to mesh, points or voxels
    MRMESH_API std::shared_ptr<Object> shallowCloneTree() const;
    /// clones current object only, without parent and/or children
    /// clones only pointers to mesh, points or voxels
    MRMESH_API virtual std::shared_ptr<Object> shallowClone() const;

    /// return several info lines that can better describe object in the UI
    MRMESH_API virtual std::vector<std::string> getInfoLines() const;
    /// return human readable name of subclass
    virtual std::string getClassName() const { return "Object"; }

    /// creates futures that save this object subtree:
    ///   models in the folder by given path and
    ///   fields in given JSON
    /// \param childId is its ordinal number within the parent
    // This would be automatically skipped in the bindings anyway because of the `Json::Value` parameter.
    // But skipping it here prevents the vector-of-futures type from being registered, which is helpful.
    // TODO: figure out how to automate this (add a flag to the parser to outright reject functions based on their parameter and return types).
    MRMESH_API MR_BIND_IGNORE Expected<std::vector<std::future<Expected<void>>>> serializeRecursive( const std::filesystem::path& path, Json::Value& root, int childId ) const;

    /// loads subtree into this Object
    ///   models from the folder by given path and
    ///   fields from given JSON
    MRMESH_API Expected<void> deserializeRecursive( const std::filesystem::path& path, const Json::Value& root,
        ProgressCallback progressCb = {}, int* objCounter = nullptr );

    /// swaps this object with other
    /// note: do not swap object signals, so listeners will get notifications from swapped object
    /// requires implementation of `swapBase_` and `swapSignals_` (if type has signals)
    MRMESH_API void swap( Object& other );

    /// returns bounding box of this object in world coordinates for default or specific viewport
    virtual Box3f getWorldBox( ViewportId = {} ) const { return {}; } ///empty box
    /// returns bounding box of this object and all children visible in given (or default) viewport in world coordinates
    MRMESH_API Box3f getWorldTreeBox( ViewportId = {} ) const;

    /// does the object have any visual representation (visible points, triangles, edges, etc.), no considering child objects
    [[nodiscard]] virtual bool hasVisualRepresentation() const { return false; }

    /// does the object have any model available (but possibly empty),
    /// e.g. ObjectMesh has valid mesh() or ObjectPoints has valid pointCloud()
    [[nodiscard]] virtual bool hasModel() const { return false; }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] MRMESH_API virtual size_t heapBytes() const;

    /// signal about xf changing
    /// triggered in setXf and setWorldXf, it is called for children too
    /// triggered in addChild and addChildBefore, it is called only for children object
    using XfChangedSignal = Signal<void() >;
    XfChangedSignal worldXfChangedSignal;
protected:
    struct ProtectedStruct{ explicit ProtectedStruct() = default; };
public:
    /// \note this ctor is public only for std::make_shared used inside clone()
    Object( ProtectedStruct, const Object& obj ) : Object( obj ) {}

protected:
    /// user should not be able to call copy implicitly, use clone() function instead
    Object( const Object& obj ) = default;

    /// swaps whole object (signals too)
    MRMESH_API virtual void swapBase_( Object& other );
    /// swaps signals, used in `swap` function to return back signals after `swapBase_`
    /// \note pls call Parent::swapSignals_ first when overriding this function
    MRMESH_API virtual void swapSignals_( Object& other );

    /// Creates future to save object model (e.g. mesh) in given file
    /// path is full filename without extension
    MRMESH_API virtual Expected<std::future<Expected<void>>> serializeModel_( const std::filesystem::path& path ) const;

    /// Write parameters to given Json::Value,
    /// \note if you override this method, please call Base::serializeFields_(root) in the beginning
    MRMESH_API virtual void serializeFields_( Json::Value& root ) const;

    /// Reads model from file
    MRMESH_API virtual Expected<void> deserializeModel_( const std::filesystem::path& path, ProgressCallback progressCb = {} );

    /// Reads parameters from json value
    /// \note if you override this method, please call Base::deserializeFields_(root) in the beginning
    MRMESH_API virtual void deserializeFields_( const Json::Value& root );

    std::string name_;
    ViewportProperty<AffineXf3f> xf_;
    ViewportMask visibilityMask_ = ViewportMask::all(); // Prefer to not read directly. Use the getter, as it can be overridden.
    bool locked_ = false;
    bool parentLocked_ = false;
    bool selected_{ false };
    bool ancillary_{ false };
    mutable bool needRedraw_{false};

    MRMESH_API virtual void propagateWorldXfChangedSignal_();
};

template <typename T>
std::shared_ptr<const T> Object::find() const
{
    for ( const auto & child : children_ )
        if ( auto res = std::dynamic_pointer_cast<T>( child ) )
            return res;
    return {}; // not found
}

template <typename T>
std::shared_ptr<const T> Object::find( const std::string_view & name ) const
{
    for ( const auto & child : children_ )
        if ( child->name() == name )
            if ( auto res = std::dynamic_pointer_cast<T>( child ) )
                return res;
    return {}; // not found
}

/// \}

} ///namespace MR
