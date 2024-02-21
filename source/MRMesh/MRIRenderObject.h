#pragma once
#include "MRMesh/MRFlagOperators.h"
#include "MRMeshFwd.h"
#include "MRViewportId.h"
#include "MRVector2.h"
#include "MRVector4.h"
#include "MRAffineXf3.h"
#include <functional>
#include <typeindex>
#include <memory>

namespace MR
{

enum class DepthFunction
{
    Never = 0,
    Less = 1,
    Equal = 2,
    Greater = 4,
    LessOrEqual = Less | Equal,
    GreaterOrEqual = Greater | Equal,
    NotEqual = Less | Greater,
    Always = Less | Equal | Greater,
    Default = 8 // usually "Less" but may differ for different object types
};
MR_MAKE_FLAG_OPERATORS( DepthFunction )

/// describes basic rendering parameters in a viewport
struct BaseRenderParams
{
    const Matrix4f& viewMatrix;
    const Matrix4f& projMatrix;
    ViewportId viewportId;       // id of the viewport
    Vector4i viewport;           // viewport x0, y0, width, height
};

/// describes parameters necessary to render an object
struct ModelRenderParams : BaseRenderParams
{
    const Matrix4f& modelMatrix;
    const Matrix4f* normMatrixPtr{ nullptr }; // normal matrix, only necessary for triangles rendering
    const Plane3f& clipPlane;    // viewport clip plane (it is not applied while object does not have clipping flag set)
    DepthFunction depthFunction = DepthFunction::Default;
    Vector3f lightPos;           // position of light source, unused for picker
    bool alphaSort{ false };     // if this flag is true shader for alpha sorting is used, unused for picker
};

struct BasicUiRenderTask
{
    virtual ~BasicUiRenderTask() = default;

    BasicUiRenderTask() = default;
    BasicUiRenderTask( const BasicUiRenderTask& ) = delete;
    BasicUiRenderTask& operator=( const BasicUiRenderTask& ) = delete;

    /// The tasks are sorted by this depth, descending (larger depth = further away).
    float renderTaskDepth = 0;

    struct BackwardPassParams
    {
        mutable bool mouseHoverConsumed = false;
    };

    /// This is an optional early pass, where you can claim exclusive control over the mouse.
    /// If you want to handle clicks or hovers, do it here, only if the argument is false. Then set it to true, if you handled the click/hover.
    virtual void earlyBackwardPass( const BackwardPassParams& params ) { (void)params; }

    /// This is the main rendering pass.
    virtual void renderPass() = 0;
};

struct UiRenderParams : BaseRenderParams
{
    /// Multiply all your sizes by this amount. Unless they are already premultipled, e.g. come from `ImGui::GetStyle()`.
    float scale = 1;

    using UiTaskList = std::vector<std::shared_ptr<BasicUiRenderTask>>;

    // Those are Z-sorted and then executed.
    UiTaskList* tasks = nullptr;
};

struct UiRenderManager
{
    virtual ~UiRenderManager() = default;

    virtual void preRenderViewport( ViewportId viewport ) { (void)viewport; }
    virtual void postRenderViewport( ViewportId viewport ) { (void)viewport; }

    // Call this once per viewport.
    virtual BasicUiRenderTask::BackwardPassParams getBackwardPassParams() { return {}; }
};

class IRenderObject
{
public:
    virtual ~IRenderObject() = default;
    // These functions do:
    // 1) bind data
    // 2) pass shaders arguments
    // 3) draw data
    virtual void render( const ModelRenderParams& params ) = 0;
    virtual void renderPicker( const ModelRenderParams& params, unsigned geomId ) = 0;
    /// returns the amount of memory this object occupies on heap
    virtual size_t heapBytes() const = 0;
    /// returns the amount of memory this object allocated in OpenGL
    virtual size_t glBytes() const = 0;
    /// binds all data for this render object, not to bind ever again (until object becomes dirty)
    virtual void forceBindAll() {}

    /// Render the UI. This is repeated for each viewport.
    /// Here you can either render immediately, or insert a task into `params.tasks`, which get Z-sorted.
    /// * `params` will remain alive as long as the tasks are used.
    /// * You'll have at most one living task at a time, so you can write a non-owning pointer to an internal task.
    virtual void renderUi( const UiRenderParams& params ) { (void)params; }
};
// Those dummy definitions remove undefined references in `RenderObjectCombinator` when it calls non-overridden pure virtual methods.
// We could check in `RenderObjectCombinator` if they're overridden or not, but it's easier to just define them.
inline size_t IRenderObject::heapBytes() const { return 0; }
inline size_t IRenderObject::glBytes() const { return 0; }

// Combines several different `IRenderObject`s into one in a meaningful way.
template <typename ...Bases>
requires ( ( std::derived_from<Bases, IRenderObject> && !std::same_as<Bases, IRenderObject> ) && ... )
class RenderObjectCombinator : public Bases...
{
public:
    RenderObjectCombinator( const VisualObject& object )
        : Bases( object )...
    {}

    size_t heapBytes() const override { return ( std::size_t{} + ... + Bases::heapBytes() ); }
    size_t glBytes() const override { return ( std::size_t{} + ... + Bases::glBytes() ); }
    void forceBindAll() override { ( Bases::forceBindAll(), ... ); }
    void renderUi( const UiRenderParams& params ) override { ( Bases::renderUi( params ), ... ); }
};

MRMESH_API std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj, const std::type_index& type );

template<typename ObjectType>
std::unique_ptr<IRenderObject> createRenderObject( const VisualObject& visObj )
{
    static_assert( std::is_base_of_v<VisualObject, std::remove_reference_t<ObjectType>>, "MR::VisualObject is not base of ObjectType" );
    return createRenderObject( visObj, typeid( ObjectType ) );
}

using IRenderObjectConstructorLambda = std::function<std::unique_ptr<IRenderObject>( const VisualObject& )>;

template<typename RenderObjType>
IRenderObjectConstructorLambda makeRenderObjectConstructor()
{
    return [] ( const VisualObject& visObj ) { return std::make_unique<RenderObjType>( visObj ); };
}

class RegisterRenderObjectConstructor
{
public:
    MRMESH_API RegisterRenderObjectConstructor( const std::type_index& type, IRenderObjectConstructorLambda lambda );
};

#define MR_REGISTER_RENDER_OBJECT_IMPL(objectType,rendObjectType)\
    static MR::RegisterRenderObjectConstructor __objectRegistrator##objectType{typeid(objectType),makeRenderObjectConstructor<rendObjectType>()};

}
