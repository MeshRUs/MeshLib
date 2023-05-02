#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRViewer.h"
#include "MRMesh/MRHistoryAction.h"
#include "MRMesh/MRViewportProperty.h"
#include <boost/signals2/signal.hpp>
#include <array>
#include <functional>
#include <string>

namespace MR
{

// Visual widget to modify transform
// present in scene (ancillary), subscribes to viewer events
class MRVIEWER_CLASS ObjectTransformWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener, PreDrawListener, DrawListener>
{
public:
    enum Axis { X, Y, Z, Count };
    enum TransformMode
    {
        RotX = 0x1,
        RotY = 0x2,
        RotZ = 0x4,
        MoveX = 0x8,
        MoveY = 0x10,
        MoveZ = 0x20,
        FullMask = 0x3f
    };
    struct Params
    {
        float radius{ -1.0f };
        float width{ -1.0f };
        // by default - center of given box
        // updated in create function
        Vector3f center;
        /// the product of this factor and width gives cone radius of the arrows
        float coneRadiusFactor{ 1.35f };
        /// the product of this factor and width gives cone size of the arrows
        float coneSizeFactor{ 2.2f };
        /// extension of the translation line in the negative direction relative to the radius
        float negativeLineExtension{ 1.15f };
        /// extension of the translation line in the positive direction relative to the radius
        float positiveLineExtension{ 1.3f };
        /// colors of widget
        std::array<Color, size_t( Axis::Count )> rotationColors{ Color::red(),Color::green(),Color::blue() };
        std::array<Color, size_t( Axis::Count )> translationColors{ Color::red(),Color::green(),Color::blue() };
        Color helperLineColor{ Color::black() };
        Color activeLineColor{ Color::white() };
    };
    // Creates transform widget around given box and applies given xf
    // subscribes to viewer events
    MRVIEWER_API void create( const Box3f& box, const AffineXf3f& xf );
    // Removes widget from scene and clears all widget objects
    // unsubscribes from viewer events
    MRVIEWER_API void reset();

    // get current width of widget controls
    // negative value means that controls are not setup
    float getWidth() const { return params_.width; }
    // get current radius of widget controls
    // negative value means that controls are not setup
    float getRadius() const { return params_.radius; }
    // get center of the widget in local space
    const Vector3f& getCenter() const { return params_.center; }
    // gets current parameters of this widget
    const Params & getParams() const { return params_; }

    // set width for this widget
    MRVIEWER_API void setWidth( float width );
    // set radius for this widget
    MRVIEWER_API void setRadius( float radius );
    // set center in local space for this widget
    MRVIEWER_API void setCenter( const Vector3f& center );
    // set current parameters of this widget
    MRVIEWER_API void setParams( const Params & );

    // Returns current transform mode mask
    uint8_t getTransformModeMask( ViewportId id = {} ) const { return transformModeMask_.get( id ); }
    // Sets transform mode mask (enabling or disabling corresponding widget controls)
    MRVIEWER_API void setTransformMode( uint8_t mask, ViewportId id = {} );

    // Enables or disables pick through mode, in this mode controls will be picked even if they are occluded by other objects
    void setPickThrough( bool on ) { pickThrough_ = on; }
    bool getPickThrough() const { return pickThrough_; }

    // Transform operation applying to object while dragging an axis. This parameter does not apply to active operation.
    enum AxisTransformMode
    {
        // object moves along an axis
        AxisTranslation,
        // object inflates or deflates along an axis depending on drag direction (away from center or toward center respectively)
        AxisScaling,
        // object inflates or deflates along all axes depending on drag direction (away from center or toward center respectively)
        UniformScaling,
    };
    // Returns current axis transform mode (translate/scale object while dragging an axis)
    AxisTransformMode getAxisTransformMode() const { return axisTransformMode_; };
    // Sets current axis transform mode (translate/scale object while dragging an axis)
    void setAxisTransformMode( AxisTransformMode mode ) { axisTransformMode_ = mode; };

    // Returns root object of widget
    std::shared_ptr<Object> getRootObject() const { return controlsRoot_; }

    // Changes controls xf (controls will affect object in basis of new xf)
    // note that rotation is applied around 0 coordinate in world space, so use xfAround to process rotation around user defined center
    // non-uniform scale will be converted to uniform one based on initial box diagonal
    MRVIEWER_API void setControlsXf( const AffineXf3f& xf, ViewportId id = {} );
    MRVIEWER_API AffineXf3f getControlsXf( ViewportId id = {} ) const;

    // This lambda is called in each frame, and returns transform mode mask for this frame in given viewport
    // if not set, full mask is return
    using ModesValidator = std::function<uint8_t( const ObjectTransformWidget&, ViewportId )>;
    void setTransformModesValidator( ModesValidator validator ) { modesValidator_ = validator; }

    // returns ModesValidator by threshold dot value (this value is duty for hiding widget controls that have small projection on screen)
    MRVIEWER_API static ModesValidator ThresholdDotValidator( float thresholdDot );

    // Subscribes to object visibility, and behave like its child
    // if obj argument is null, stop following
    MRVIEWER_API void followObjVisibility( const std::weak_ptr<Object>& obj );

    // Sets callback that will be called in draw function during scaling with current scale arg
    void setScaleTooltipCallback( std::function<void( float )> callback ) { scaleTooltipCallback_ = callback; }
    // Sets callback that will be called in draw function during translation with current shift arg
    void setTranslateTooltipCallback( std::function<void( float )> callback ) { translateTooltipCallback_ = callback; }
    // Sets callback that will be called in draw function during rotation with current angle in rad
    void setRotateTooltipCallback( std::function<void( float )> callback ) { rotateTooltipCallback_ = callback; }

    // Sets callback that will be called when modification of widget stops
    void setStopModifyCallback( std::function<void()> callback ) { stopModifyCallback_ = callback; }
    // Sets callback that will be called when modification of widget starts
    void setStartModifyCallback( std::function<void()> callback ) { startModifyCallback_ = callback; }
    // Sets callback that will be called when widget gets addictive transform
    void setAddXfCallback( std::function<void( const AffineXf3f& )> callback ) { addXfCallback_ = callback; }
    // Sets callback that will be called when widget gets addictive transform
    // The callback should return true to approve transform and false to reject it
    void setApproveXfCallback( std::function<bool( const AffineXf3f& )> callback ) { approveXfCallback_ = callback; }

    // History action for TransformWidget
    class ChangeXfAction : public HistoryAction
    {
    public:
        ChangeXfAction( const std::string& name, ObjectTransformWidget& widget ) :
            widget_{ widget },
            name_{ name }
        {
            if ( widget_.controlsRoot_ )
            {
                xf_ = widget_.controlsRoot_->xfsForAllViewports();
                scaledXf_ = widget_.scaledXf_;
            }
        }

        virtual std::string name() const override
        {
            return name_;
        }

        virtual void action( HistoryAction::Type ) override
        {
            if ( !widget_.controlsRoot_ )
                return;
            auto tmpXf = widget_.controlsRoot_->xfsForAllViewports();
            widget_.controlsRoot_->setXfsForAllViewports( xf_ );
            xf_ = tmpXf;

            std::swap( scaledXf_, widget_.scaledXf_ );
        }

        [[nodiscard]] virtual size_t heapBytes() const override
        {
            return name_.capacity();
        }

    private:
        ObjectTransformWidget& widget_;
        ViewportProperty<AffineXf3f> xf_;
        ViewportProperty<AffineXf3f> scaledXf_;
        std::string name_;
    };
    class ChangeParamsAction : public HistoryAction
    {
    public:
        ChangeParamsAction( const std::string& name, ObjectTransformWidget& widget ) :
            widget_{ widget },
            name_{ name }
        {
            params_ = widget_.getParams();
        }

        virtual std::string name() const override
        {
            return name_;
        }

        virtual void action( HistoryAction::Type ) override
        {
            auto params = widget_.getParams();
            widget_.setParams( params_ );
            params_ = params;
        }

        [[nodiscard]] virtual size_t heapBytes() const override
        {
            return name_.capacity();
        }

    private:
        ObjectTransformWidget& widget_;
        Params params_;
        std::string name_;
    };
private:
    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
    MRVIEWER_API virtual void preDraw_() override;
    MRVIEWER_API virtual void draw_() override;

    void passiveMove_();
    void activeMove_( bool press = false );

    void processScaling_( Axis ax, bool press );
    void processTranslation_( Axis ax, bool press );
    void processRotation_( Axis ax, bool press );

    void setControlsXf_( const AffineXf3f& xf, bool updateScaled, ViewportId id = {} );

    std::weak_ptr<Object> visibilityParent_;
    std::shared_ptr<ObjectMesh> currentObj_;

    void updateVisualTransformMode_( uint8_t showMask, ViewportMask viewportMask );

    void setActiveLineFromPoints_( const std::vector<Vector3f>& points );

    // undiformAddXf - for ActiveEditMode::ScalingMode only, to scale widget uniformly
    void addXf_( const AffineXf3f& addXf );
    void stopModify_();

    int findCurrentObjIndex_() const;

    void makeControls_();

    Params params_;

    // main object that holds all other controls
    std::shared_ptr<Object> controlsRoot_;
    std::array<std::shared_ptr<ObjectMesh>, size_t( Axis::Count )> translateControls_;
    std::array<std::shared_ptr<ObjectMesh>, size_t( Axis::Count )> rotateControls_;

    // if active line is visible, other lines are not
    std::shared_ptr<ObjectLines> activeLine_;
    std::array<std::shared_ptr<ObjectLines>, size_t( Axis::Count )> translateLines_;
    std::array<std::shared_ptr<ObjectLines>, size_t( Axis::Count )> rotateLines_;

    AxisTransformMode axisTransformMode_{ AxisTranslation };

    enum ActiveEditMode
    {
        TranslationMode,
        ScalingMode,
        UniformScalingMode,
        RotationMode,
    };
    ActiveEditMode activeEditMode_{ TranslationMode };

    // Initial box diagonal vector (before transformation), 
    // it is needed to correctly convert non-uniform scaling to uniform one and apply it to this widget
    Vector3f boxDiagonal_;
    // same as controlsRoot_->xf() but with non uniform scaling applied
    ViewportProperty<AffineXf3f> scaledXf_;
    // this is needed for tooltip only
    float currentScaling_ = 1.0f;

    Vector3f prevScaling_;
    Vector3f startTranslation_;
    Vector3f prevTranslation_;
    AffineXf3f startRotXf_;
    float startAngle_ = 0;
    float accumAngle_ = 0;

    ViewportProperty<uint8_t> transformModeMask_{ FullMask };
    float thresholdDot_{ 0.0f };
    bool picked_{ false };
    bool pickThrough_{ false };

    ModesValidator modesValidator_;

    std::function<void( float )> scaleTooltipCallback_;
    std::function<void( float )> translateTooltipCallback_;
    std::function<void( float )> rotateTooltipCallback_;

    std::function<void()> startModifyCallback_;
    std::function<void()> stopModifyCallback_;
    std::function<void( const AffineXf3f& )> addXfCallback_;
    std::function<bool( const AffineXf3f& )> approveXfCallback_;
    bool approvedChange_ = true; // if controlsRoot_ xf changed without approve, user modification stops
    boost::signals2::connection xfValidatorConnection_;
};

}