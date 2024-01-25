#pragma once

#include "MRViewer.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRSurfacePointPicker.h"
#include "MRMesh/MRObjectMeshHolder.h"
#include "MRMesh/MRHistoryStore.h"
#include "MRViewer/MRGladGlfw.h"
#include <unordered_map>

namespace MR
{

class SurfaceContoursWidget : public MultiListener<
    MouseDownListener,
    MouseMoveListener>
{
public:

    struct SurfaceContoursWidgetParams {
        int widgetContourCloseMod = GLFW_MOD_CONTROL;
        int widgetDeletePointMod = GLFW_MOD_SHIFT;
        bool writeHistory = true;
        bool flashHistoryAtEnd = true;
        SurfacePointWidget::Parameters surfacePointParams;
        MR::Color ordinaryPointColor = Color::gray();
        MR::Color lastPoitColor = Color::green();
        MR::Color closeContourPointColor = Color::transparent();
    };

    using PickerPointCallBack = std::function<void( std::shared_ptr<MR::ObjectMeshHolder> )>;
    using PickerPointObjectChecker = std::function<bool( std::shared_ptr<MR::ObjectMeshHolder> )>;

    using SurfaceContour = std::vector<std::shared_ptr<SurfacePointWidget>>;
    using SurfaceContours = std::unordered_map <std::shared_ptr<MR::ObjectMeshHolder>, SurfaceContour>;

    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;

    // enable or disable widget
    MRVIEWER_API void enable( bool isEnaled );

    // create a widget and connect it. 
    MRVIEWER_API void create( 
        PickerPointCallBack onPointAdd, 
        PickerPointCallBack onPointMove, 
        PickerPointCallBack onPointMoveFinish, 
        PickerPointCallBack onPointRemove,
        PickerPointObjectChecker isObjectValidToPick
    );

    // clear temp internal variables.
    MRVIEWER_API void clear();

    // reset widget, clear internal variables and detach from signals.
    MRVIEWER_API void reset();

    // return contour for specific object, i.e. ordered vector of surface points
    [[nodiscard]] const SurfaceContour& getSurfaceContour( const std::shared_ptr<MR::ObjectMeshHolder>& obj )
    {
        return pickedPoints_[obj];
    }

    // return all contours, i.e. per object umap of ordered surface points [vestor].
    [[nodiscard]] const SurfaceContours& getSurfaceContours() const
    {
        return pickedPoints_;
    }

    // chech is contour closed for particular object.
    [[nodiscard]] bool isClosedCountour( const std::shared_ptr<ObjectMeshHolder>& obj );

    // shared variables. which need getters and setters.
    int activeIndex{ 0 };
    std::shared_ptr<MR::ObjectMeshHolder> activeObject = nullptr;


    // configuration params
    SurfaceContoursWidgetParams params;

private:

    // creates point widget for add to contour.
    [[nodiscard]] std::shared_ptr<SurfacePointWidget> createPickWidget_( const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& pt );

    // SurfaceContoursWidget interlal variables 
    bool moveClosedPoint_ = false;
    bool activeChange_ = false;
    bool isPickerActive_ = false;

    // data storage
    SurfaceContours pickedPoints_;

    // CallBack functions
    PickerPointCallBack onPointAdd_;
    PickerPointCallBack onPointMove_;
    PickerPointCallBack onPointMoveFinish_;
    PickerPointCallBack onPointRemove_;
    PickerPointObjectChecker isObjectValidToPick_;

    friend class AddPointActionPickerPoint;
    friend class RemovePointActionPickerPoint;
    friend class ChangePointActionPickerPoint;
};


// History classes;
class AddPointActionPickerPoint : public HistoryAction
{
public:
    AddPointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point }
    {};

    virtual std::string name() const override;
    virtual void action( Type actionType ) override;
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    SurfaceContoursWidget& widget_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
};

class RemovePointActionPickerPoint : public HistoryAction
{
public:
    RemovePointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point, int index ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override;
    virtual void action( Type actionType ) override;
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    SurfaceContoursWidget& widget_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
    int index_;
};

class ChangePointActionPickerPoint : public HistoryAction
{
public:
    ChangePointActionPickerPoint( SurfaceContoursWidget& widget, const std::shared_ptr<MR::ObjectMeshHolder>& obj, const MeshTriPoint& point, int index ) :
        widget_{ widget },
        obj_{ obj },
        point_{ point },
        index_{ index }
    {};

    virtual std::string name() const override;
    virtual void action( Type ) override;
    [[nodiscard]] virtual size_t heapBytes() const override;
private:
    SurfaceContoursWidget& widget_;
    const std::shared_ptr<MR::ObjectMeshHolder> obj_;
    MeshTriPoint point_;
    int index_;
};



}