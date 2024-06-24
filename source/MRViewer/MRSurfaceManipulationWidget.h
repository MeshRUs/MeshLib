#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRViewer.h"
#include "MRMesh/MRChangeMeshAction.h"
#include "MRMesh/MREnums.h"
#include "MRViewer/MRViewport.h"
#include <chrono>

namespace MR
{

/// @brief widget for surface modifying
/// @detail available 3 modes:
/// add (move surface region in direction of normal)
/// remove (move surface region in opposite direction to normal)
/// relax (relax surface region)
class MRVIEWER_CLASS SurfaceManipulationWidget :
    public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener,
                         PostDrawListener>
{
public:
    MRVIEWER_API SurfaceManipulationWidget();
    MRVIEWER_API ~SurfaceManipulationWidget();

    /// widget work modes
    enum class WorkMode
    {
        Add,
        Remove,
        Relax,
        Laplacian,
        Patch
    };

    /// Mesh change settings
    struct Settings
    {
        WorkMode workMode = WorkMode::Add;
        float radius = 1.f; // radius of editing region
        float relaxForce = 0.2f; // speed of relaxing, typical values (0 - 0.5]
        float editForce = 1.f; // the force of changing mesh
        float sharpness = 50.f; // effect of force on points far from center editing area. [0 - 100]
        float relaxForceAfterEdit = 0.25f; //  force of relaxing modified area after editing (add / remove) is complete. [0 - 0.5], 0 - not relax
        EdgeWeights edgeWeights = EdgeWeights::Cotan; // edge weights for Laplacian and Patch
    };

    /// initialize widget according ObjectMesh
    MRVIEWER_API void init( const std::shared_ptr<ObjectMesh>& objectMesh );
    /// reset widget state
    MRVIEWER_API void reset();

    /// set widget settings (mesh change settings)
    MRVIEWER_API void setSettings( const Settings& settings );
    /// get widget settings 
    MRVIEWER_API const Settings& getSettings() { return settings_; };

    // mimum radius of editing area.
    MRVIEWER_API float getMinRadius() { return minRadius_; };

private:
    /// start modifying mesh surface
    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    /// stop modifying mesh surface, generate history action
    MRVIEWER_API bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    /// update
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y ) override;
    /// need to visualize bad region (draw grey circle)
    MRVIEWER_API void postDraw_() override;

    void initConnections_();
    void resetConnections_();

    void changeSurface_();
    void updateUVmap_( bool set );
    void updateRegion_( const Vector2f& mousePos );
    void abortEdit_();
    // Laplacian
    void laplacianPickVert_( const PointOnFace& pick );
    void laplacianMoveVert_( const Vector2f& mousePos );

    void updateVizualizeSelection_( const ObjAndPick& objAndPick );

    Settings settings_;

    std::shared_ptr<ObjectMesh> obj_;
    float diagonal_ = 1.f;
    float minRadius_ = 1.f;
    Vector2f mousePos_; ///< mouse position of last updateRegion_
    VertBitSet singleEditingRegion_;  ///< current (under the cursor) region of tool application
    VertBitSet visualizationRegion_;  ///< vertices of triangles partially or fully highlighted with red
    VertBitSet generalEditingRegion_; ///< united region of tool application since the last mouse down
    VertScalars pointsShift_;
    VertScalars editingDistanceMap_;
    VertScalars visualizationDistanceMap_;
    std::shared_ptr<ObjectMesh> oldMesh_;
    bool firstInit_ = true; // need to save settings in re-initial
    bool badRegion_ = false; // in selected region less than 3 points

    bool mousePressed_ = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> timePoint_;
    boost::signals2::scoped_connection meshChangedConnection_;
    bool ownMeshChangedSignal_ = false;

    bool connectionsInitialized_ = false;

    // Laplacian
    VertId touchVertId_; // we fix this vertex in Laplacian and move it manually
    Vector3f touchVertIniPos_; // initial position of fixed vertex
    Vector2i storedDown_;
    std::unique_ptr<Laplacian> laplacian_;
    std::shared_ptr<ChangeMeshAction> historyAction_; // this action is prepared beforehand for better responsiveness, but pushed only on mouse move
    bool appendHistoryAction_ = false;
};

}
