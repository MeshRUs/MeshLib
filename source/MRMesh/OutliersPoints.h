#pragma once
#include "MRMeshFwd.h"

namespace MR
{

class DetectOutliersPoints
{
public:
    void init( std::shared_ptr<PointCloud> pointCloud );

    // process all point for preparing statistics data
    void preprocess( ProgressCallback progress ); // prepareCachedData

    // process statistics data to find outliers points
    const VertBitSet& calculate( ProgressCallback progress );

    // general setting
    void setRadius( float );
    float getRadius() const;



    class OutliersType
    {
    public:
        const VertBitSet& calculate( const VertBitSet&, ProgressCallback ) = 0; // prepareToUpdatePoints
        const VertBitSet& getLastResult() const;

        void setActive( bool active );
        void setAvailable( bool available );

    private:
        bool available_ = true;
        bool active_ = true;

        VertBitSet resultVerts_;
    };

    class SmallComponents : public OutliersType
    {
    public:
        SmallComponents() : OutliersType( true )
        {}

        void calculate();

        void setMaxClasterSize( int maxClasterSize );
        int getMaxClasterSize() const;

    private:
        int maxClasterSize_;
    };

    class WeaklyConnected : public OutliersType
    {
    public:

        void setMaxNeighbors( int maxNeighbors );
        int getMaxNeighbors() const;

    private:
        int maxNeighbors_;
    };

    class FarSurface : public OutliersType
    {
    public:

        void setMinHeight( float minHeight );
        float getMinHeight() const;

    private:
        float minHeight_;
    };

    class AwayNormal : public OutliersType
    {
    public:

        void setMinAngle( float minAngle );
        float getMinAngle() const;

    private:
        float minAngle_;
    };

    const SmallComponents& getSmallComponents();
    const WeaklyConnected& getWeaklyConnected();
    const FarSurface& getFarSurface();
    const AwayNormal& getAwayNormal();

private:
    float radius_ = 0.f;
    std::shared_ptr<PointCloud> pointCloud_;

    SmallComponents smallComponents_;
    WeaklyConnected weaklyConnected_;
    FarSurface farSurface_;
    AwayNormal awayNormal_;
};

}
