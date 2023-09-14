#pragma once

#include "MRId.h"
#include "MRHeap.h"
#include <cfloat>

namespace MR
{

class PrecipitationSimulator
{
public:
    MRMESH_API PrecipitationSimulator( WatershedGraph& wg );

    enum class Event
    {
        Finish,     ///< all basins are full and water goes outside
        BasinFull,  ///< one basin just became full
        Merge       ///< two basins just merged
    };

    struct SimulationStep
    {
        Event event = Event::Finish;
        float time = FLT_MAX;
        GraphVertId basin;     ///< BasinFull: this basin just became full
                               ///< Merge: this basin just absorbed the other basin
        GraphVertId neiBasin;  ///< BasinFull: the flow from full basin will first go here (may be not the last destination)
                               ///< Merge: this basin was just absorbed
    };

    /// processes the next event in precipitation
    MRMESH_API SimulationStep simulateOne();

private:
    WatershedGraph& wg_;
    static constexpr float infTime = FLT_MAX;
    Heap<float, GraphVertId, std::greater<float>> heap_;
};

} //namespace MR

