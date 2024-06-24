#include "MRBestFitParabola.h"
#include "MRGTest.h"

namespace MR
{

// verifies that template can be instantiated with typical parameters
template class BestFitParabola<float>;
template class BestFitParabola<double>;

TEST(MRMesh, BestFitParabola)
{
    const Parabolad p( 1, 2, 3 );

    BestFitParabolad fitter;
    fitter.addPoint( 1, p( 1 ) );
    fitter.addPoint( 2, p( 2 ) );
    fitter.addPoint( 3, p( 3 ) );

    const auto p1 = fitter.getBestParabola();
    constexpr double eps = 2e-11;
    EXPECT_NEAR( std::abs( p.a - p1.a ), 0., eps );
    EXPECT_NEAR( std::abs( p.b - p1.b ), 0., eps );
    EXPECT_NEAR( std::abs( p.c - p1.c ), 0., eps );
}

} //namespace MR
