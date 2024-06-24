#include "MRUIRectAllocator.h"

#include "MRViewer/MRViewer.h"

#include <imgui_internal.h>

namespace MR::UI
{

RectAllocator::RectAllocator() {}

RectAllocator::FindFreeRectResult RectAllocator::findFreeRect(
    Box2f preferredRect,
    Box2f preferredBounds,
    FindPotentiallyOverlappingRects findOverlaps,
    ImVec2 axisWeights
)
{
    FindFreeRectResult bestRect{ .rect = preferredRect, .ok = false };

    visitedCoords.clear();
    coordsToVisitHeap.clear();

    auto getOverlapWithBounds = [&]( Box2f rect ) -> std::array<float, 4>
    {
        std::array<float, 4> ret = {
            rect.max.x - preferredBounds.max.x,
            rect.max.y - preferredBounds.max.y,
            preferredBounds.min.x - rect.min.x,
            preferredBounds.min.y - rect.min.y,
        };

        for ( float& x : ret )
        {
            if ( x < 0 )
                x = 0;
        }
        return ret;
    };

    visitedCoords.try_emplace( preferredRect.min, 0 );
    coordsToVisitHeap.push_back( { .pos = preferredRect.min, .cost = 0, .overlapWithBounds = getOverlapWithBounds( preferredRect ) } );

    auto heapComparator = []( const CoordsToVisit& a, const CoordsToVisit& b )
    {
        // Min (sic!) heap by cost.
        return a.cost > b.cost;
    };

    while ( !coordsToVisitHeap.empty() )
    {
        const CoordsToVisit curCoords = coordsToVisitHeap.front();
        std::pop_heap( coordsToVisitHeap.begin(), coordsToVisitHeap.end(), heapComparator );
        coordsToVisitHeap.pop_back();

        const Box2f thisRect = Box2f::fromMinAndSize( curCoords.pos, preferredRect.size() );

        // Intersect against existing rects.
        bool anyIntersections = false;
        findOverlaps( thisRect, [&]( const char* name, Box2f otherRect )
        {
            (void)name;

            if ( !rectRectOverlap( thisRect, otherRect ) )
                return;

            anyIntersections = true;

            Box2f neighbors[4] = {
                { Vector2f( otherRect.max.x, thisRect.min.y ), Vector2f( otherRect.max.x + thisRect.size().x, thisRect.max.y ) }, // +X
                { Vector2f( thisRect.min.x, otherRect.max.y ), Vector2f( thisRect.max.x, otherRect.max.y + thisRect.size().y ) }, // +Y
                { Vector2f( otherRect.min.x - thisRect.size().x, thisRect.min.y ), Vector2f( otherRect.min.x, thisRect.max.y ) }, // -X
                { Vector2f( thisRect.min.x, otherRect.min.y - thisRect.size().y ), Vector2f( thisRect.max.x, otherRect.min.y ) }, // -Y
            };
            float deltaCosts[4] = {
                ( otherRect.max.x - thisRect.min.x ) * axisWeights[0],
                ( otherRect.max.y - thisRect.min.y ) * axisWeights[1],
                ( thisRect.max.x - otherRect.min.x ) * axisWeights[0],
                ( thisRect.max.y - otherRect.min.y ) * axisWeights[1],
            };

            for ( int i = 0; i < 4; i++ )
            {
                const Box2f& neighbor = neighbors[i];
                float cost = deltaCosts[i];
                assert( cost > 0 );
                cost += curCoords.cost;

                auto neighborOverlapWithBounds = getOverlapWithBounds( neighbor );
                bool badBoundsOverlap = false;
                for ( int j = 0; j < 4; j++ )
                {
                    if ( neighborOverlapWithBounds[j] > curCoords.overlapWithBounds[j] )
                    {
                        badBoundsOverlap = true;
                        break;
                    }
                }
                if ( badBoundsOverlap )
                    continue; // This rect is more out of bounds than the previous one, ignore.

                auto [visIter, visNew] = visitedCoords.try_emplace( neighbor.min, cost );
                if ( !visNew )
                {
                    if ( visIter->second <= cost )
                        continue; // Already visited with less cost, ignore.
                    visIter->second = cost;
                }

                coordsToVisitHeap.push_back( { .pos = neighbor.min, .cost = cost, .overlapWithBounds = neighborOverlapWithBounds } );
                std::push_heap( coordsToVisitHeap.begin(), coordsToVisitHeap.end(), heapComparator );
            }
        } );

        if ( !anyIntersections )
        {
            // Use this rect.
            bestRect.rect = thisRect;
            bestRect.ok = true;
            break;
        }
    }

    return bestRect;
}

void WindowRectAllocator::setFreeNextWindowPos( const char* expectedWindowName, ImVec2 defaultPos, ImGuiCond cond, ImVec2 pivot )
{
    // Once per frame, update the window list.
    if ( lastFrameCount != ImGui::GetFrameCount() )
    {
        lastFrameCount = ImGui::GetFrameCount();

        for ( auto it = windows.begin(); it != windows.end(); )
        {
            if ( !std::exchange( it->second.visitedThisFrame, false ) )
            {
                it = windows.erase( it );
                continue;
            }

            ++it;
        }
    }

    bool findLocation = false;
    ImGuiWindow* window = nullptr;

    if ( cond != 0 && cond != ImGuiCond_Always )
    {
        window = ImGui::FindWindowByName( expectedWindowName );
        if ( window )
        {
            auto [iter, isNew] = windows.try_emplace( expectedWindowName );
            if ( isNew )
            {
                findLocation = true;
                defaultPos = window->Pos;
            }
            else
            {
                iter->second.visitedThisFrame = true;
            }
        }
    }

    if ( findLocation )
    {
        Box2f bounds = getViewerInstance().getViewportsBounds();
        Box2f boundsFixed = bounds;
        boundsFixed.min.y = ImGui::GetIO().DisplaySize.y - boundsFixed.max.y;
        boundsFixed.max.y = boundsFixed.min.y + bounds.size().y;

        auto result = findFreeRect( Box2f::fromMinAndSize( defaultPos, window->Size ), boundsFixed, [&]( Box2f rect, std::function<void( const char*, Box2f )> func )
        {
            // Just output all the rects for now.
            // FIXME: An AABB tree would be nice here, for better performance.
            (void)rect;

            for ( const ImGuiWindow* win : ImGui::GetCurrentContext()->Windows )
            {
                if ( !win->WasActive || ( win->Flags & ImGuiWindowFlags_ChildWindow ) )
                    continue; // Skip inactive windows and child windows.
                std::string_view winNameView = win->Name;
                if ( auto pos = winNameView.find( "##" ); pos != std::string_view::npos && winNameView.find( "[rect_allocator_ignore]", pos + 2 ) != std::string_view::npos )
                    continue; // Ignore if the name contains the special tag.
                if ( std::strcmp( win->Name, expectedWindowName ) == 0 )
                    continue; // Skip the target window itself.
                func( win->Name, Box2f::fromMinAndSize( win->Pos, win->Size ) );
            }
        }, ImVec2( 5, 1 ) );

        defaultPos = result.rect.min;
        cond = ImGuiCond_Always;
    }

    ImGui::SetNextWindowPos( defaultPos, cond, pivot );
}

WindowRectAllocator& getDefaultWindowRectAllocator()
{
    static WindowRectAllocator ret;
    return ret;
}

}
