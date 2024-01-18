#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"
#include "MRRibbonSchema.h"
#include <string>
#include <functional>

namespace MR
{

class RibbonButtonDrawer;
class RibbonFontManager;

// separate class for search in ribbon menu
class MRVIEWER_CLASS RibbonMenuSearch
{
public:
    // returns search imgui popup window name
    const char* windowName() const { return "##RibbonGlobalSearchPopup"; }
    // add item to recent items list
    void pushRecentItem( const std::shared_ptr<RibbonMenuItem>& item );

    struct Parameters
    {
        RibbonButtonDrawer& btnDrawer;
        RibbonFontManager& fontManager;
        std::function<void( int )> changeTabFunc;
        float scaling;
    };
    // draws search elements and window with its logic
    MRVIEWER_API void drawMenuUI( const Parameters& params );

    MRVIEWER_API bool isSmallUI();

    MRVIEWER_API float getWidthMenuUI();
private:
    bool smallSearchButton_( const Parameters& params );

    void drawWindow_( const Parameters& params );

    void deactivateSearch_();

    std::string searchLine_;
    std::vector<RibbonSchemaHolder::SearchResult> searchResult_;
    std::vector<RibbonSchemaHolder::SearchResult> recentItems_;
    int hightlightedSearchItem_{ -1 };
    bool popupWasOpen_ = false;

    bool active_ = false;
    bool isSmallUILast_ = false;
    bool windowInputWasActive_ = false;
    bool mainInputActive_ = false;
};

}