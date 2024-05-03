#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRphmap.h"
#include <array>
#include <filesystem>

namespace MR
{

// this class holds icons for ribbon items
class MRVIEWER_CLASS RibbonIcons
{
public:
    enum class ColorType
    {
        Colored,
        White
    };
    enum class IconType
    {
        RibbonItemIcon,   // have four sizes
        ObjectTypeIcon,   // have two sizes
        IndependentIcons, // have two sizes
        Count
    };
    // this should be called once on start of programm (called in RibbonMenu::init)
    MRVIEWER_API static void load();
    // this should be called once before programm stops (called in RibbonMenu::shutdown)
    MRVIEWER_API static void free();
    // finds icon with best fitting size, if there is no returns nullptr
    MRVIEWER_API static const ImGuiImage* findByName( const std::string& name, float width, 
                                                      ColorType colorType, IconType iconType );
private:
    RibbonIcons();
    ~RibbonIcons() = default;

    static RibbonIcons& instance_();

    struct Icons
    {
        std::unique_ptr<ImGuiImage> colored;
        std::unique_ptr<ImGuiImage> white;
    };

    enum class Sizes
    {
        X0_5,
        X0_75,
        X1,
        X3,
        Count,
    };

    using SizedIcons = std::array<Icons, size_t( Sizes::Count )>;

    static const char* sizeSubFolder_( Sizes sz );

    Sizes findRequiredSize_( float width, IconType iconType ) const;

    void load_( IconType type );

    struct IconTypeData
    {
        std::filesystem::path pathDirectory;
        std::pair<Sizes, Sizes> size;
        ColorType colorType;
        HashMap<std::string, SizedIcons> map;
        std::array<int, size_t( Sizes::Count )> loadSize;
    };

    std::array<IconTypeData, size_t( IconType::Count )> data_;
};

}