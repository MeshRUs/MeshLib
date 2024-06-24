#include "MRPdf.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_PDF )
#include "MRSerializer.h"
#include "MRImageSave.h"
#include "MRVector2.h"
#include "MRImage.h"
#include "MRBox.h"
#include "MRStringConvert.h"
#include "MRGTest.h"
#include "MRPch/MRSpdlog.h"
#include "MRSystem.h"
#include <fstream>

#undef NOMINMAX

#include "hpdf.h"

namespace MR
{

namespace
{
// size of A4 page in pixels (uses 72 PPI)
// https://www.papersizes.org/a-sizes-in-pixels.htm
// TODO need get this value from PoDoFo
constexpr HPDF_REAL pageWidth = 595.;
constexpr HPDF_REAL pageHeight = 842.;
constexpr HPDF_REAL scaleFactor = static_cast<HPDF_REAL>(17. / 6.); // ~2.8(3)

constexpr HPDF_REAL borderFieldLeft = 20 * scaleFactor;
constexpr HPDF_REAL borderFieldRight = pageWidth - 10 * scaleFactor;
constexpr HPDF_REAL borderFieldTop = pageHeight - 10 * scaleFactor;
constexpr HPDF_REAL borderFieldBottom = 10 * scaleFactor;
constexpr HPDF_REAL pageWorkWidth = borderFieldRight - borderFieldLeft;
constexpr HPDF_REAL pageWorkHeight = borderFieldTop - borderFieldBottom;

constexpr HPDF_REAL spacing = 6 * scaleFactor;

constexpr HPDF_REAL textSpacing = 4 * scaleFactor;

constexpr HPDF_REAL labelHeight = 10 * scaleFactor;
constexpr HPDF_REAL marksHeight = 10 * scaleFactor;
constexpr HPDF_REAL marksWidth = 30 * scaleFactor;

}

Pdf::Pdf( const std::filesystem::path& documentPath, const PdfParameters& params /*= PdfParameters()*/ )
    : filename_{ documentPath }
    , params_( params )
{
    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;

    document_ = HPDF_New( NULL, NULL );
    if ( !document_ )
    {
        spdlog::warn( "Can't create PDF document. HPDF error code {}", HPDF_GetError( document_ ) );
        return;
    }
    HPDF_SetCompressionMode( document_, HPDF_COMP_ALL );
    activePage_ = HPDF_AddPage( document_ );
    if ( !activePage_ )
    {
        spdlog::warn( "Can't create page. HPDF error code {}", HPDF_GetError( document_ ) );
        return;
    }

    HPDF_Page_SetSize( activePage_, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT );
    activeFont_ = HPDF_GetFont( document_, params_.fontName.c_str(), NULL );
    if ( !activeFont_ )
    {
        spdlog::debug( "Can't find font: \"{}\". HPDF error code {}", params_.fontName, HPDF_GetError( document_ ) );
        return;
    }

    HPDF_Page_SetFontAndSize( activePage_, activeFont_, params_.textSize );
}

Pdf::~Pdf()
{
    close();
}

void Pdf::addText(const std::string& text, bool isTitle /*= false*/)
{
    if ( !document_ )
    {
        spdlog::warn( "Can't add text to pdf page: no valid document" );
        return;
    }

    int strNum = 1;
    size_t pos = text.find( '\n', 0 );
    while ( pos != std::string::npos )
    {
        ++strNum;
        pos = text.find( '\n', pos + 1 );
    }


    const HPDF_REAL textHeight = static_cast<HPDF_REAL>(( isTitle ? params_.titleSize : params_.textSize ) * strNum + textSpacing * 2.);

    if ( cursorY_ - textHeight < borderFieldBottom )
        newPage();

    HPDF_TextAlignment alignment = isTitle ? HPDF_TALIGN_CENTER : HPDF_TALIGN_LEFT;
    HPDF_Page_SetFontAndSize( activePage_, activeFont_, ( isTitle ? params_.titleSize : params_.textSize ) );
    HPDF_Page_BeginText( activePage_ );
    HPDF_Page_SetTextLeading( activePage_, textSpacing );
    HPDF_Page_TextRect( activePage_, cursorX_, cursorY_, cursorX_ + pageWorkWidth, cursorY_ - textHeight, text.c_str(), alignment, nullptr );
    HPDF_Page_EndText( activePage_ );

    cursorY_ -= textHeight;
    if ( cursorY_ - spacing < borderFieldBottom )
        newPage();
    else
        cursorY_ -= spacing;
}

void Pdf::addImageFromFile( const std::filesystem::path& imagePath, const std::string& caption /*= {}*/,
        const std::vector<std::pair<double, std::string>>& valuesMarks /*= {}*/ )
{
    if ( !document_ )
    {
        spdlog::warn( "Can't add image to pdf page: no valid document" );
        return;
    }

    HPDF_Image pdfImage = HPDF_LoadPngImageFromFile( document_, utf8string( imagePath ).c_str() );
    if ( !pdfImage )
    {
        spdlog::warn( "Failed to load image from file. HPDF error code {}", HPDF_GetError( document_ ) );
        return;
    }

    const HPDF_REAL additionalHeight = marksHeight * valuesMarks.empty() + marksHeight * !valuesMarks.empty() + labelHeight * !caption.empty();
    const HPDF_REAL scalingFactor = std::min( ( pageWorkHeight - additionalHeight ) / HPDF_Image_GetHeight( pdfImage ), pageWorkWidth / HPDF_Image_GetWidth( pdfImage ) );
    const HPDF_REAL scalingWidth = scalingFactor * HPDF_Image_GetWidth( pdfImage );
    const HPDF_REAL scalingHeight = scalingFactor * HPDF_Image_GetHeight( pdfImage );

    if ( cursorY_ - scalingHeight - additionalHeight < borderFieldBottom )
        newPage();

    cursorY_ -= scalingHeight;
    HPDF_Page_DrawImage( activePage_, pdfImage, cursorX_, cursorY_, scalingWidth, scalingHeight );

    if ( !valuesMarks.empty() )
    {
        HPDF_REAL step = pageWorkWidth - marksWidth / 2.;
        if ( valuesMarks.size() > 1)
            step /= ( valuesMarks.size() - 1);
        HPDF_REAL posX = cursorX_;
        for ( auto& mark : valuesMarks )
        {
            HPDF_Page_BeginText( activePage_ );
            HPDF_Page_SetFontAndSize( activePage_, activeFont_, params_.textSize );
            HPDF_Page_MoveTextPos( activePage_, posX, cursorY_ - marksHeight / 2 );
            HPDF_Page_ShowText( activePage_, mark.second.c_str() );
            HPDF_Page_EndText( activePage_ );
            posX += step;
        }
        cursorY_ -= marksHeight;
    }
    if ( !caption.empty() )
    {
        cursorY_ -= textSpacing / 2.;
        HPDF_Page_BeginText( activePage_ );
        HPDF_Page_SetFontAndSize( activePage_, activeFont_, params_.textSize );
        HPDF_Page_TextRect( activePage_, cursorX_, cursorY_, cursorX_ + pageWorkWidth, cursorY_ - labelHeight, caption.c_str(), HPDF_TALIGN_CENTER, nullptr );
        HPDF_Page_EndText( activePage_ );
        cursorY_ -= labelHeight;
    }

    if ( cursorY_ - spacing < borderFieldBottom )
        newPage();
    else
        cursorY_ -= spacing;
}

void Pdf::newPage()
{
    if ( !document_ )
    {
        spdlog::warn( "Can't create new pdf page: no valid document" );
        return;
    }

    activePage_ = HPDF_AddPage( document_ );
    if ( !activePage_ )
    {
        spdlog::warn( "Error while creating new pdf page: {}", HPDF_GetError( document_ ) );
        return;
    }

    cursorX_ = borderFieldLeft;
    cursorY_ = borderFieldTop;
    HPDF_Page_SetSize( activePage_, HPDF_PAGE_SIZE_A4, HPDF_PAGE_PORTRAIT);
}

void Pdf::close()
{
    if ( document_ )
    {
        // reset all errors before saving document
        HPDF_ResetError( document_ );

        auto pathString = utf8string( filename_ );
        HPDF_SaveToFile( document_, pathString.c_str() );

        HPDF_STATUS status = HPDF_GetError( document_ );
        if (status != HPDF_OK)
        {
            spdlog::error( "HPDF Error while saving pdf: {}", status );
            HPDF_ResetError( document_ );
        }
        HPDF_Free( document_ );
        document_ = nullptr;
    }
    activePage_ = nullptr;
    activeFont_ = nullptr;
}

TEST( MRMesh, Pdf )
{
    UniqueTemporaryFolder pathFolder( {} );
    Pdf pdfTest( pathFolder / std::filesystem::path( "test.pdf" ) );
    pdfTest.addText( "Test Title", true );
    pdfTest.addText( "Test text"
        "\nstring 1"
        "\nstring 2" );

    const int colorMapSizeX = int( pageWorkWidth);
    const int colorMapSizeY = int( 10 * scaleFactor );
    std::vector<Color> pixels( colorMapSizeX * colorMapSizeY );
    Color colorLeft = Color::blue();
    Color colorRight = Color::red();
    for ( int i = 0; i < colorMapSizeX; ++i )
    {
        for ( int j = 0; j < colorMapSizeY; ++j )
        {
            const float c = float( i ) / colorMapSizeX;
            pixels[i + j * colorMapSizeX] = ( 1 - c ) * colorLeft + c * colorRight;
        }
    }

    auto colorMapPath = pathFolder / std::filesystem::path( "color_map.png" );
    auto res = ImageSave::toAnySupportedFormat( { pixels, Vector2i( colorMapSizeX, colorMapSizeY ) }, colorMapPath );

    pdfTest.addImageFromFile( colorMapPath, "test image" );
    pdfTest.close();
}

}
#endif
