#include "MRImGuiMeasurementIndicators.h"

#include "MRViewer/MRColorTheme.h"
#include "MRViewer/MRImGuiVectorOperators.h"

namespace MR::ImGuiMeasurementIndicators
{
    using namespace ImGuiMath;

    static void expandTriangle( ImVec2& a, ImVec2& b, ImVec2& c, float value )
    {
        if ( value > 0 )
        {
            ImVec2 da = normalize( b - a );
            ImVec2 db = normalize( c - b );
            ImVec2 dc = normalize( a - c );
            a += ( dc - da ) / std::abs( dot( dc, rot90( da ) ) ) * value;
            b += ( da - db ) / std::abs( dot( da, rot90( db ) ) ) * value;
            c += ( db - dc ) / std::abs( dot( db, rot90( dc ) ) ) * value;
        }
    };

    template <typename F>
    static void forEachElement( Element elem, F&& func )
    {
        if ( bool( elem & Element::outline ) )
            func( Element::outline );
        if ( bool( elem & Element::main ) )
            func( Element::main );
    }

    Params::Params()
    {
        bool isDark = ColorTheme::getPreset() == ColorTheme::Preset::Dark;
        colorMain = isDark ? Color( 1.f, 1.f, 1.f, 1.f ) : Color( 0.f, 0.f, 0.f, 1.f );
        colorOutline = isDark ? Color( 0.f, 0.f, 0.f, 0.5f ) : Color( 1.f, 1.f, 1.f, 0.5f );
        colorText = colorMain;
    }

    float StringWithIcon::getIconWidth() const
    {
        switch ( icon )
        {
        case StringIcon::none:
            return 0;
        case StringIcon::diameter:
            return std::round( ImGui::GetTextLineHeight() );
        }
    }

    ImVec2 StringWithIcon::calcTextSize() const
    {
        return ImGui::CalcTextSize( string.data(), string.data() + string.size() ) + ImVec2( getIconWidth(), 0 );
    }

    void StringWithIcon::draw( ImDrawList& list, float menuScaling, ImVec2 pos, ImU32 color )
    {
        if ( icon == StringIcon{} )
        {
            list.AddText( pos, color, string.data(), string.data() + string.size() );
        }
        else
        {
            assert( iconPos <= string.size() );

            ImVec2 iconPixelPos = pos + ImVec2( ImGui::CalcTextSize( string.data(), string.data() + iconPos ).x, 0 );
            ImVec2 iconPixelSize( getIconWidth(), ImGui::GetTextLineHeight() );

            list.AddText( pos, color, string.data(), string.data() + iconPos );
            list.AddText( iconPixelPos + ImVec2( iconPixelSize.x, 0 ), color, string.data() + iconPos, string.data() + string.size() );

            switch ( icon )
            {
            case StringIcon::none:
                // Nothing, and this should be unreachable.
                break;
            case StringIcon::diameter:
                list.AddCircle( iconPixelPos + iconPixelSize / 2, iconPixelSize.x / 2 - 2 * menuScaling, color, 0, 1.1f * menuScaling );
                list.AddLine(
                    iconPixelPos + ImVec2( iconPixelSize.x - 1.5f, 0.5f ) - ImVec2( 0.5f, 0.5f ),
                    iconPixelPos + ImVec2( 1.5f, iconPixelSize.y - 0.5f ) - ImVec2( 0.5f, 0.5f ),
                    color, 1.1f * menuScaling
                );
                break;
            }
        }
    }

    void text( Element elem, float menuScaling, const Params& params, ImVec2 center, StringWithIcon string, ImVec2 push )
    {
        if ( ( elem & Element::both ) == Element{} )
            return; // Nothing to draw.

        if ( string.isEmpty() )
            return;

        float textOutlineWidth = params.textOutlineWidth * menuScaling;
        float textOutlineRounding = params.textOutlineRounding * menuScaling;
        float textToLineSpacingRadius = params.textToLineSpacingRadius * menuScaling;
        ImVec2 textToLineSpacingA = params.textToLineSpacingA * menuScaling;
        ImVec2 textToLineSpacingB = params.textToLineSpacingB * menuScaling;

        ImVec2 textSize = string.calcTextSize();
        ImVec2 textPos = center - textSize / 2;

        if ( push != ImVec2{} )
        {
            ImVec2 point = ImVec2( push.x > 0 ? textPos.x - textToLineSpacingA.x : textPos.x + textSize.x + textToLineSpacingB.x, push.y > 0 ? textPos.y - textToLineSpacingA.y : textPos.y + textSize.y + textToLineSpacingB.y );
            textPos += push * ( -dot( push, point - center ) + textToLineSpacingRadius );
        }

        if ( bool( elem & Element::outline ) )
            params.list->AddRectFilled( round( textPos ) - textToLineSpacingA - textOutlineWidth, textPos + textSize + textToLineSpacingB + textOutlineWidth, params.colorOutline.getUInt32(), textOutlineRounding );
        if ( bool( elem & Element::main ) )
            string.draw( *params.list, menuScaling, round( textPos ), params.colorText.getUInt32() );
    }

    void arrowTriangle( Element elem, float menuScaling, const Params& params, ImVec2 point, ImVec2 dir )
    {
        if ( ( elem & Element::both ) == Element{} )
            return; // Nothing to draw.

        float outlineWidth = params.outlineWidth * menuScaling;
        float arrowLen = params.arrowLen * menuScaling;
        float arrowHalfWidth = params.arrowHalfWidth * menuScaling;

        dir = normalize( dir );
        ImVec2 n = rot90( dir );

        ImVec2 a = point;
        ImVec2 b = a - dir * arrowLen + n * arrowHalfWidth;
        ImVec2 c =  a - dir * arrowLen - n * arrowHalfWidth;

        if ( bool( elem & Element::outline ) )
        {
            ImVec2 a2 = a;
            ImVec2 b2 = b;
            ImVec2 c2 = c;
            expandTriangle( a2, b2, c2, outlineWidth );

            params.list->AddTriangleFilled( a2, b2, c2, params.colorOutline.getUInt32() );
        }

        if ( bool( elem & Element::main ) )
            params.list->AddTriangleFilled( a, b, c, params.colorMain.getUInt32() );
    }

    void line( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, const LineParams& lineParams )
    {
        if ( ( elem & Element::both ) == Element{} )
            return; // Nothing to draw.

        if ( a == b )
            return;

        float lineWidth = ( bool( lineParams.flags & LineFlags::narrow ) ? params.smallWidth : params.width ) * menuScaling;
        float outlineWidth = params.outlineWidth * menuScaling;
        float arrowLen = params.arrowLen * menuScaling;

        forEachElement( elem, [&]( Element thisElem )
        {
            ImVec2 points[2] = {a, b};

            for ( bool front : { false, true } )
            {
                ImVec2& point = points[front];
                ImVec2 d = front
                    ? normalize( b - ( lineParams.midPoints.empty() ? a : lineParams.midPoints.back() ) )
                    : normalize( a - ( lineParams.midPoints.empty() ? b : lineParams.midPoints.front() ) );
                LineCap thisCap = front ? lineParams.capB : lineParams.capA;

                switch ( thisCap )
                {
                case LineCap::nothing:
                    if ( thisElem == Element::outline )
                        point += d * outlineWidth;
                    break;
                case LineCap::arrow:
                    arrowTriangle( thisElem, menuScaling, params, point, d );
                    point += d * ( -arrowLen + 1 ); // +1 is to avoid a hairline gap here, we intentionally don't multiply it by `menuScaling`.
                    break;
                }
            }

            params.list->PathLineTo( points[0] );
            for ( ImVec2 point : lineParams.midPoints )
                params.list->PathLineTo( point );
            params.list->PathLineTo( points[1] );

            params.list->PathStroke( ( thisElem == Element::main ? params.colorMain : params.colorOutline ).getUInt32(), 0, lineWidth + ( outlineWidth * 2 ) * ( thisElem == Element::outline ) );
        } );
    }

    void distance( Element elem, float menuScaling, const Params& params, ImVec2 a, ImVec2 b, StringWithIcon string )
    {
        if ( ( elem & Element::both ) == Element{} )
            return; // Nothing to draw.

        float textToLineSpacingRadius = params.textToLineSpacingRadius * menuScaling;
        ImVec2 textToLineSpacingA = params.textToLineSpacingA * menuScaling;
        ImVec2 textToLineSpacingB = params.textToLineSpacingB * menuScaling;
        float arrowLen = params.arrowLen * menuScaling;
        float totalLenThreshold = params.totalLenThreshold * menuScaling;
        float invertedOverhang = params.invertedOverhang * menuScaling;

        bool useInvertedStyle = lengthSq( b - a ) < totalLenThreshold * totalLenThreshold;
        bool drawTextOutOfLine = useInvertedStyle;

        ImVec2 dir = normalize( b - a );
        ImVec2 n( -dir.y, dir.x );

        ImVec2 center = a + ( b - a ) / 2;

        ImVec2 gapA, gapB;

        if ( !string.isEmpty() && !useInvertedStyle )
        {
            ImVec2 textSize = string.calcTextSize();
            ImVec2 textPos = a + ( ( b - a ) - textSize ) / 2.f;

            ImVec2 boxA = textPos - textToLineSpacingA - center;
            ImVec2 boxB = textPos + textSize + textToLineSpacingB - center;
            auto isInBox = [&]( ImVec2 pos ) { return CompareAll( pos ) >= boxA && CompareAll( pos ) <= boxB; };

            if ( isInBox( a ) || isInBox( b ) )
            {
                drawTextOutOfLine = true;
            }
            else
            {
                ImVec2 deltaA = a - center;
                ImVec2 deltaB = b - center;

                for ( ImVec2* delta : { &deltaA, &deltaB } )
                {
                    for ( bool axis : { false, true } )
                    {
                        if ( (*delta)[axis] < boxA[axis] )
                        {
                            (*delta)[!axis] *= boxA[axis] / (*delta)[axis];
                            (*delta)[axis] = boxA[axis];
                        }
                        else if ( (*delta)[axis] > boxB[axis] )
                        {
                            (*delta)[!axis] *= boxB[axis] / (*delta)[axis];
                            (*delta)[axis] = boxB[axis];
                        }
                    }
                }

                gapA = center + deltaA;
                gapB = center + deltaB;

                if ( length( a - gapA ) + length( b - gapB ) < totalLenThreshold + textToLineSpacingRadius * 2 )
                {
                    drawTextOutOfLine = true;
                }
                else
                {
                    gapA -= dir * textToLineSpacingRadius;
                    gapB += dir * textToLineSpacingRadius;
                }
            }
        }

        if ( useInvertedStyle )
        {
            gapA = a - dir * invertedOverhang;
            gapB = b + dir * invertedOverhang;
        }

        forEachElement( elem, [&]( Element thisElem )
        {
            if ( !useInvertedStyle && ( string.isEmpty() || drawTextOutOfLine ) )
            {
                line( thisElem, menuScaling, params, a, b, { .capA = LineCap::arrow, .capB = LineCap::arrow } );
            }
            else
            {
                line( thisElem, menuScaling, params, gapA, a, { .capB = LineCap::arrow } );
                line( thisElem, menuScaling, params, gapB, b, { .capB = LineCap::arrow } );

                if ( useInvertedStyle )
                    line( thisElem, menuScaling, params, a - dir * ( arrowLen / 2 ), b + dir * ( arrowLen / 2 ), { .flags = LineFlags::narrow } );
            }

            text( thisElem, menuScaling, params, center, string, drawTextOutOfLine ? n : ImVec2{} );
        } );
    }

    void radiusArrow( Element elem, float menuScaling, const Params& params, ImVec2 point, ImVec2 dir, float length, StringWithIcon string )
    {
        if ( ( elem & Element::both ) == Element{} )
            return; // Nothing to draw.

        float leaderLineLen = params.leaderLineLen * menuScaling;

        if ( length < 0 )
        {
            length = -length;
            dir = -dir;
        }

        dir = normalize( dir );
        ImVec2 dir2 = ImVec2( dir.x < 0 ? -1.f : 1.f, 0 );

        ImVec2 pointB = point - dir * length;
        ImVec2 pointA = pointB - dir2 * leaderLineLen;

        forEachElement( elem, [&]( Element thisElem )
        {
            line( thisElem, menuScaling, params, pointA, point, { .capB = LineCap::arrow, .midPoints = { &pointB, 1 } } );

            text( thisElem, menuScaling, params, pointA, string, -dir2 );
        } );
    }
}
