#include "MRUITestEngine.h"
#include "MRPch/MRSpdlog.h"

#include <imgui.h>

#ifndef MR_ENABLE_UI_TEST_ENGINE
// Set to 0 to disable the UI test engine. All functions will act as if no UI elements are registered.
#define MR_ENABLE_UI_TEST_ENGINE 1
#endif

namespace MR::UI::TestEngine
{

namespace
{

struct State
{
    // The root element group.
    GroupEntry root;

    // The frame counter, as per `ImGui::GetFrameCount()`. When this changes, we prune dead elements.
    int curFrame = -1;

    // The stack for `pushTree()`, `popTree()`. Always has at least one element.
    std::vector<GroupEntry*> stack = { &root };
};
// Our global state. Stores the current tree of buttons and button groups.
State state;

void checkForNewFrame()
{
    // If this is a new frame...
    if ( ImGui::GetFrameCount() != state.curFrame )
    {
        state.curFrame = ImGui::GetFrameCount();

        // Make sure the tree stack is fine.
        assert(state.stack.size() == 1 && "Missing `UI::TestEngine::popTree()`.");

        // Prune stale elements from the tree.
        auto pruneDeadEntries = [&]( auto& pruneDeadEntries, GroupEntry& group ) -> void
        {
            for ( auto it = group.elems.begin(); it != group.elems.end(); )
            {
                if ( !it->second.visitedOnThisFrame )
                {
                    it = group.elems.erase( it );
                }
                else
                {
                    it->second.visitedOnThisFrame = false;

                    if ( auto subgroup = std::get_if<GroupEntry>( &it->second.value ) )
                        pruneDeadEntries( pruneDeadEntries, *subgroup );

                    ++it;
                    continue;
                }
            }
        };
        pruneDeadEntries( pruneDeadEntries, state.root );
    }
}

} // namespace

template <typename T>
std::optional<T> detail::createValueLow( std::string_view name, std::optional<BoundedValue<T>> value )
{
    #if MR_ENABLE_UI_TEST_ENGINE

    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
    {
        iter = map.try_emplace( std::string( name ) ).first;
    }
    else
    {
        // If you see this assert, you likely have duplicate drag/slider names.
        assert( !iter->second.visitedOnThisFrame && "Registering the same entry more than once in a single frame!" );
    }

    ValueEntry* entry = std::get_if<ValueEntry>( &iter->second.value );
    if ( !entry )
        entry = &iter->second.value.emplace<ValueEntry>();

    std::optional<T> ret;

    ValueEntry::Value<T>* val = std::get_if<ValueEntry::Value<T>>( &entry->value );
    if ( val )
    {
        ret = std::move( val->simulatedValue );
        val->simulatedValue = {};
    }
    else
    {
        val = &entry->value.emplace<ValueEntry::Value<T>>();
    }

    if ( value )
    {
        iter->second.visitedOnThisFrame = true;
        val->value = std::move( value->value ); // Could also read `ret` here, but that would be a bit weird, I guess?
        if constexpr ( std::is_same_v<T, std::string> )
        {
            val->allowedValues = value->allowedValues;
        }
        else
        {
            val->min = value->min;
            val->max = value->max;
        }
    }

    return ret;

    #else
    return {};
    #endif
}

template std::optional<std::int64_t> detail::createValueLow( std::string_view name, std::optional<BoundedValue<std::int64_t>> value );
template std::optional<std::uint64_t> detail::createValueLow( std::string_view name, std::optional<BoundedValue<std::uint64_t>> value );
template std::optional<double> detail::createValueLow( std::string_view name, std::optional<BoundedValue<double>> value );
template std::optional<std::string> detail::createValueLow( std::string_view name, std::optional<BoundedValue<std::string>> value );

bool createButton( std::string_view name )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
    {
        iter = map.try_emplace( std::string( name ) ).first;
    }
    else
    {
        // If you see this assert, you likely have duplicate button names.
        // If you truly have several buttons with the same name in one place, add unique `##...` suffixes to them.
        // If the buttons are in different parts of the application and shouldn't collide,
        // use `UI::TestEngine::pushTree("...")` and `popTree()` to group them into named groups, with unique names in each group.
        assert( !iter->second.visitedOnThisFrame && "Registering the same entry more than once in a single frame!" );
    }

    ButtonEntry* button = std::get_if<ButtonEntry>( &iter->second.value );
    if ( !button )
        button = &iter->second.value.emplace<ButtonEntry>();

    iter->second.visitedOnThisFrame = true;

    if ( button->simulateClick )
        spdlog::info( "Button {} click simulation", name );

    return std::exchange( button->simulateClick, false );
    #else
    return false;
    #endif
}

std::optional<std::string> createValue( std::string_view name, std::string value, std::optional<std::vector<std::string>> allowedValues )
{
    return detail::createValueLow<std::string>( name, detail::BoundedValue<std::string>{ .value = std::move( value ), .allowedValues = std::move( allowedValues ) } );
}

void pushTree( std::string_view name )
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    auto& map = state.stack.back()->elems;
    auto iter = map.find( name ); // I wish I could use `std::try_emplace` here directly...
    if ( iter == map.end() )
        iter = map.try_emplace( std::string( name ) ).first;
    else
        assert( !iter->second.visitedOnThisFrame && "Registering the same entry more than once in a single frame!" );

    GroupEntry* subgroup = std::get_if<GroupEntry>( &iter->second.value );
    if ( !subgroup )
        subgroup = &iter->second.value.emplace<GroupEntry>();

    iter->second.visitedOnThisFrame = true;

    state.stack.push_back( subgroup );
    #endif
}

void popTree()
{
    #if MR_ENABLE_UI_TEST_ENGINE
    checkForNewFrame();

    assert( state.stack.size() > 1 && "Excessive `UI::TestEngine::popTree()`." );
    if ( state.stack.size() <= 1 )
        return;

    state.stack.pop_back();
    #endif
}

std::string_view Entry::getKindName() const
{
    return std::visit( []<typename T>( const T & ){ return T::kindName; }, value );
}

const GroupEntry& getRootEntry()
{
    return state.root;
}

}
