#pragma once
#include "MRFrameRedrawRequest.h"
#include <imgui.h>
#include <functional>
#include <atomic>
#include <thread>

namespace MR
{

// This class shows application progress bar for long operations
// note! if class don't setup, then order and orderWithMainThreadPostProcessing methods call task directly
class ProgressBar
{
public:
    using TaskWithMainThreadPostProcessing = std::function< std::function<void()>() >;
    // this function should be called only once for each frame (it is called in MR::Menu (MR::RibbonMenu))
    MRVIEWER_API static void setup( float scaling );

    // this shall be called in order to start concurrent task execution with progress bar display
    MRVIEWER_API static void order(const char * name, const std::function<void()>& task, int taskCount = 1 );

    // in this version the task returns a function to be executed in main thread
    MRVIEWER_API static void orderWithMainThreadPostProcessing( const char* name, TaskWithMainThreadPostProcessing task, int taskCount = 1 );

    /// primitive coroutine-like task interface
    template <typename T>
    class ResumableTask
    {
    public:
        using result_type = T;

        virtual ~ResumableTask() = default;
        /// start the task
        virtual void start() = 0;
        /// resume the task, return true if the task is finished, false if it should be re-invoked later
        virtual bool resume() = 0;
        /// get the result
        virtual result_type result() const = 0;
    };
    /// in this version the task is being run in the main thread but performs as a coroutine (suspends its execution from time to time)
    MRVIEWER_API static void orderWithResumableTask( const char * name, std::shared_ptr<ResumableTask<void>> task, int taskCount = 1 );

    MRVIEWER_API static bool isCanceled();

    MRVIEWER_API static bool isFinished();

    MRVIEWER_API static float getProgress();

    // sets the current progress and returns false if the user has pressed Cancel button
    MRVIEWER_API static bool setProgress(float p);

    MRVIEWER_API static void nextTask();
    MRVIEWER_API static void nextTask(const char * s);

    MRVIEWER_API static void setTaskCount( int n );

    // returns true if progress bar was ordered and not finished
    MRVIEWER_API static bool isOrdered();

    // these callbacks allow canceling
    MRVIEWER_API static bool callBackSetProgress(float p);
    // these callbacks do not allow canceling
    MRVIEWER_API static bool simpleCallBackSetProgress( float p );
private:
    static ProgressBar& instance_();

    ProgressBar();
    ~ProgressBar();

    void initialize_();

    // cover task execution with try catch block (in release only)
    // if catches exception shows error in main thread overriding user defined main thread post-processing
    bool tryRun_( const std::function<bool ()>& task );
    bool tryRunWithSehHandler_( const std::function<bool ()>& task );

    void resumeBackgroundTask_();

    void finish_();

    float progress_;
    int currentTask_, taskCount_;
    std::string taskName_, title_;

    FrameRedrawRequest frameRequest_;

    // parameter is needed for logging progress
    std::atomic<int> percents_;

    std::thread thread_;
    std::function<void()> onFinish_;

    // needed to be able to call progress bar from any point, not only from ImGui frame scope
    struct DeferredInit
    {
        int taskCount;
        std::string name;
        std::function<void ()> postInit;
    };
    std::unique_ptr<DeferredInit> deferredInit_;

    // required to perform long-time tasks in single-threaded environments
    using BackgroundTask = std::shared_ptr<ResumableTask<void>>;
    BackgroundTask backgroundTask_;

    std::atomic<bool> allowCancel_;
    std::atomic<bool> canceled_;
    std::atomic<bool> finished_;
    ImGuiID setupId_ = ImGuiID( -1 );

    bool isOrdered_{ false };
    bool isInit_{ false };
    // this is needed to show full progress before closing
    bool closeDialogNextFrame_{ false };
};

}