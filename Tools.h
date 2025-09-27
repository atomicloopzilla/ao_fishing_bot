#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <iostream>
#include <thread>
#include <functional>
#include <mutex>
#include <random>
#include <iostream>
#include <format>

#define NOMINMAX
#include <windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#if HAVE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#elif HAVE_CL
#include <opencv2/core/ocl.hpp>
#endif

class ScopedTimer
{
public:
    ScopedTimer(const std::string& label = "");
    ~ScopedTimer();

    static void PrintAverageTimes(int32_t const everyFrames = 1000);

private:
    std::string m_label;
    std::chrono::high_resolution_clock::time_point m_start;

    static std::map<std::string, double> s_averageTimes;
};

class AsyncJob
{
public:
    friend class AsyncJobsExecutor;

    explicit AsyncJob(std::function<void(std::atomic<bool>& canceled)> job);

    AsyncJob(AsyncJob const&) = delete;
    AsyncJob& operator=(AsyncJob const&) = delete;
    AsyncJob(AsyncJob&&) = delete;
    AsyncJob& operator=(AsyncJob&&) = delete;

    virtual ~AsyncJob();

    void Wait();

    bool IsDone() const { return m_done; }
    bool IsCancelled() const { return m_cancelled; }
    void Cancel() { m_cancelled = true; }
    
    virtual void BeginRun() {}
    virtual void EndRun() {}
private:
    void Run();

    std::function<void(std::atomic<bool>& canceled)>    m_job;
    std::atomic<bool>                                   m_done {false};
    std::atomic<bool>                                   m_cancelled {false};
};

class AsyncJobsExecutor
{
public:
    static AsyncJobsExecutor* s_instance;

    AsyncJobsExecutor(int32_t threadsInPool);
    ~AsyncJobsExecutor();

    void ExecuteJob(std::shared_ptr<AsyncJob> jobPtr);
    void WaitAll();
    void CancelAll();
    void Tick();
private:
    std::vector<std::thread> m_threadPool;
    std::deque<std::shared_ptr<AsyncJob>> m_jobs;
    std::vector<std::shared_ptr<AsyncJob>> m_runningJobs;
    std::atomic<bool> m_cancel{false};
    std::mutex m_queueMutex;
};

static double RandomFloat(double min, double max)
{
    static std::mutex mtx;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::lock_guard<std::mutex> lock(mtx);
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

class JsonConfig
{
public:
    static JsonConfig* s_instance;
    
    JsonConfig(const std::string& filename);
    ~JsonConfig() = default;

    struct TemplateEntry
    {
        std::string m_filePath;
        double      m_matchThreshold;
        int32_t     m_offset = 0;
    };

    std::vector<TemplateEntry> GetTemplates(std::string const& key) const;
    std::vector<std::string> GetAllKeys() const;
private:
    std::map<std::string, std::vector<TemplateEntry> > m_templates;
};

class FramesStack
{
public:
    static FramesStack* s_instance;

    FramesStack(int32_t maxFrames);
    ~FramesStack() = default;
    
    void AddFrame(cv::Mat const& frame);
    void SaveFrame(int32_t index, std::string const& fileName);

    cv::Mat const& GetFrame(int32_t index) const { return m_frames[index]; }
    int32_t GetFramesCount() const { return static_cast<int32_t>(m_frames.size()); }

private:
    std::deque<cv::Mat> m_frames;
    int32_t m_maxFrames;
};