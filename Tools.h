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

#define NOMINMAX
#include <windows.h>

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
    };

    std::vector<TemplateEntry> GetTemplates(std::string const& key) const;
    std::vector<std::string> GetAllKeys() const;
private:
    std::map<std::string, std::vector<TemplateEntry> > m_templates;
};