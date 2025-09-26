#include "Tools.h"
#include "json.hpp"
#include <fstream>

std::map<std::string, double> ScopedTimer::s_averageTimes = {};

ScopedTimer::ScopedTimer(const std::string & label)
    : m_label(label), m_start(std::chrono::high_resolution_clock::now())
{
}

ScopedTimer::~ScopedTimer()
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();
    if (s_averageTimes.find(m_label) == s_averageTimes.end())
    {
        s_averageTimes[m_label] = (double)duration;
    }
    else
    {
        s_averageTimes[m_label] = (s_averageTimes[m_label] + duration) / 2.0;
    }
}

void ScopedTimer::PrintAverageTimes(int32_t const everyFrames)
{
    static int32_t callCount = 0;
    callCount++;
    if (callCount < everyFrames)
        return;

    callCount = 0;
    std::cout << "=== Performance average stats ===" << std::endl;
    for (const auto& entry : s_averageTimes)
    {
        std::cout << entry.first << ": " << entry.second << " microseconds" << std::endl;
    }
    std::cout << "=====================" << std::endl;
}


// jobs implementation

AsyncJob::AsyncJob(std::function<void(std::atomic<bool>& canceled)> job)
    : m_job(job)
{
}

AsyncJob::~AsyncJob()
{
    Wait();
}

void AsyncJob::Wait()
{
    while (!m_done && !m_cancelled)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void AsyncJob::Run()
{
    if (m_job && !m_cancelled)
    {
        this->BeginRun();
        m_job(m_cancelled);
        this->EndRun();
    }
    m_done = true;
}

AsyncJobsExecutor* AsyncJobsExecutor::s_instance = nullptr;

AsyncJobsExecutor::AsyncJobsExecutor(int32_t threadsInPool)
{
    for (int32_t i = 0; i < threadsInPool; ++i)
    {
        m_threadPool.emplace_back([this]()
        {
            while (true)
            {
                std::shared_ptr<AsyncJob> job;
                {
                    std::unique_lock<std::mutex> lock(m_queueMutex);
                    if (m_cancel)
                        break;
                    if (m_jobs.empty())
                    {
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        continue;
                    }
                    job = m_jobs.front();
                    m_jobs.pop_front();
                    m_runningJobs.push_back(job);
                }
                job->Run();
            }
        });
    }
    s_instance = this;
}

AsyncJobsExecutor::~AsyncJobsExecutor()
{
    CancelAll();
    WaitAll();
    m_cancel = true;
    for (auto& thread : m_threadPool)
    {
        if (thread.joinable())
            thread.join();
    }
}

void AsyncJobsExecutor::ExecuteJob(std::shared_ptr<AsyncJob> jobPtr)
{
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_jobs.push_back(jobPtr);
}

void AsyncJobsExecutor::WaitAll()
{
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            if (m_jobs.empty() && m_runningJobs.empty())
                break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Tick();
    }
}

void AsyncJobsExecutor::CancelAll()
{
    std::unique_lock<std::mutex> lock(m_queueMutex);
    for (auto& job : m_jobs)
    {
        job->Cancel();
    }
    for (auto& job : m_runningJobs)
    {
        job->Cancel();
    }
}

void AsyncJobsExecutor::Tick()
{
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_runningJobs.erase(std::remove_if(m_runningJobs.begin(), m_runningJobs.end(),
        [](std::shared_ptr<AsyncJob> job) { return job->IsDone() || job->IsCancelled(); }),
        m_runningJobs.end());
}

JsonConfig* JsonConfig::s_instance = nullptr;

JsonConfig::JsonConfig(const std::string& filename)
{
    using json = nlohmann::json;
    s_instance = this;

    std::ifstream fileStream(filename);
    if (!fileStream.is_open())
    {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        return;
    }
    try
    {
        json data = json::parse(fileStream);
        for (auto& [key, value] : data.items())
        {
            if (value.is_array())
            {
                std::vector<TemplateEntry> templates;
                for (const auto& item : value)
                {
                    if (item.is_object())
                    {
                        TemplateEntry entry;
                        entry.m_filePath = item["file"].get<std::string>();
                        entry.m_matchThreshold = item["threshold"].get<double>();
                        templates.push_back(entry);
                    }
                    else
                    {
                        std::cerr << "Invalid template entry format in array for key: " << key << std::endl;
                    }
                }
                m_templates[key] = templates;
            }
            else if (value.is_object())
            {
                TemplateEntry entry;
                entry.m_filePath = value["file"].get<std::string>();
                entry.m_matchThreshold = value["threshold"].get<double>();
                m_templates[key] = {entry};
            }
            else
            {
                std::cerr << "Invalid format for key: " << key << std::endl;
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing JSON config file: " << e.what() << std::endl;
    }
}

std::vector<JsonConfig::TemplateEntry> JsonConfig::GetTemplates(std::string const& key) const
{
    auto it = m_templates.find(key);
    if (it != m_templates.end())
    {
        return it->second;
    }
    return {};
}

std::vector<std::string> JsonConfig::GetAllKeys() const
{
    std::vector<std::string> keys;
    for (const auto& pair : m_templates)
    {
        keys.push_back(pair.first);
    }
    return keys;
}