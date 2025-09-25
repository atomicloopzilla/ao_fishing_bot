#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#define NOMINMAX
#include <windows.h>

#include <map>


class GameScreen
{
public:
    static GameScreen* s_instance;

    GameScreen(HWND hwnd);
    ~GameScreen();

    void CaptureScreen();

    int32_t GetWidth() const { return m_width; }
    int32_t GetHeight() const { return m_height; }

    cv::Mat const & GetFrame() const { return m_frame; }
    cv::cuda::GpuMat const & GetGpuFrame() { return m_gpuFrame; }

    bool FindTemplateInFrame(std::string const& templateKey, cv::Point& matchLoc, cv::Rect searchRegion = cv::Rect());
    cv::Rect GetTemplateRect(std::string const& templateKey);
    void PrintMaxValues(int32_t const everyFrames = 1000);

    void LoadTemplates();

    // input handling
    void SendMouseDown(int32_t x, int32_t y);
    void SendMouseUp(int32_t x, int32_t y);
private:
    HWND            m_hwnd;
    HDC             m_hdc;
    HDC             m_memDC;
    HBITMAP         m_hBitmap;
    BITMAPINFO      m_bmi;
    int32_t         m_width;
    int32_t         m_height;
    int32_t         m_left;
    int32_t         m_top;

    cv::Mat             m_frame;
    cv::cuda::GpuMat    m_gpuFrame;

    cv::Ptr<cv::cuda::TemplateMatching> m_matcher;
    bool m_matcherInitialized = false;

    struct Template
    {
        cv::cuda::GpuMat image;
        double threshold;
        cv::Rect imageSize;
    };

    std::map<std::string, std::vector<Template> > m_templates;
    std::map<std::string, double> m_maxMatchValues;
};