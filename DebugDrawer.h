#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


class DebugDrawer
{
public:
    static DebugDrawer* s_instance;

    DebugDrawer();
    ~DebugDrawer() = default;

    void DrawFrame(const cv::Mat& frame);
    void DrawAnnotation(std::string const & text, cv::Rect const & rect, cv::Scalar const & color);
    void FlushFrame();
    void SaveFrameToAFile(std::string const& filename);
private:
    cv::Mat m_displayFrame;
};