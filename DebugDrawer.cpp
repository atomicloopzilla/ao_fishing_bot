#include "DebugDrawer.h"

DebugDrawer* DebugDrawer::s_instance = nullptr;

DebugDrawer::DebugDrawer()
{
    s_instance = this;
}

void DebugDrawer::DrawFrame(const cv::Mat& frame)
{
    if (frame.empty())
        return;

    // Resize display frame if necessary
    if (m_displayFrame.size() != frame.size())
    {
        m_displayFrame = cv::Mat(frame.size(), frame.type());
    }
    frame.copyTo(m_displayFrame);
}

void DebugDrawer::DrawAnnotation(std::string const& text, cv::Rect const& rect, cv::Scalar const& color)
{
    if (m_displayFrame.empty())
        return;

    // Draw rectangle
    cv::rectangle(m_displayFrame, rect, color, 2);
    // Draw text background
    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::Point textOrigin(rect.x, rect.y - 5);
    if (textOrigin.y - textSize.height < 0)
    {
        textOrigin.y = rect.y + textSize.height + 5; // Move text below the rectangle if too close to top
    }
    cv::rectangle(m_displayFrame, textOrigin + cv::Point(0, baseLine),
        textOrigin + cv::Point(textSize.width, -textSize.height),
        color, cv::FILLED);
    // Put text
    cv::putText(m_displayFrame, text, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(255, 255, 255), 1);
}

void DebugDrawer::FlushFrame()
{
    cv::imshow("Debug Frame", m_displayFrame);
    cv::waitKey(1); // Small delay to allow the image to be rendered
}

void DebugDrawer::SaveFrameToAFile(std::string const& filename)
{
    cv::imwrite(filename, m_displayFrame);
}