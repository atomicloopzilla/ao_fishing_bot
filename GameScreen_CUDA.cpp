#include "GameScreen.h"
#include "Tools.h"
#include "DebugDrawer.h"

#if HAVE_CUDA

bool GameScreen::FindTemplateInFrame(std::string const& templateName, cv::Point& matchLoc, cv::Rect searchRegion)
{
    ScopedTimer timer("GameScreen::FindTemplateInFrame:" + templateName);
    if (m_frame.empty())
        return 0.0;

    if (m_templates.find(templateName) == m_templates.end())
    {
        throw std::runtime_error("Template not found: " + templateName);
    }

    auto templ = m_templates[templateName];
    for (auto const& t : templ)
    {
        cv::cuda::GpuMat gpuTempl = t.image;
        cv::cuda::GpuMat gpuSearchRegion;
        if (searchRegion.area() > 0)
        {
            cv::Rect validRegion = searchRegion & cv::Rect(0, 0, m_frame.cols, m_frame.rows);
            if (validRegion.area() <= 0)
            {
                throw std::runtime_error("Search region is out of bounds.");
            }
            gpuSearchRegion = m_gpuFrame(validRegion);
        }
        else
        {
            gpuSearchRegion = m_gpuFrame;
        }
        int resultCols = gpuSearchRegion.cols;// - gpuTempl.cols + 1;
        int resultRows = gpuSearchRegion.rows;// - gpuTempl.rows + 1;
        if (resultCols <= 0 || resultRows <= 0)
        {
            throw std::runtime_error("Template is larger than search region.");
        }
        cv::cuda::GpuMat gpuResult;
        gpuResult.create(resultRows, resultCols, CV_32FC1);
        if (!m_matcherInitialized)
        {
            m_matcher = cv::cuda::createTemplateMatching(gpuSearchRegion.type(), cv::TM_CCOEFF_NORMED);
            m_matcherInitialized = true;
        }

        m_matcher->match(gpuSearchRegion, gpuTempl, gpuResult);
        
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        
        cv::cuda::minMaxLoc(gpuResult, &minVal, &maxVal, &minLoc, &maxLoc);
        double maxSavedValue = m_maxMatchValues[templateName];
        if (maxVal > maxSavedValue)
        {
            m_maxMatchValues[templateName] = maxVal;
            std::cout << "New max match value for template (" << templateName << ") : " << maxVal << std::endl;
        }
        double averageValue = m_averageValues[templateName];
        averageValue = (averageValue + maxVal) / 2;
        m_averageValues[templateName] = averageValue;

        //std::cout << "Template (" << templateName << ") match max value : " << maxVal << " (threshold : " << t.threshold << ")" << std::endl;
        if (maxVal >= t.threshold)
        {
            std::cout << "Template hit: " << templateName << " with value: " << maxVal << " (threshold: " << t.threshold << ")" << std::endl;

            matchLoc = maxLoc;
            if (searchRegion.area() > 0)
            {
                matchLoc.x += searchRegion.x;
                matchLoc.y += searchRegion.y;
            }
            return true;
        }
    }
    return false;
}

#endif