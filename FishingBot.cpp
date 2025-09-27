#include "FishingBot.h"
#include "DebugDrawer.h"
#include "FishingClassifier.h"
#include <filesystem>

FishingState::FishingState()
    : m_name("FishingState")
{
}

FishingState::~FishingState()
{
}

void FishingState::entry()
{
    std::cout << "Entering state: " << m_name << std::endl;
    // Here you can add code to handle the fishing state, e.g., cast the fishing line
}

void FishingState::exit()
{
    AsyncJobsExecutor::s_instance->WaitAll();
    std::cout << "Exiting state: " << m_name << std::endl;
}

StateIdle::StateIdle()
{
    m_name = "StateIdle";
}

void StateIdle::tick()
{
    static double s_waterBounds = 300.0;
    static double s_maxRobTimeMs = 1000.0;
    static double s_maxRobDist = 500.0;

    FishingState::tick();
    // In the idle state, we can check for water to cast the line
    cv::Point matchLoc;
    
    int32_t searchX = (int32_t)RandomFloat(s_waterBounds, (double)(GameScreen::s_instance->GetWidth()) - s_waterBounds);
    int32_t searchY = (int32_t)RandomFloat(s_waterBounds, (double)(GameScreen::s_instance->GetHeight()) - s_waterBounds);
    cv::Rect searchRegion(searchX, searchY, 200, 200);
    
    if (WaitForFish::s_annotationMode)
    {
        transit<WaitForFishinFloat>();
        return;
    }

    if (GameScreen::s_instance->FindTemplateInFrame("water", matchLoc, searchRegion))
    {
        std::cout << "Water found at: " << matchLoc << std::endl;
        DebugDrawer::s_instance->DrawAnnotation("Water", cv::Rect(matchLoc.x, matchLoc.y, 50, 50), cv::Scalar(255, 0, 0));

        double centerX = GameScreen::s_instance->GetWidth() / 2.0;
        double centerY = GameScreen::s_instance->GetHeight() / 2.0;
        double dist = std::sqrt((matchLoc.x - centerX) * (matchLoc.x - centerX) + (matchLoc.y - centerY) * (matchLoc.y - centerY));
        double timeToRobMs = (dist / s_maxRobDist) * s_maxRobTimeMs;
        
        AsyncJobsExecutor::s_instance->ExecuteJob(std::make_shared<AsyncJob>([matchLoc, timeToRobMs](std::atomic<bool>& cancelled)
        {
            if (cancelled)
                return;

            auto randomWaitMs = RandomFloat(200, 1000);
            GameScreen::s_instance->SendMouseDown(matchLoc.x + 5, matchLoc.y + 5);
            std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)timeToRobMs));
            GameScreen::s_instance->SendMouseUp(matchLoc.x + 5, matchLoc.y + 5);
            std::cout << "Casting fishing line at: (" << (matchLoc.x + 5) << ", " << (matchLoc.y + 5) << ")" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)2000));
            
        }));
        transit<WaitForFishinFloat>();
    }

}


WaitForFishinFloat::WaitForFishinFloat()
{
    m_name = "WaitForFishinFloat";
}

void WaitForFishinFloat::entry()
{
    FishingState::entry();
    m_start = std::chrono::high_resolution_clock::now();
}

cv::Point WaitForFishinFloat::s_lastFloatPos = cv::Point(500, 500);
int64_t WaitForFishinFloat::s_maxWaitFloatSeconds = 3;

void WaitForFishinFloat::tick()
{
    FishingState::tick();
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start).count();
    if (elapsed < 1)
    {
        return;
    }

    // In this state, we wait for the fishing float to appear
    cv::Point matchLoc;
    if (GameScreen::s_instance->FindTemplateInFrame("fishing_float", matchLoc, cv::Rect()))
    {
        std::cout << "Fishing float found at: " << matchLoc << std::endl;
        s_lastFloatPos = matchLoc;
        transit<WaitForFish>();
    }
    else if (elapsed > s_maxWaitFloatSeconds)
    {
        std::cout << "Timeout waiting for fishing float, returning to idle." << std::endl;
        transit<StateIdle>();
    }
}

WaitForFish::WaitForFish()
{
    m_name = "WaitForFish";
    m_classifier = std::make_unique<FishingClassifier>("fishing_classifier_traced.pt");
}

bool WaitForFish::s_annotationMode = false;
int64_t WaitForFish::s_maxWaitFishSeconds = 60;
float WaitForFish::s_predictionThreshold = 0.8f;

void WaitForFish::entry()
{
    FishingState::entry();
    m_start = std::chrono::high_resolution_clock::now();
}

void WaitForFish::CaughtFish(cv::Point const& matchLoc, int32_t offset)
{
    std::cout << "Caught fish at: " << matchLoc << std::endl;
    if (s_annotationMode)
    {
        AnnotateFrames(offset);
        transit<StateIdle>();
        return;
    }

    AsyncJobsExecutor::s_instance->ExecuteJob(std::make_shared<AsyncJob>([matchLoc](std::atomic<bool>& cancelled)
    {
        if (cancelled)
            return;

        auto randomWaitMs = RandomFloat(100, 500);
        std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)randomWaitMs));
        randomWaitMs = RandomFloat(50, 100);
        GameScreen::s_instance->SendMouseDown(matchLoc.x + 5, matchLoc.y + 5);
        std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)randomWaitMs));
        GameScreen::s_instance->SendMouseUp(matchLoc.x + 5, matchLoc.y + 5);
        std::cout << "Reeling in fish at: (" << (matchLoc.x + 5) << ", " << (matchLoc.y + 5) << ")" << std::endl;
    }));
    transit<CatchingFish>();
}

static int32_t GetLastSavedCountFromFile()
{
    int32_t maxCount = 0;
    std::string path = "out/annotated/";
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_regular_file())
        {
            std::string filename = entry.path().filename().string();
            size_t underscorePos = filename.find('_');
            if (underscorePos != std::string::npos)
            {
                try
                {
                    int32_t count = std::stoi(filename.substr(0, underscorePos));
                    if (count > maxCount)
                    {
                        maxCount = count;
                    }
                }
                catch (const std::invalid_argument&)
                {
                    // Not a number, ignore
                }
            }
        }
    }
    return maxCount;
}

void WaitForFish::AnnotateFrames(int32_t offset)
{
    static int32_t s_annotatedCount = GetLastSavedCountFromFile();
    static int32_t s_animationFramesCount = 5;

    s_annotatedCount++;
    std::cout << "-------------------- Annotation Mode --------------------" << std::endl;
    std::cout << "Annotating frame: " << s_annotatedCount << std::endl;

    int32_t framesCount = FramesStack::s_instance->GetFramesCount();
    if (framesCount < (s_animationFramesCount * 2))
    {
        std::cout << "Not enough frames to annotate, need at least " << (s_animationFramesCount * 2) << " frames." << std::endl;
        return;
    }
    
    int32_t startFrameBite = std::max(framesCount - offset - 1, 0);
    int32_t startFramsNotBite = std::max(startFrameBite - s_animationFramesCount - 1, 0);

    for (int32_t i = 0; i < s_animationFramesCount; ++i)
    {
        std::ostringstream fileName;
        fileName << "out/annotated/" << s_annotatedCount << "_nobite_" << i << ".png";
        FramesStack::s_instance->SaveFrame(startFramsNotBite + i, fileName.str());
        std::cout << "Saved not bite frame: " << fileName.str() << std::endl;
    }

    startFrameBite = framesCount - s_animationFramesCount - 1;
    for (int32_t i = 0; i < std::min(offset, s_animationFramesCount); ++i)
    {
        std::ostringstream fileName;
        fileName << "out/annotated/" << s_annotatedCount << "_bite_" << i << ".png";
        FramesStack::s_instance->SaveFrame(startFrameBite + i, fileName.str());
        std::cout << "Saved bite frame: " << fileName.str() << std::endl;
    }
    std::cout << "---------------------------------------------------------" << std::endl;
}


cv::Rect WaitForFish::GetFloatRegion()
{
    cv::Rect floatRegion(WaitForFishinFloat::s_lastFloatPos.x - 35, WaitForFishinFloat::s_lastFloatPos.y - 35,
        100, 100);
    cv::Rect screenRect(0, 0, GameScreen::s_instance->GetWidth(), GameScreen::s_instance->GetHeight());
    floatRegion = floatRegion & screenRect;
    return floatRegion;
}

void WaitForFish::tick()
{
    FishingState::tick();
    

    // In this state, we wait for the fish to bite (float moves)
    cv::Point matchLoc;
    cv::Rect floatRegion = GetFloatRegion();
    
    cv::Mat frame = GameScreen::s_instance->GetFrame()(floatRegion);
    if (s_annotationMode)
    {
        FramesStack::s_instance->AddFrame(frame);
    }

    float prediction = m_classifier->PredictBite(frame);

    DebugDrawer::s_instance->DrawAnnotation("Float Region ROI", floatRegion, cv::Scalar(50, 255, 0));


    /*if (GameScreen::s_instance->FindTemplateInFrame("fish1", matchLoc, floatRegion))
    {
        std::cout << "Fish1 bite detected at: " << matchLoc << std::endl;
        auto offset = GameScreen::s_instance->GetTemplateOffset("fish1");
        CaughtFish(matchLoc, offset);
    }
    else if (GameScreen::s_instance->FindTemplateInFrame("fish2", matchLoc, floatRegion))
    {
        std::cout << "Fish2 bite detected at: " << matchLoc << std::endl;
        auto offset = GameScreen::s_instance->GetTemplateOffset("fish2");
        CaughtFish(matchLoc, offset);
    }*/
    if (prediction > s_predictionThreshold)
    {
        std::cout << "Fish bite detected with prediction: " << prediction << std::endl;
        CaughtFish(matchLoc, 3);
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start).count();
        if (elapsed > s_maxWaitFishSeconds)
        {
            std::cout << "Timeout waiting for fish bite, returning to idle." << std::endl;
            transit<StateIdle>();
        }
    }
}

CatchingFish::CatchingFish()
{
    m_name = "CatchingFish";
}

void CatchingFish::entry()
{
    FishingState::entry();
    std::cout << "Starting to catch fish, holding mouse down at float position." << std::endl;
    GameScreen::s_instance->SendMouseDown(WaitForFishinFloat::s_lastFloatPos.x, WaitForFishinFloat::s_lastFloatPos.y);
    m_mouseDown = true;
    m_start = std::chrono::high_resolution_clock::now();
}

void CatchingFish::tick()
{
    static int64_t s_maxHoldMouseMs = 1000;
    FishingState::tick();
    cv::Point matchLoc;
    cv::Rect centerRegion(
        (GameScreen::s_instance->GetWidth() / 2) - 100,
        (GameScreen::s_instance->GetHeight() / 2) - 100, 200, 200);

    int32_t centerX = (GameScreen::s_instance->GetWidth() / 2);
    int32_t centerY = (GameScreen::s_instance->GetHeight() / 2);

    DebugDrawer::s_instance->DrawAnnotation("Float Region ROI", centerRegion, cv::Scalar(50, 255, 0));

    bool floatMedium = GameScreen::s_instance->FindTemplateInFrame("float_medium", matchLoc, centerRegion);
    bool floatHigh = GameScreen::s_instance->FindTemplateInFrame("float_high", matchLoc, centerRegion);
    if (floatMedium || floatHigh)
    {
        if (floatHigh && m_mouseDown)
        {
            GameScreen::s_instance->SendMouseUp(WaitForFishinFloat::s_lastFloatPos.x, WaitForFishinFloat::s_lastFloatPos.y);
            m_mouseDown = false;
            std::cout << "Unholding fish..." << std::endl;
        }
        else if (!m_mouseDown)
        {
            GameScreen::s_instance->SendMouseDown(WaitForFishinFloat::s_lastFloatPos.x, WaitForFishinFloat::s_lastFloatPos.y);
            m_mouseDown = true;
            std::cout << "Holding fish..." << std::endl;
        }
        m_start = std::chrono::high_resolution_clock::now();
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start).count();
        if (elapsed > s_maxHoldMouseMs)
        {
            std::cout << "Fish caught or lost, returning to idle." << std::endl;
            if (m_mouseDown)
            {
                GameScreen::s_instance->SendMouseUp(centerX, centerY);
            }
            transit<StateIdle>();
        }
    }
}


FSM_INITIAL_STATE(FishingState, StateIdle)