#include "FishingBot.h"
#include "DebugDrawer.h"

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
    FishingState::tick();
    // In the idle state, we can check for water to cast the line
    cv::Point matchLoc;
    int32_t searchX = (int32_t)RandomFloat(0.0, (double)(GameScreen::s_instance->GetWidth() - 200));
    int32_t searchY = (int32_t)RandomFloat(0.0, (double)(GameScreen::s_instance->GetHeight() - 200));
    cv::Rect searchRegion(searchX, searchY, 200, 200);
    if (GameScreen::s_instance->FindTemplateInFrame("water", matchLoc, searchRegion))
    {
        std::cout << "Water found at: " << matchLoc << std::endl;
        DebugDrawer::s_instance->DrawAnnotation("Water", cv::Rect(matchLoc.x, matchLoc.y, 50, 50), cv::Scalar(255, 0, 0));
        
        AsyncJobsExecutor::s_instance->ExecuteJob(std::make_shared<AsyncJob>([matchLoc](std::atomic<bool>& cancelled)
        {
            if (cancelled)
                return;

            auto randomWaitMs = RandomFloat(200, 1000);
            GameScreen::s_instance->SendMouseDown(matchLoc.x + 5, matchLoc.y + 5);
            std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)randomWaitMs));
            GameScreen::s_instance->SendMouseUp(matchLoc.x + 5, matchLoc.y + 5);
            randomWaitMs = RandomFloat(500, 1500);
            std::cout << "Casting fishing line at: (" << (matchLoc.x + 5) << ", " << (matchLoc.y + 5) << ")" 
                << " and delay of " << (int32_t)randomWaitMs << " ms" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)randomWaitMs));
            
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

void WaitForFishinFloat::tick()
{
    FishingState::tick();
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start).count();
    if (elapsed < 3)
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
    else
    {
        std::cout << "Timeout waiting for fishing float, returning to idle." << std::endl;
        transit<StateIdle>();
    }
}

WaitForFish::WaitForFish()
{
    m_name = "WaitForFish";
}

void WaitForFish::entry()
{
    FishingState::entry();
    m_start = std::chrono::high_resolution_clock::now();
}

void WaitForFish::tick()
{
    FishingState::tick();
    // In this state, we wait for the fish to bite (float moves)
    cv::Point matchLoc;
    auto templRect = GameScreen::s_instance->GetTemplateRect("fish");
    cv::Rect floatRegion(WaitForFishinFloat::s_lastFloatPos.x - 50, WaitForFishinFloat::s_lastFloatPos.y - 50,
        templRect.width + 150, templRect.height + 150);
    cv::Rect screenRect(0, 0, GameScreen::s_instance->GetWidth(), GameScreen::s_instance->GetHeight());
    floatRegion = floatRegion & screenRect;
    DebugDrawer::s_instance->DrawAnnotation("Float Region ROI", floatRegion, cv::Scalar(50, 255, 0));
    if (GameScreen::s_instance->FindTemplateInFrame("fish", matchLoc, floatRegion))
    {
        std::cout << "Fish bite detected at: " << matchLoc << std::endl;

        AsyncJobsExecutor::s_instance->ExecuteJob(std::make_shared<AsyncJob>([matchLoc](std::atomic<bool>& cancelled)
        {
            if (cancelled)
                return;
            auto randomWaitMs = RandomFloat(50, 100);
            GameScreen::s_instance->SendMouseDown(matchLoc.x + 5, matchLoc.y + 5);
            std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)randomWaitMs));
            GameScreen::s_instance->SendMouseUp(matchLoc.x + 5, matchLoc.y + 5);
            std::cout << "Reeling in fish at: (" << (matchLoc.x + 5) << ", " << (matchLoc.y + 5) << ")" << std::endl;
            //std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)100));
        }));
        transit<CatchingFish>();
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start).count();
        if (elapsed > 30)
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
    m_start = std::chrono::high_resolution_clock::now();
}

void CatchingFish::tick()
{
    FishingState::tick();
    cv::Point matchLoc;
    cv::Rect centerRegion(
        (GameScreen::s_instance->GetWidth() / 2) - 100,
        (GameScreen::s_instance->GetHeight() / 2) - 100, 200, 200);

    DebugDrawer::s_instance->DrawAnnotation("Float Region ROI", centerRegion, cv::Scalar(50, 255, 0));

    bool floatLow = GameScreen::s_instance->FindTemplateInFrame("float_low", matchLoc, centerRegion);
    bool floatMedium = GameScreen::s_instance->FindTemplateInFrame("float_medium", matchLoc, centerRegion);
    bool floatHigh = GameScreen::s_instance->FindTemplateInFrame("float_high", matchLoc, centerRegion);
    if (floatLow || floatMedium || floatHigh)
    {
        if (floatHigh)
        {
            GameScreen::s_instance->SendMouseUp(WaitForFishinFloat::s_lastFloatPos.x, WaitForFishinFloat::s_lastFloatPos.y);
            std::cout << "Unholding fish..." << std::endl;
        }
        else
        {
            GameScreen::s_instance->SendMouseDown(WaitForFishinFloat::s_lastFloatPos.x, WaitForFishinFloat::s_lastFloatPos.y);
            std::cout << "Holding fish..." << std::endl;
        }
        m_start = std::chrono::high_resolution_clock::now();
    }
    else
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start).count();
        GameScreen::s_instance->SendMouseUp(WaitForFishinFloat::s_lastFloatPos.x, WaitForFishinFloat::s_lastFloatPos.y);
        if (elapsed > 1000)
        {
            std::cout << "Fish caught or lost, returning to idle." << std::endl;
            transit<StateIdle>();
        }
        else
        {
            std::cout << "Waiting to confirm fish catch..." << std::endl;
        }
    }

}


FSM_INITIAL_STATE(FishingState, StateIdle)