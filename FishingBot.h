#pragma once

#include "tinyfsm.hpp"
#include "GameScreen.h"
#include "Tools.h"

class GameScreen;
class AsyncJobsExecutor;

class FishingState : public tinyfsm::Fsm<FishingState>
{
public:
    FishingState();
    virtual ~FishingState();

    void react(tinyfsm::Event const&) {}

    virtual void entry();
    virtual void exit();
    virtual void tick() {}

    std::string m_name;
};

class FishingEvent : public tinyfsm::Event
{
};

class StateIdle : public FishingState
{
public:
    StateIdle();
    virtual void tick() override;
};

class WaitForFishinFloat : public FishingState
{
public:
    WaitForFishinFloat();
    virtual void entry() override;
    virtual void tick() override;

    static cv::Point s_lastFloatPos;
private:
    std::chrono::high_resolution_clock::time_point m_start;
};

class WaitForFish : public FishingState
{
public:
    WaitForFish();
    virtual void entry() override;
    virtual void tick() override;
private:
    std::chrono::high_resolution_clock::time_point m_start;
};

class CatchingFish : public FishingState
{
public:
    CatchingFish();
    virtual void entry() override;
    virtual void tick() override;
private:
    std::chrono::high_resolution_clock::time_point m_start;
};