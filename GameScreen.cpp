#include "GameScreen.h"
#include "Tools.h"

GameScreen* GameScreen::s_instance = nullptr;

GameScreen::GameScreen(HWND hwnd)
    : m_hwnd(hwnd)
{
    // Get window dimensions
    RECT windowRect;
    GetClientRect(m_hwnd, &windowRect);
    m_width = windowRect.right - windowRect.left;
    m_height = windowRect.bottom - windowRect.top;
    // Get desktop device context
    m_hdc = GetDC(nullptr);
    m_memDC = CreateCompatibleDC(m_hdc);
    
    // Create bitmap
    m_hBitmap = CreateCompatibleBitmap(m_hdc, m_width, m_height);
    SelectObject(m_memDC, m_hBitmap);
    // Initialize BITMAPINFO
    ZeroMemory(&m_bmi, sizeof(BITMAPINFO));
    m_bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    m_bmi.bmiHeader.biWidth = m_width;
    m_bmi.bmiHeader.biHeight = -m_height; // Top-down
    m_bmi.bmiHeader.biPlanes = 1;
    m_bmi.bmiHeader.biBitCount = 24; // 3 channels (BGR)
    m_bmi.bmiHeader.biCompression = BI_RGB;
    // Initialize frame Mat
    m_frame = cv::Mat(m_height, m_width, CV_8UC3);
    s_instance = this;
}

GameScreen::~GameScreen()
{
    DeleteObject(m_hBitmap);
    DeleteDC(m_memDC);
    ReleaseDC(nullptr, m_hdc);
}

void GameScreen::CaptureScreen()
{
    ScopedTimer timer("GameScreen::CaptureScreen");
    // Get current window position
    RECT windowRect;
    GetWindowRect(m_hwnd, &windowRect);
    m_left = windowRect.left;
    m_top = windowRect.top;
    // Capture the screen region where the window is located
    BitBlt(m_memDC, 0, 0, m_width, m_height, m_hdc, m_left, m_top, SRCCOPY | CAPTUREBLT);
    GdiFlush();
    GetDIBits(m_memDC, m_hBitmap, 0, m_height, m_frame.data, &m_bmi, DIB_RGB_COLORS);

    /*// Convert frame to grayscale
    cv::Mat grayscaleFrame;
    if (m_frame.channels() == 3) {
        cv::cvtColor(m_frame, grayscaleFrame, cv::COLOR_BGR2GRAY);
        m_frame = grayscaleFrame;
    } else if (m_frame.channels() == 1) {
        // Already grayscale, no conversion needed
    } else {
        std::cerr << "Unexpected number of channels: " << m_frame.channels() << std::endl;
    }*/
#if HAVE_CUDA
    m_gpuFrame.upload(m_frame);
#else
    m_frame.copyTo(m_gpuFrame);
#endif
    auto end = std::chrono::high_resolution_clock::now();
}

void GameScreen::LoadTemplates()
{
    for (auto const & key : JsonConfig::s_instance->GetAllKeys())
    {
        auto const& entries = JsonConfig::s_instance->GetTemplates(key);
        std::cout << "Loading templates for key: " << key << " with " << entries.size() << " entries." << std::endl;
        m_templates[key] = std::vector<Template>();
        for (auto const& entry : entries)
        {
            cv::Mat templ = cv::imread(entry.m_filePath);
            if (templ.empty())
            {
                std::cerr << "Failed to load template image: " << entry.m_filePath << std::endl;
                throw std::runtime_error("Failed to load template image: " + entry.m_filePath);
            }
            /*// Convert template to grayscale
            cv::Mat grayscaleTempl;
            if (templ.channels() == 3) {
                cv::cvtColor(templ, grayscaleTempl, cv::COLOR_BGR2GRAY);
            } else if (templ.channels() == 1) {
                grayscaleTempl = templ.clone();
            } else {
                std::cerr << "Unexpected template channels: " << templ.channels() << " for " << entry.m_filePath << std::endl;
                continue;
            }*/
            std::cout << "Uploading template to GPU: " << entry.m_filePath << std::endl;
            Template temp;
#if HAVE_CUDA
            cv::cuda::GpuMat gpuTempl;
            gpuTempl.upload(templ);
            temp.image = gpuTempl;
#elif HAVE_CL
            cv::UMat gpuTempl;
            templ.copyTo(gpuTempl);
            temp.image = gpuTempl;
#else
            temp.image = templ;
#endif
            temp.threshold = entry.m_matchThreshold;
            temp.imageSize = cv::Rect(0, 0, templ.cols, templ.rows);
            m_templates[key].push_back(temp);
        }
    }
}

cv::Rect GameScreen::GetTemplateRect(std::string const& templateName)
{
    if (m_templates.find(templateName) == m_templates.end())
    {
        throw std::runtime_error("Template not found: " + templateName);
    }

    cv::Rect maxRect;
    auto templ = m_templates[templateName];
    for (auto const& t : templ)
    {
        if (t.imageSize.area() > maxRect.area())
        {
            maxRect = t.imageSize;
        }
    }
    return maxRect;
}

void GameScreen::PrintMaxValues(int32_t const everyFrames)
{
    static int32_t callCount = 0;
    callCount++;
    if (callCount < everyFrames)
        return;
    callCount = 0;
    std::cout << "=== Template max match values ===" << std::endl;
    for (const auto& entry : m_maxMatchValues)
    {
        std::cout << entry.first << ": " << entry.second << std::endl;
    }
    std::cout << "=====================" << std::endl;
}

void GameScreen::SendMouseDown(int32_t x, int32_t y)
{
    // Convert to screen coordinates
    int screenX = m_left + x;
    int screenY = m_top + y;
    // Set cursor position
    SetCursorPos(screenX, screenY);
    // Prepare mouse down event
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dx = (screenX * 65536) / GetSystemMetrics(SM_CXSCREEN);
    input.mi.dy = (screenY * 65536) / GetSystemMetrics(SM_CYSCREEN);
    input.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN;
    // Send mouse down event
    SendInput(1, &input, sizeof(INPUT));

}

void GameScreen::SendMouseUp(int32_t x, int32_t y)
{
    // Convert to screen coordinates
    int screenX = m_left + x;
    int screenY = m_top + y;
    // Set cursor position
    SetCursorPos(screenX, screenY);
    // Prepare mouse up event
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dx = (screenX * 65536) / GetSystemMetrics(SM_CXSCREEN);
    input.mi.dy = (screenY * 65536) / GetSystemMetrics(SM_CYSCREEN);
    input.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP;
    // Send mouse up event
    SendInput(1, &input, sizeof(INPUT));
}