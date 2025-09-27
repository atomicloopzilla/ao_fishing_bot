#include "Tools.h"

#include <torch/torch.h>

#if HAVE_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#endif
#include <opencv2/opencv.hpp>

#include "DebugDrawer.h"
#include "GameScreen.h"
#include "FishingBot.h"

void PrintLibTorchInfo()
{
    std::cout << "=== LibTorch Information ===" << std::endl;
    std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "."
        << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

    // Build configuration
    std::cout << "Build configuration:" << std::endl;

    // Backend support
    std::cout << "\nBackend support:" << std::endl;
    std::cout << "  CPU backend: Available" << std::endl;

    // CUDA Support
    std::cout << "\n--- CUDA Support ---" << std::endl;
    if (torch::cuda::is_available())
    {
        std::cout << "✓ CUDA is available" << std::endl;
        std::cout << "  CUDA device count: " << torch::cuda::device_count()
            << std::endl;
        std::cout << "  CUDNN is available: " << torch::cuda::cudnn_is_available()
            << std::endl;

        // Test basic CUDA operation
        try
        {
            torch::Tensor cuda_tensor = torch::ones({2, 2}, torch::device(torch::kCUDA));
            std::cout << "    Basic CUDA tensor creation: ✓ Working" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "    Basic CUDA tensor creation: ✗ Failed (" << e.what()
                << ")" << std::endl;
        }
    }
    else
    {
        std::cout << "✗ CUDA is NOT available" << std::endl;
        std::cout << "  Possible reasons:" << std::endl;
        std::cout << "    - CUDA runtime not installed" << std::endl;
        std::cout << "    - LibTorch built without CUDA support" << std::endl;
        std::cout << "    - No CUDA-capable GPU detected" << std::endl;
        std::cout << "    - CUDA driver version incompatible" << std::endl;
    }

    // cuDNN Support
    std::cout << "\n--- cuDNN Support ---" << std::endl;
    if (torch::cuda::cudnn_is_available())
    {
        std::cout << "✓ cuDNN is available" << std::endl;
        std::cout << "  Benefits:" << std::endl;
        std::cout << "    - Optimized convolution operations" << std::endl;
        std::cout << "    - Faster RNN/LSTM operations" << std::endl;
        std::cout << "    - GPU-accelerated batch normalization" << std::endl;
        std::cout << "    - Optimized activation functions" << std::endl;
    }
    else
    {
        std::cout << "✗ cuDNN is NOT available" << std::endl;
        std::cout << "  Impact: Neural network operations will be slower"
            << std::endl;
    }

    // Compute backends
    std::cout << "\n--- Compute Backends ---" << std::endl;
    std::cout << "Available backends:" << std::endl;
    std::cout << "  CPU: ✓ Always available" << std::endl;
    std::cout << "  CUDA: "
        << (torch::cuda::is_available() ? "✓ Available" : "✗ Not available")
        << std::endl;

    // Performance recommendations
    std::cout << "\n--- Performance Recommendations ---" << std::endl;
    if (torch::cuda::is_available())
    {
        std::cout << "For optimal performance:" << std::endl;
        std::cout << "  1. Use .to(torch::kCUDA) to move tensors to GPU"
            << std::endl;
        std::cout << "  2. Keep data on GPU throughout computation pipeline"
            << std::endl;
        std::cout << "  3. Use larger batch sizes to utilize GPU efficiently"
            << std::endl;
        std::cout << "  4. Enable cuDNN benchmarking for consistent input sizes"
            << std::endl;
        std::cout << "  5. Use mixed precision training when possible" << std::endl;
    }
    else
    {
        std::cout << "CPU-only configuration detected:" << std::endl;
        std::cout << "  1. Use optimized CPU operations (MKL-DNN if available)"
            << std::endl;
        std::cout << "  2. Set number of threads: torch::set_num_threads()"
            << std::endl;
        std::cout << "  3. Consider smaller model architectures" << std::endl;
    }

    // Thread configuration
    std::cout << "\n--- Threading Configuration ---" << std::endl;
    std::cout << "Number of threads: " << torch::get_num_threads() << std::endl;
    std::cout << "Number of inter-op threads: "
        << torch::get_num_interop_threads() << std::endl;

    std::cout << std::endl;
}

void PrintOpenCVInfo()
{
    std::cout << "=== OpenCV Information ===" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Major version: " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version: " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version: " << CV_SUBMINOR_VERSION << std::endl;

    // Build information
    std::cout << "\n--- Build Information ---" << std::endl;
    std::cout << "Build date: " << cv::getBuildInformation() << std::endl;

    // GPU/CUDA Support
    std::cout << "\n--- GPU/CUDA Support ---" << std::endl;
#ifdef HAVE_CUDA
    std::cout << "✓ OpenCV built with CUDA support" << std::endl;

    try
    {
        int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "  CUDA-enabled devices: " << cuda_devices << std::endl;

        if (cuda_devices > 0)
        {
            for (int i = 0; i < cuda_devices; ++i)
            {
                cv::cuda::DeviceInfo dev_info(i);
                std::cout << "\n  Device " << i << ":" << std::endl;
                std::cout << "    Name: " << dev_info.name() << std::endl;
                std::cout << "    Compute capability: " << dev_info.majorVersion()
                    << "." << dev_info.minorVersion() << std::endl;
                std::cout << "    Total memory: "
                    << (dev_info.totalMemory() / (1024 * 1024)) << " MB"
                    << std::endl;
                std::cout << "    Free memory: "
                    << (dev_info.freeMemory() / (1024 * 1024)) << " MB"
                    << std::endl;
                std::cout << "    Multiprocessors: " << dev_info.multiProcessorCount()
                    << std::endl;
                std::cout << "    Max threads per block: "
                    << dev_info.maxThreadsPerBlock() << std::endl;
                std::cout << "    Supports overlap: "
                    << (dev_info.asyncEngineCount() > 0 ? "Yes" : "No")
                    << std::endl;
                std::cout << "    Unified addressing: "
                    << (dev_info.unifiedAddressing() ? "Yes" : "No") << std::endl;
            }

            // Test basic CUDA operation
            try
            {
                cv::cuda::GpuMat gpu_mat;
                cv::Mat cpu_mat = cv::Mat::ones(100, 100, CV_8UC1);
                gpu_mat.upload(cpu_mat);
                std::cout << "\n  Basic CUDA operations: ✓ Working" << std::endl;
            }
            catch (const cv::Exception& e)
            {
                std::cout << "\n  Basic CUDA operations: ✗ Failed (" << e.what() << ")"
                    << std::endl;
            }
        }
    }
    catch (const cv::Exception& e)
    {
        std::cout << "✗ CUDA support available but error accessing devices: "
            << e.what() << std::endl;
    }
#else
    std::cout << "✗ OpenCV built WITHOUT CUDA support" << std::endl;
    std::cout << "  To enable CUDA:" << std::endl;
    std::cout << "    - Rebuild OpenCV with -DWITH_CUDA=ON" << std::endl;
    std::cout << "    - Ensure CUDA toolkit is installed" << std::endl;
    std::cout << "    - Verify GPU compute capability compatibility" << std::endl;
#endif

#ifdef HAVE_OPENCV_DNN
    std::cout << "  dnn: ✓ Available (deep neural networks)" << std::endl;
#else
    std::cout << "  dnn: ✗ Not available" << std::endl;
#endif

#ifdef HAVE_OPENCV_OBJDETECT
    std::cout << "  objdetect: ✓ Available (object detection)" << std::endl;
#else
    std::cout << "  objdetect: ✗ Not available" << std::endl;
#endif

#ifdef HAVE_OPENCV_FEATURES2D
    std::cout << "  features2d: ✓ Available (2D feature detection)" << std::endl;
#else
    std::cout << "  features2d: ✗ Not available" << std::endl;
#endif

    // Performance recommendations
    std::cout << "\n--- Performance Recommendations ---" << std::endl;
    std::cout << "For optimal OpenCV performance:" << std::endl;

#ifdef HAVE_CUDA
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        std::cout << "  1. Use cv::cuda namespace for GPU operations" << std::endl;
        std::cout << "  2. Upload data to GPU once, process multiple times"
            << std::endl;
        std::cout << "  3. Use cv::cuda::Stream for asynchronous operations"
            << std::endl;
        std::cout << "  4. Prefer GPU-compatible data types (CV_32F, CV_8U)"
            << std::endl;
    }
#endif

    std::cout << "  General tips:" << std::endl;
    std::cout << "    - Use appropriate data types to minimize memory usage"
        << std::endl;
    std::cout << "    - Reuse allocated memory when possible" << std::endl;
    std::cout << "    - Use multi-threading with cv::parallel_for_" << std::endl;
    std::cout << "    - Enable compiler optimizations (-O2/-O3)" << std::endl;

    // Threading information
    std::cout << "\n--- Threading Configuration ---" << std::endl;
    std::cout << "Number of threads: " << cv::getNumThreads() << std::endl;

    std::cout << std::endl;
}

struct WindowInfo
{
    HWND hwnd;
    std::string title;
    double similarity;
};

// Callback function for EnumWindows to collect all visible windows
BOOL CALLBACK enumWindowsProc(HWND hwnd, LPARAM lParam)
{
    std::vector<WindowInfo>* windows = reinterpret_cast<std::vector<WindowInfo>*>(lParam);

    if (IsWindowVisible(hwnd))
    {
        char windowText[256];
        GetWindowTextA(hwnd, windowText, sizeof(windowText));

        if (strlen(windowText) > 0)
        {
            WindowInfo info;
            info.hwnd = hwnd;
            info.title = windowText;
            info.similarity = 0.0;
            windows->push_back(info);
        }
    }
    return TRUE;
}

HWND FindGameWindow(const std::string& windowTitle)
{
    // First try exact match for backward compatibility
    HWND exactMatch = FindWindowA(nullptr, windowTitle.c_str());
    if (exactMatch != nullptr)
    {
        return exactMatch;
    }

    // If exact match fails, find closest match
    std::vector<WindowInfo> windows;
    EnumWindows(enumWindowsProc, reinterpret_cast<LPARAM>(&windows));

    if (!windows.empty())
    {
        std::cout << "Select a window to capture:" << std::endl;
        for (size_t i = 0; i < windows.size(); ++i)
        {
            std::cout << "  [" << i << "] " << windows[i].title << std::endl;
        }
        std::cout
            << "Enter the number of the window to capture (or -1 to cancel): ";
        int choice = -1;
        while (true)
        {
            std::cin >> choice;
            if (std::cin.fail() || choice < -1 || choice >= static_cast<int>(windows.size()))
            {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cout << "Invalid input. Please enter a valid number: ";
            }
            else
            {
                break;
            }
        }
        if (choice >= 0 && choice < static_cast<int>(windows.size()))
        {
            return windows[choice].hwnd;
        }
    }

    return nullptr;
}

cv::Mat captureWindow(HWND hwnd)
{
    if (!hwnd)
    {
        std::cerr << "Invalid window handle" << std::endl;
        return cv::Mat();
    }

    RECT windowRect;
    GetClientRect(hwnd, &windowRect);

    int width = windowRect.right - windowRect.left;
    int height = windowRect.bottom - windowRect.top;

    if (width <= 0 || height <= 0)
    {
        std::cerr << "Invalid window dimensions" << std::endl;
        return cv::Mat();
    }

    HDC hdcWindow = GetDC(hwnd);
    if (!hdcWindow)
    {
        std::cerr << "Failed to get window DC" << std::endl;
        return cv::Mat();
    }

    HDC hdcMemDC = CreateCompatibleDC(hdcWindow);
    if (!hdcMemDC)
    {
        std::cerr << "Failed to create memory DC" << std::endl;
        ReleaseDC(hwnd, hdcWindow);
        return cv::Mat();
    }

    HBITMAP hbmScreen = CreateCompatibleBitmap(hdcWindow, width, height);
    if (!hbmScreen)
    {
        std::cerr << "Failed to create bitmap" << std::endl;
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        return cv::Mat();
    }

    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdcMemDC, hbmScreen);

    BOOL bitBltResult = BitBlt(hdcMemDC, 0, 0, width, height, hdcWindow, 0, 0, SRCCOPY);
    if (!bitBltResult)
    {
        std::cerr << "BitBlt failed" << std::endl;
        SelectObject(hdcMemDC, hOldBitmap);
        DeleteObject(hbmScreen);
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        return cv::Mat();
    }

    GdiFlush();

    BITMAPINFOHEADER bi;
    ZeroMemory(&bi, sizeof(BITMAPINFOHEADER));
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = width * height * 4;

    cv::Mat frame(height, width, CV_8UC4);

    int scanLines = GetDIBits(hdcWindow, hbmScreen, 0, height, frame.data,
        (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    if (scanLines == 0)
    {
        std::cerr << "GetDIBits failed" << std::endl;
        SelectObject(hdcMemDC, hOldBitmap);
        DeleteObject(hbmScreen);
        DeleteDC(hdcMemDC);
        ReleaseDC(hwnd, hdcWindow);
        return cv::Mat();
    }

    cv::Mat result;
    cv::cvtColor(frame, result, cv::COLOR_BGRA2BGR);

    SelectObject(hdcMemDC, hOldBitmap);
    DeleteObject(hbmScreen);
    DeleteDC(hdcMemDC);
    ReleaseDC(hwnd, hdcWindow);

    return result;
}

void TestGPUCapabilities()
{
    std::cout << "=== GPU Performance Tests ===" << std::endl;

    // LibTorch GPU test
    std::cout << "\n--- LibTorch GPU Performance Test ---" << std::endl;
    if (torch::cuda::is_available())
    {
        try
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Create large tensors for performance test
            torch::Tensor cpu_tensor = torch::randn({1000, 1000}, torch::kFloat32);
            auto cpu_end = std::chrono::high_resolution_clock::now();

            torch::Tensor gpu_tensor = torch::randn({1000, 1000}, torch::device(torch::kCUDA));
            auto gpu_end = std::chrono::high_resolution_clock::now();

            // Matrix multiplication test
            auto cpu_start = std::chrono::high_resolution_clock::now();
            torch::Tensor cpu_result = torch::mm(cpu_tensor, cpu_tensor);
            auto cpu_mm_end = std::chrono::high_resolution_clock::now();

            auto gpu_start = std::chrono::high_resolution_clock::now();
            torch::Tensor gpu_result = torch::mm(gpu_tensor, gpu_tensor);
            torch::cuda::synchronize(); // Wait for GPU to finish
            auto gpu_mm_end = std::chrono::high_resolution_clock::now();

            auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                cpu_mm_end - cpu_start)
                .count();
            auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                gpu_mm_end - gpu_start)
                .count();

            std::cout << "  Matrix multiplication (1000x1000):" << std::endl;
            std::cout << "    CPU time: " << cpu_time << " microseconds" << std::endl;
            std::cout << "    GPU time: " << gpu_time << " microseconds" << std::endl;
            std::cout << "    Speedup: " << (float)cpu_time / gpu_time << "x"
                << std::endl;

        }
        catch (const std::exception& e)
        {
            std::cout << "  GPU performance test failed: " << e.what() << std::endl;
        }
    }
    else
    {
        std::cout << "  CUDA not available - skipping LibTorch GPU tests"
            << std::endl;
    }

    // OpenCV GPU test
    std::cout << "\n--- OpenCV GPU Performance Test ---" << std::endl;
#ifdef HAVE_CUDA
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        try
        {
            cv::Mat cpu_image = cv::Mat::ones(2000, 2000, CV_8UC3) * 128;
            cv::cuda::GpuMat gpu_image;
            gpu_image.upload(cpu_image);

            cv::Mat cpu_result;
            cv::cuda::GpuMat gpu_result;

            // Gaussian blur test
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cv::GaussianBlur(cpu_image, cpu_result, cv::Size(15, 15), 0);
            auto cpu_end = std::chrono::high_resolution_clock::now();

            auto gpu_start = std::chrono::high_resolution_clock::now();
            // Replace this line:
            // cv::cuda::GaussianBlur(gpu_image, gpu_result, cv::Size(15, 15), 0);

            // With the following code using cv::cuda::createGaussianFilter:
            cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(gpu_image.type(), gpu_result.type(),
                cv::Size(15, 15), 0);
            gaussian_filter->apply(gpu_image, gpu_result);
            cv::cuda::Stream::Null().waitForCompletion();
            auto gpu_end = std::chrono::high_resolution_clock::now();

            auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                cpu_end - cpu_start)
                .count();
            auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
                gpu_end - gpu_start)
                .count();

            std::cout << "  Gaussian Blur (2000x2000 image):" << std::endl;
            std::cout << "    CPU time: " << cpu_time << " microseconds" << std::endl;
            std::cout << "    GPU time: " << gpu_time << " microseconds" << std::endl;
            std::cout << "    Speedup: " << (float)cpu_time / gpu_time << "x"
                << std::endl;

        }
        catch (const cv::Exception& e)
        {
            std::cout << "  OpenCV GPU performance test failed: " << e.what()
                << std::endl;
        }
    }
    else
    {
        std::cout
            << "  CUDA-enabled OpenCV not available - skipping OpenCV GPU tests"
            << std::endl;
    }
#else
    std::cout << "  OpenCV built without CUDA - skipping OpenCV GPU tests"
        << std::endl;
#endif

    // Memory bandwidth test
    std::cout << "\n--- Memory Transfer Performance ---" << std::endl;
    if (torch::cuda::is_available())
    {
        try
        {
            const int size = 100 * 1024 * 1024; // 100MB of float data
            torch::Tensor cpu_data = torch::randn({size}, torch::kFloat32);

            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor gpu_data = cpu_data.to(torch::kCUDA);
            auto end = std::chrono::high_resolution_clock::now();

            auto transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
            float bandwidth = (size * sizeof(float) / (1024.0f * 1024.0f)) / (transfer_time / 1e6f);

            std::cout << "  CPU to GPU transfer (100MB):" << std::endl;
            std::cout << "    Time: " << transfer_time << " microseconds"
                << std::endl;
            std::cout << "    Bandwidth: " << bandwidth << " MB/s" << std::endl;

        }
        catch (const std::exception& e)
        {
            std::cout << "  Memory transfer test failed: " << e.what() << std::endl;
        }
    }

    std::cout << std::endl;
}

void testFrameCapture(std::string windowTitle = "")
{
    std::cout << "=== Frame Capture Test ===" << std::endl;

    HWND targetWindow = nullptr;

    if (!windowTitle.empty())
    {
        std::cout << "Searching for window: \"" << windowTitle << "\"" << std::endl;
        targetWindow = FindGameWindow(windowTitle);

        if (targetWindow)
        {
            std::cout << "Found target window!" << std::endl;
        }
        else
        {
            std::cout << "Window \"" << windowTitle << "\" not found." << std::endl;
            std::cout << "Available windows (first 10):" << std::endl;
            int count = 0;
            EnumWindows(
                [](HWND hwnd, LPARAM lParam) -> BOOL
            {
                int* pCount = (int*)lParam;
                if (*pCount >= 10)
                    return FALSE;

                char windowText[256];
                GetWindowTextA(hwnd, windowText, sizeof(windowText));

                if (strlen(windowText) > 0 && IsWindowVisible(hwnd))
                {
                    std::cout << "  " << windowText << std::endl;
                    (*pCount)++;
                }
                return TRUE;
            },
                (LPARAM)&count);

            std::cout << "Using desktop window instead..." << std::endl;
            targetWindow = GetDesktopWindow();
        }
    }
    else
    {
        std::cout << "No window title specified, capturing desktop..." << std::endl;
        targetWindow = GetDesktopWindow();
    }

    std::cout << "\nCapturing window..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat frame = captureWindow(targetWindow);
    auto end = std::chrono::high_resolution_clock::now();
    auto capture_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
    std::cout << "Capture time: " << capture_time << " ms" << std::endl;

    if (!frame.empty())
    {
        std::cout << "Successfully captured frame: " << frame.cols << "x"
            << frame.rows << std::endl;
        cv::imshow("Captured Frame", frame);
        cv::waitKey(3000);
        cv::destroyAllWindows();
    }
    else
    {
        std::cout << "Failed to capture frame" << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "Fishing Bot" << std::endl;
    std::cout << "===========" << std::endl;

    std::string windowTitle = "";
    if (argc > 1)
    {
        windowTitle = argv[1];
        std::cout << "Window title from command line: \"" << windowTitle << "\"" << std::endl;
    }
    else
    {
        windowTitle = "Albion Online Client";
    }

    bool justCapture = false;
    bool showDebug = false;
    bool workOnlyOverWindow = true;
    // get list of command line arguments
    std::set<std::string> args;
    for (int i = 1; i < argc; i++)
    {
        args.insert(argv[i]);
    }
    
    if (args.find("--justcapture") != args.end())
    {
        justCapture = true;
        std::cout << "Running in just capture mode." << std::endl;
    }
    if (args.find("--showdebug") != args.end())
    {
        showDebug = true;
        std::cout << "Running in show debug mode." << std::endl;
    }
    if (args.find("--annotate") != args.end())
    {
        WaitForFish::s_annotationMode = true;
        WaitForFish::s_maxWaitFishSeconds = 15;
        std::cout << "Running in annotation mode." << std::endl;
    }
    if (args.find("--nowindowcheck") != args.end())
    {
        workOnlyOverWindow = false;
        std::cout << "Disabling window focus check." << std::endl;
    }
    

    HWND hwnd = FindGameWindow(windowTitle);
    if (!hwnd)
    {
        std::cout << "Could not find window with title: \"" << windowTitle << "\""
            << std::endl;
        return 2;
    }
    else
    {
        char foundTitle[256];
        GetWindowTextA(hwnd, foundTitle, sizeof(foundTitle));
        std::cout << "Found window: \"" << foundTitle << "\"" << std::endl;
    }


    PrintLibTorchInfo();
    PrintOpenCVInfo();
    TestGPUCapabilities();

    std::cout << "Starting main loop..." << std::endl;
    
    GameScreen gameScreen(hwnd);
    DebugDrawer debugDrawer;
    AsyncJobsExecutor jobExecutor(3);
    JsonConfig config("config.json");
    FramesStack framesStack(100);
    gameScreen.LoadTemplates();

    FishingState::start();
    
    int32_t frameCount = 0;
    try
    {
        while (true)
        {
            frameCount++;
            if (GetAsyncKeyState(VK_CONTROL) && GetAsyncKeyState(0x43)) // 'C' key
            {
                std::cout << "Ctrl-C detected, exiting..." << std::endl;
                break;
            }
            gameScreen.CaptureScreen();
            if (showDebug || justCapture)
            {
                debugDrawer.DrawFrame(gameScreen.GetFrame());
            }

            if (justCapture)
            {
                std::string filename = "out/screenshot_" + std::to_string(frameCount) + ".png";
                debugDrawer.SaveFrameToAFile(filename);
                std::cout << "Saved screenshot to " << filename << std::endl;
                continue;
            }

            // Only tick the state if mouse cursor is over the target window
            if (!workOnlyOverWindow || gameScreen.IsCursorOverWindow())
            {
                FishingState::current_state_ptr->tick();
            } else
            {
                if (frameCount % 100 == 0)
                {
                    std::cout << "Please move the mouse cursor over the game window." << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds((int32_t)10));
                }
            }

            if (frameCount % 100 == 0)
            {
                std::cout << "Current State: " << FishingState::current_state_ptr->m_name << std::endl;
            }
            if (showDebug)
            {
                debugDrawer.FlushFrame();
            }
            ScopedTimer::PrintAverageTimes(100);
            gameScreen.PrintMaxValues(100);
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "Exception in main loop: " << e.what() << std::endl;
    }

    std::cout << "bb" << std::endl;
    return 0;
}