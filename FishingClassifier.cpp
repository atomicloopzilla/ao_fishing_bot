#include "FishingClassifier.h"

FishingClassifier::FishingClassifier(const std::string& model_path)
{
    try
    {
        // Load the traced model
        model = torch::jit::load(model_path);
        model.eval();
        std::cout << "Model loaded successfully!" << std::endl;
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error loading model: " << e.msg() << std::endl;
        throw;
    }
}

float FishingClassifier::PredictBite(const cv::Mat& image)
{
    ScopedTimer timer("FishingClassifier::PredictBite");
    torch::Tensor tensor = PreprocessImage(image.clone());

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    try
    {
        at::Tensor output = model.forward(inputs).toTensor();
        return output.item<float>();
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error during model inference: " << e.msg() << std::endl;
        return 0.0f;
    }
    catch (...)
    {
        std::cerr << "Unknown error during model inference." << std::endl;
        return 0.0f;
    }
}


torch::Tensor FishingClassifier::PreprocessImage(const cv::Mat& image)
{
    // Resize to 100x100
    cv::Mat resized;
    if (image.cols != 100 || image.rows != 100)
    {
        cv::resize(image, resized, cv::Size(100, 100));
    }
    else
    {
        resized = image;
    }

    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize to [0, 1]
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // Convert to tensor [1, 3, 100, 100]
    torch::Tensor tensor = torch::from_blob(rgb.data, {100, 100, 3}, torch::kFloat);
    tensor = tensor.permute({2, 0, 1}); // HWC to CHW
    tensor = tensor.unsqueeze(0); // Add batch dimension

    // ImageNet normalization (same as training)
    tensor[0][0] = (tensor[0][0] - 0.485) / 0.229; // R channel
    tensor[0][1] = (tensor[0][1] - 0.456) / 0.224; // G channel
    tensor[0][2] = (tensor[0][2] - 0.406) / 0.225; // B channel

    return tensor;
}