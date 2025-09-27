#pragma once
#include "Tools.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

class FishingClassifier
{
private:
    torch::jit::script::Module model;

public:
    FishingClassifier(const std::string& model_path);

    float PredictBite(const cv::Mat& image);

private:
    torch::Tensor PreprocessImage(const cv::Mat& image);
};