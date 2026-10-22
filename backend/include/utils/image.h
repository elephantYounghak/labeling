#pragma once

#include <opencv2/core/mat.hpp>

#include <string_view>

struct ImageSize
{
    int width = 0;
    int height = 0;
};

cv::Mat image_from_bytes(std::string_view bytes);
ImageSize image_size(const cv::Mat& image);
