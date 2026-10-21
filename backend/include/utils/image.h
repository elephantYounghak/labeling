#pragma once

#include <opencv2/core/mat.hpp>

#include <string_view>

cv::Mat image_from_bytes(std::string_view bytes);
