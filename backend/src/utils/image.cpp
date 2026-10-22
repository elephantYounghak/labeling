#include "utils/image.h"

#include <opencv2/imgcodecs.hpp>

#include <stdexcept>
#include <vector>

cv::Mat image_from_bytes(std::string_view bytes)
{
    if (bytes.empty()) {
        throw std::runtime_error("이미지가 없습니다.");
    }

    const std::vector<unsigned char> encoded(bytes.begin(), bytes.end());
    cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("이미지를 열지 못했습니다.");
    }
    return image;
}

ImageSize image_size(const cv::Mat& image)
{
    if (image.empty()) {
        throw std::runtime_error("이미지가 없습니다.");
    }
    return {image.cols, image.rows};
}
