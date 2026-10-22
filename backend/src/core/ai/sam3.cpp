#include "core/sam3.h"
#include "utils/image.h"

#include <opencv2/opencv.hpp>

#include <stdexcept>

namespace fs = std::filesystem;

namespace sam3 {

namespace {

Ort::Env& ort_env()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam3");
    return env;
}

void require_file(const fs::path& path)
{
    if (!fs::exists(path)) {
        throw std::runtime_error("파일이 없습니다: " + path.string());
    }
}

std::vector<SamPoint> contour_points(const cv::Mat& mask, bool simplify)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {};
    }

    size_t best = 0;
    double best_area = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area > best_area) {
            best_area = area;
            best = i;
        }
    }

    std::vector<cv::Point> shape = contours[best];
    if (simplify) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(shape, approx, 0.01 * cv::arcLength(shape, true), true);
        if (approx.size() >= 3) {
            shape = std::move(approx);
        }
    }

    std::vector<SamPoint> points;
    points.reserve(shape.size());
    for (const cv::Point& point : shape) {
        points.push_back({static_cast<float>(point.x), static_cast<float>(point.y)});
    }
    return points;
}

void append_detection(SamOutput& output, const Prediction& prediction, const std::string& tool)
{
    if (tool == "bbox") {
        output.boxes.push_back({prediction.x1, prediction.y1, prediction.x2, prediction.y2, prediction.score});
        return;
    }
    SamMask mask;
    mask.score = prediction.score;
    mask.points = contour_points(prediction.mask, tool == "polygon");
    if (!mask.points.empty()) {
        output.masks.push_back(std::move(mask));
    }
}

}  // namespace

Ort::SessionOptions cpu_options()
{
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    return options;
}

Ort::SessionOptions gpu_options()
{
    Ort::SessionOptions options = cpu_options();
    OrtCUDAProviderOptions cuda;
    cuda.device_id = 0;
    options.AppendExecutionProvider_CUDA(cuda);
    return options;
}

std::unique_ptr<Ort::Session> open_session(const fs::path& path, const Ort::SessionOptions& options)
{
    require_file(path);
#ifdef _WIN32
    return std::make_unique<Ort::Session>(ort_env(), path.wstring().c_str(), options);
#else
    return std::make_unique<Ort::Session>(ort_env(), path.string().c_str(), options);
#endif
}

std::string normalized_tool(const std::string& tool)
{
    if (tool == "bbox" || tool == "polygon" || tool == "brush") {
        return tool;
    }
    throw std::runtime_error("도구는 bbox, polygon, brush 중 하나여야 합니다.");
}

cv::Mat input_rgb(const SamInput& input)
{
    if (input.image.empty()) {
        throw std::runtime_error("이미지가 없습니다.");
    }
    cv::Mat rgb;
    cv::cvtColor(input.image, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

std::vector<uint8_t> rgb_to_chw1008(const cv::Mat& rgb)
{
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(kImageSize, kImageSize), 0, 0, cv::INTER_LINEAR);

    std::vector<uint8_t> chw(3 * kImageSize * kImageSize);
    for (int y = 0; y < kImageSize; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kImageSize; ++x) {
            chw[0 * kImageSize * kImageSize + y * kImageSize + x] = row[x][0];
            chw[1 * kImageSize * kImageSize + y * kImageSize + x] = row[x][1];
            chw[2 * kImageSize * kImageSize + y * kImageSize + x] = row[x][2];
        }
    }
    return chw;
}

Ort::Value view_float(const Ort::MemoryInfo& memory, Ort::Value& source)
{
    const auto info = source.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    return Ort::Value::CreateTensor<float>(
        memory,
        source.GetTensorMutableData<float>(),
        info.GetElementCount(),
        shape.data(),
        shape.size());
}

Ort::Value view_bool(const Ort::MemoryInfo& memory, Ort::Value& source)
{
    const auto info = source.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    return Ort::Value::CreateTensor<bool>(
        memory,
        source.GetTensorMutableData<bool>(),
        info.GetElementCount(),
        shape.data(),
        shape.size());
}

SamOutput to_output(const std::vector<Prediction>& predictions, const std::string& tool)
{
    SamOutput output;
    for (const Prediction& prediction : predictions) {
        append_detection(output, prediction, tool);
    }
    return output;
}

}  // namespace sam3

SamInput make_sam_input(std::string_view bytes, float x, float y, const std::string& tool)
{
    SamInput input;
    input.image = image_from_bytes(bytes);
    input.x = x;
    input.y = y;
    input.tool = tool;
    return input;
}

Json::Value sam_output_json(const SamOutput& output)
{
    Json::Value result(Json::objectValue);
    Json::Value boxes(Json::arrayValue);
    for (const SamBox& box : output.boxes) {
        Json::Value item(Json::objectValue);
        item["x1"] = box.x1;
        item["y1"] = box.y1;
        item["x2"] = box.x2;
        item["y2"] = box.y2;
        item["score"] = box.score;
        boxes.append(std::move(item));
    }
    Json::Value masks(Json::arrayValue);
    for (const SamMask& mask : output.masks) {
        Json::Value points(Json::arrayValue);
        for (const SamPoint& point : mask.points) {
            Json::Value row(Json::objectValue);
            row["x"] = point.x;
            row["y"] = point.y;
            points.append(std::move(row));
        }
        Json::Value item(Json::objectValue);
        item["points"] = std::move(points);
        item["score"] = mask.score;
        masks.append(std::move(item));
    }
    result["boxes"] = std::move(boxes);
    result["masks"] = std::move(masks);
    return result;
}
